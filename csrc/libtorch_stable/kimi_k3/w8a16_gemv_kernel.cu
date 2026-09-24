// Kimi-K3 decode: MXFP8-weight x bf16-activation GEMV for at most 8 rows,
// reading Marlin's packed MXFP8 payload in place (no second weight copy).
//
// Payload layout (gptq_marlin_repack, 8-bit; decoded the same way by
// `_marlin_fp8_unrepack_dequant_kernel` in vllm/model_executor/kernels/
// linear/mxfp8/marlin_hybrid.py):
//   weight bytes [padded_k / 16][padded_n / 64][1024]: within a 16 x 64 tile,
//     element (column n < 64, row k < 16) is byte
//     (k >> 3) | (k & 1) << 1 | (n >> 3) << 2 | (k & 6) << 4 | (n & 7) << 7;
//   e8m0 scale bytes [padded_k / 32][padded_n]: the scale of column n of a
//     64-column group is byte
//     (n & 16) >> 4 | (n & 8) >> 2 | (n & 32) >> 3 | (n & 7) << 3.
// Lane l of a warp reads the 16-byte chunks l and l + 32 (h = 0, 1) of every
// tile. Chunk byte j holds column (l >> 3 & 3) | h << 2 | (j >> 2) << 3 |
// (l & 1) << 5 and row k0 + {0, 8, 1, 9}[j & 3] with k0 = 2 * (l >> 1 & 3):
// every lane owns 8 fixed columns and 4 rows of each tile, and the 4 scales
// of a chunk's columns form one 32-bit word.
//
// Arithmetic: e4m3 -> f16 -> f32 is exact and every product w * x is exact
// in fp32 (4 x 8 significant bits). Per column and row, each lane sums the 8
// products of a 32-row scale group as a pairwise tree, multiplies by the
// group's power-of-two scale (exact; Marlin's bf16 scale value 2^(e-127),
// e = 0 -> 0) and accumulates with one rounding per group. Lanes, warps and
// the CTAs of a K-split cluster then reduce in fixed trees (distributed
// shared memory, no workspace, no atomics): the result is deterministic.
// Marlin's bf16 MMA path computes the same exact products with tensor-core
// fp32 accumulation in another order.
//
// PDL: before `griddepcontrol.wait` the CTA prefetches its weight tiles and
// scales into L2 and loads its first scale group into registers; the
// activations are read after the wait.
#include "../torch_utils.h"

#include <torch/csrc/stable/library.h>
#include <torch/headeronly/core/ScalarType.h>

#include <cooperative_groups.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>

#include <cstdint>

namespace {

namespace cg = cooperative_groups;

constexpr int kTileN = 64;   // columns per Marlin tile
constexpr int kTileK = 16;   // rows per Marlin tile
constexpr int kGroupK = 32;  // rows per MXFP8 scale group (two tiles)
constexpr int kTileBytes = 1024;

struct GemvParams {
  const uint8_t* weight;  // packed payload
  const uint8_t* scales;  // e8m0 bytes, Marlin layout
  const __nv_bfloat16* x;
  __nv_bfloat16* out;
  int64_t stride_x;
  int64_t stride_out;
  int rows;
  int size_n;
  int size_k;
  int padded_n;
  int groups;  // padded_k / 32
};

__device__ __forceinline__ void griddep_wait() {
  asm volatile("griddepcontrol.wait;" ::: "memory");
}

__device__ __forceinline__ void griddep_launch_dependents() {
  asm volatile("griddepcontrol.launch_dependents;" :::);
}

__device__ __forceinline__ void prefetch_l2(const void* ptr, uint32_t bytes) {
  asm volatile("cp.async.bulk.prefetch.L2.global [%0], %1;" ::"l"(ptr),
               "r"(bytes)
               : "memory");
}

// Two e4m3 values -> two floats, exactly (via f16).
__device__ __forceinline__ float2 e4m3x2_to_float2(uint16_t v) {
  const __half2_raw h = __nv_cvt_fp8x2_to_halfraw2(
      static_cast<__nv_fp8x2_storage_t>(v), __NV_E4M3);
  return __half22float2(*reinterpret_cast<const __half2*>(&h));
}

// e8m0 byte -> 2^(e - 127) as Marlin's bf16 scale (e = 0 -> +0).
__device__ __forceinline__ float e8m0_to_float(uint32_t e) {
  return __uint_as_float(e << 23);
}

__device__ __forceinline__ float2 bf16x2_to_float2(uint32_t v) {
  return make_float2(__uint_as_float(v << 16),
                     __uint_as_float(v & 0xffff0000u));
}

// Pairwise sum of four exact products: (x0*w0 + x1*w1) + (x8*w8 + x9*w9).
__device__ __forceinline__ float dot4(float2 xa, float2 xb, float2 w08,
                                      float2 w19) {
  // xa = (x[k0], x[k0 + 1]), xb = (x[k0 + 8], x[k0 + 9]);
  // w08 = (w[k0], w[k0 + 8]), w19 = (w[k0 + 1], w[k0 + 9]).
  const float lo = __fmaf_rn(xa.y, w19.x, __fmul_rn(xa.x, w08.x));
  const float hi = __fmaf_rn(xb.y, w19.y, __fmul_rn(xb.x, w08.y));
  return __fadd_rn(lo, hi);
}

template <int CL>
__device__ __forceinline__ float cluster_tree(const float (&r)[CL]) {
  if constexpr (CL == 1) {
    return r[0];
  } else if constexpr (CL == 2) {
    return __fadd_rn(r[0], r[1]);
  } else if constexpr (CL == 4) {
    return __fadd_rn(__fadd_rn(r[0], r[1]), __fadd_rn(r[2], r[3]));
  } else {
    static_assert(CL == 8, "cluster size 1, 2, 4 or 8");
    return __fadd_rn(__fadd_rn(__fadd_rn(r[0], r[1]), __fadd_rn(r[2], r[3])),
                     __fadd_rn(__fadd_rn(r[4], r[5]), __fadd_rn(r[6], r[7])));
  }
}

template <int NW>
__device__ __forceinline__ float warp_tree(const float* v, int stride) {
  if constexpr (NW == 1) {
    return v[0];
  } else if constexpr (NW == 2) {
    return __fadd_rn(v[0], v[stride]);
  } else if constexpr (NW == 4) {
    return __fadd_rn(__fadd_rn(v[0], v[stride]),
                     __fadd_rn(v[2 * stride], v[3 * stride]));
  } else {
    static_assert(NW == 8, "warps 1, 2, 4 or 8");
    return __fadd_rn(__fadd_rn(__fadd_rn(v[0], v[stride]),
                               __fadd_rn(v[2 * stride], v[3 * stride])),
                     __fadd_rn(__fadd_rn(v[4 * stride], v[5 * stride]),
                               __fadd_rn(v[6 * stride], v[7 * stride])));
  }
}

// MR: padded row count (1, 2, 4, 8). NW: warps per CTA. CL: CTAs per
// 64-column group (the K split, one thread-block cluster).
template <int MR, int NW, int CL>
__global__ void __launch_bounds__(NW * 32)
    w8a16_gemv_kernel(const GemvParams p) {
  extern __shared__ __align__(16) unsigned char smem_raw[];
  const int lane = threadIdx.x & 31;
  const int warp = threadIdx.x >> 5;
  const int rank = CL == 1 ? 0 : static_cast<int>(blockIdx.x % CL);
  const int nc = blockIdx.x / CL;

  // This CTA's scale groups [g_begin, g_end).
  const int g_begin =
      static_cast<int>(static_cast<int64_t>(p.groups) * rank / CL);
  const int g_end =
      static_cast<int>(static_cast<int64_t>(p.groups) * (rank + 1) / CL);
  const int slice_k = (g_end - g_begin) * kGroupK;

  // Shared memory: activations [MR][slice_k] bf16, then the per-warp and
  // per-CTA column sums.
  __nv_bfloat16* x_s = reinterpret_cast<__nv_bfloat16*>(smem_raw);
  float* s_red = reinterpret_cast<float*>(
      smem_raw + ((static_cast<size_t>(MR) * slice_k * 2 + 15) & ~size_t{15}));
  float* s_cta = s_red + NW * kTileN * MR;  // [kTileN][MR], read cluster-wide

  const int64_t row_bytes = static_cast<int64_t>(p.padded_n) * kTileK;
  const uint8_t* tile_col = p.weight + static_cast<int64_t>(nc) * kTileBytes;
  const uint8_t* scale_col = p.scales + static_cast<int64_t>(nc) * kTileN;

  // ---- before the dependency: weights and scales only ----
  {
    const int tiles = 2 * (g_end - g_begin);
    for (int t = threadIdx.x; t < tiles; t += NW * 32) {
      prefetch_l2(tile_col + static_cast<int64_t>(2 * g_begin + t) * row_bytes,
                  kTileBytes);
    }
    for (int g = g_begin + threadIdx.x; g < g_end; g += NW * 32) {
      prefetch_l2(scale_col + static_cast<int64_t>(g) * p.padded_n, kTileN);
    }
  }

  const int l0 = lane & 1;
  const int l12 = (lane >> 1) & 3;
  const int l34 = (lane >> 3) & 3;
  const int k0 = 2 * l12;
  // Scale word offsets of the lane's two chunks (h = 0, 1).
  const int sw0 = (l0 << 2) | (l34 << 3);
  const int sw1 = (l0 << 2) | ((l34 | 4) << 3);

  auto load_group = [&](int g, uint4(&w)[2][2], uint32_t (&s)[2]) {
    const uint8_t* t0 = tile_col + static_cast<int64_t>(2 * g) * row_bytes;
    const uint8_t* t1 = t0 + row_bytes;
    w[0][0] = reinterpret_cast<const uint4*>(t0)[lane];
    w[0][1] = reinterpret_cast<const uint4*>(t0)[lane + 32];
    w[1][0] = reinterpret_cast<const uint4*>(t1)[lane];
    w[1][1] = reinterpret_cast<const uint4*>(t1)[lane + 32];
    const uint8_t* srow = scale_col + static_cast<int64_t>(g) * p.padded_n;
    s[0] = *reinterpret_cast<const uint32_t*>(srow + sw0);
    s[1] = *reinterpret_cast<const uint32_t*>(srow + sw1);
  };

  uint4 w_cur[2][2];
  uint32_t s_cur[2];
  int g = g_begin + warp;
  if (g < g_end) load_group(g, w_cur, s_cur);

  griddep_wait();

  // ---- activations of this CTA's K slice ----
  {
    const int vec_per_row = slice_k / 8;
    const int k_base = g_begin * kGroupK;
    for (int i = threadIdx.x; i < MR * vec_per_row; i += NW * 32) {
      const int m = i / vec_per_row;
      const int v = i - m * vec_per_row;
      const int k = k_base + v * 8;
      uint4 val = make_uint4(0u, 0u, 0u, 0u);
      if (m < p.rows && k < p.size_k) {
        if (k + 8 <= p.size_k) {
          val = *reinterpret_cast<const uint4*>(p.x + m * p.stride_x + k);
        } else {
          __nv_bfloat16 tmp[8];
#pragma unroll
          for (int e = 0; e < 8; ++e) {
            tmp[e] = k + e < p.size_k ? p.x[m * p.stride_x + k + e]
                                      : __float2bfloat16_rn(0.f);
          }
          val = *reinterpret_cast<const uint4*>(tmp);
        }
      }
      reinterpret_cast<uint4*>(x_s + static_cast<int64_t>(m) * slice_k)[v] =
          val;
    }
  }
  __syncthreads();

  // ---- main loop: warp w takes groups g_begin + w, + NW, ... ----
  float acc[2][4][MR];
#pragma unroll
  for (int h = 0; h < 2; ++h)
#pragma unroll
    for (int q = 0; q < 4; ++q)
#pragma unroll
      for (int m = 0; m < MR; ++m) acc[h][q][m] = 0.f;

  for (; g < g_end; g += NW) {
    uint4 w_next[2][2];
    uint32_t s_next[2];
    const bool has_next = g + NW < g_end;
    if (has_next) load_group(g + NW, w_next, s_next);

    const int kl = (g - g_begin) * kGroupK + k0;  // local row of (tile 0, k0)
    // Per row: x pairs (k0, k0 + 1), (k0 + 8, k0 + 9) of both tiles.
    float2 xa[2][MR], xb[2][MR];
#pragma unroll
    for (int t = 0; t < 2; ++t)
#pragma unroll
      for (int m = 0; m < MR; ++m) {
        const uint32_t* xr = reinterpret_cast<const uint32_t*>(
            x_s + static_cast<int64_t>(m) * slice_k + kl + t * kTileK);
        xa[t][m] = bf16x2_to_float2(xr[0]);
        xb[t][m] = bf16x2_to_float2(xr[4]);
      }

#pragma unroll
    for (int h = 0; h < 2; ++h) {
      const uint32_t* w0 = reinterpret_cast<const uint32_t*>(&w_cur[0][h]);
      const uint32_t* w1 = reinterpret_cast<const uint32_t*>(&w_cur[1][h]);
#pragma unroll
      for (int q = 0; q < 4; ++q) {
        // Word bytes: (k0, k0 + 8, k0 + 1, k0 + 9).
        const float2 a08 = e4m3x2_to_float2(static_cast<uint16_t>(w0[q]));
        const float2 a19 = e4m3x2_to_float2(static_cast<uint16_t>(w0[q] >> 16));
        const float2 b08 = e4m3x2_to_float2(static_cast<uint16_t>(w1[q]));
        const float2 b19 = e4m3x2_to_float2(static_cast<uint16_t>(w1[q] >> 16));
        // Scale byte of column q within the word: q = 0, 1, 2, 3 -> 0, 2, 1, 3.
        const int sb = ((q >> 1) | ((q & 1) << 1)) * 8;
        const float scale = e8m0_to_float((s_cur[h] >> sb) & 0xffu);
#pragma unroll
        for (int m = 0; m < MR; ++m) {
          const float part = __fadd_rn(dot4(xa[0][m], xb[0][m], a08, a19),
                                       dot4(xa[1][m], xb[1][m], b08, b19));
          acc[h][q][m] = __fmaf_rn(part, scale, acc[h][q][m]);
        }
      }
    }

    if (has_next) {
#pragma unroll
      for (int t = 0; t < 2; ++t)
#pragma unroll
        for (int h = 0; h < 2; ++h) w_cur[t][h] = w_next[t][h];
      s_cur[0] = s_next[0];
      s_cur[1] = s_next[1];
    }
  }

  griddep_launch_dependents();

  // ---- reductions: lanes (rows k0), warps, cluster CTAs ----
#pragma unroll
  for (int h = 0; h < 2; ++h)
#pragma unroll
    for (int q = 0; q < 4; ++q)
#pragma unroll
      for (int m = 0; m < MR; ++m) {
        float v = acc[h][q][m];
        v = __fadd_rn(v, __shfl_xor_sync(0xffffffffu, v, 2));
        v = __fadd_rn(v, __shfl_xor_sync(0xffffffffu, v, 4));
        acc[h][q][m] = v;
      }
  if (l12 == 0) {
#pragma unroll
    for (int h = 0; h < 2; ++h)
#pragma unroll
      for (int q = 0; q < 4; ++q) {
        const int n = l34 | (h << 2) | (q << 3) | (l0 << 5);
#pragma unroll
        for (int m = 0; m < MR; ++m)
          s_red[(warp * kTileN + n) * MR + m] = acc[h][q][m];
      }
  }
  __syncthreads();
  for (int i = threadIdx.x; i < kTileN * MR; i += NW * 32) {
    s_cta[i] = warp_tree<NW>(s_red + i, kTileN * MR);
  }

  if constexpr (CL == 1) {
    __syncthreads();
    for (int i = threadIdx.x; i < kTileN * MR; i += NW * 32) {
      const int n = i / MR, m = i - (i / MR) * MR;
      const int col = nc * kTileN + n;
      if (m < p.rows && col < p.size_n)
        p.out[m * p.stride_out + col] = __float2bfloat16_rn(s_cta[i]);
    }
  } else {
    cg::cluster_group cluster = cg::this_cluster();
    cluster.sync();
    // CTA `rank` finishes columns [rank * 64 / CL, (rank + 1) * 64 / CL).
    constexpr int kCols = kTileN / CL;
    for (int i = threadIdx.x; i < kCols * MR; i += NW * 32) {
      const int idx = rank * kCols * MR + i;
      float r[CL];
#pragma unroll
      for (int c = 0; c < CL; ++c)
        r[c] = *cluster.map_shared_rank(s_cta + idx, c);
      const float total = cluster_tree<CL>(r);
      const int n = idx / MR, m = idx - (idx / MR) * MR;
      const int col = nc * kTileN + n;
      if (m < p.rows && col < p.size_n)
        p.out[m * p.stride_out + col] = __float2bfloat16_rn(total);
    }
    // No CTA may exit while another can still read its shared memory.
    cluster.sync();
  }
}

// ---------------------------------------------------------------------------
using KernelFn = void (*)(const GemvParams);

template <int MR, int NW>
KernelFn select_cluster(int cl) {
  switch (cl) {
    case 1:
      return w8a16_gemv_kernel<MR, NW, 1>;
    case 2:
      return w8a16_gemv_kernel<MR, NW, 2>;
    case 4:
      return w8a16_gemv_kernel<MR, NW, 4>;
    case 8:
      return w8a16_gemv_kernel<MR, NW, 8>;
    default:
      return nullptr;
  }
}

template <int MR>
KernelFn select_warps(int nw, int cl) {
  switch (nw) {
    case 2:
      return select_cluster<MR, 2>(cl);
    case 4:
      return select_cluster<MR, 4>(cl);
    case 8:
      return select_cluster<MR, 8>(cl);
    default:
      return nullptr;
  }
}

KernelFn select_kernel(int mr, int nw, int cl) {
  switch (mr) {
    case 1:
      return select_warps<1>(nw, cl);
    case 2:
      return select_warps<2>(nw, cl);
    case 4:
      return select_warps<4>(nw, cl);
    case 8:
      return select_warps<8>(nw, cl);
    default:
      return nullptr;
  }
}

void w8a16_gemv(torch::stable::Tensor& out, torch::stable::Tensor const& x,
                torch::stable::Tensor const& weight,
                torch::stable::Tensor const& scales, int64_t size_n,
                int64_t size_k, int64_t cluster, int64_t warps, bool pdl) {
  using torch::headeronly::ScalarType;
  STD_TORCH_CHECK(x.dim() == 2 && out.dim() == 2, "w8a16_gemv: 2-D x and out");
  const int64_t rows = x.size(0);
  STD_TORCH_CHECK(rows >= 1 && rows <= 8, "w8a16_gemv: 1-8 rows");
  STD_TORCH_CHECK(x.scalar_type() == ScalarType::BFloat16 &&
                      out.scalar_type() == ScalarType::BFloat16,
                  "w8a16_gemv: bf16 x and out");
  STD_TORCH_CHECK(
      x.size(1) == size_k && out.size(0) == rows && out.size(1) == size_n,
      "w8a16_gemv: shape mismatch");
  STD_TORCH_CHECK(x.stride(1) == 1 && out.stride(1) == 1 &&
                      x.stride(0) % 8 == 0 &&
                      reinterpret_cast<uintptr_t>(x.data_ptr()) % 16 == 0,
                  "w8a16_gemv: x rows must be 16-byte aligned, unit stride");
  STD_TORCH_CHECK(weight.scalar_type() == ScalarType::Int &&
                      weight.dim() == 2 && weight.is_contiguous(),
                  "w8a16_gemv: packed int32 Marlin weight");
  STD_TORCH_CHECK(
      scales.element_size() == 1 && scales.dim() == 2 && scales.is_contiguous(),
      "w8a16_gemv: one-byte e8m0 Marlin scales");
  const int64_t padded_k = weight.size(0) * kTileK;
  const int64_t padded_n = weight.size(1) * 4 / kTileK;
  STD_TORCH_CHECK(padded_n % kTileN == 0 && padded_k % kGroupK == 0 &&
                      size_n <= padded_n && size_k <= padded_k,
                  "w8a16_gemv: Marlin padding");
  STD_TORCH_CHECK(
      scales.size(0) == padded_k / kGroupK && scales.size(1) == padded_n,
      "w8a16_gemv: scale shape");
  STD_TORCH_CHECK(cluster == 1 || cluster == 2 || cluster == 4 || cluster == 8,
                  "w8a16_gemv: cluster 1, 2, 4 or 8");
  STD_TORCH_CHECK(warps == 2 || warps == 4 || warps == 8,
                  "w8a16_gemv: warps 2, 4 or 8");

  GemvParams p;
  p.weight = static_cast<const uint8_t*>(weight.data_ptr());
  p.scales = static_cast<const uint8_t*>(scales.data_ptr());
  p.x = static_cast<const __nv_bfloat16*>(x.data_ptr());
  p.out = static_cast<__nv_bfloat16*>(out.data_ptr());
  p.stride_x = x.stride(0);
  p.stride_out = out.stride(0);
  p.rows = static_cast<int>(rows);
  p.size_n = static_cast<int>(size_n);
  p.size_k = static_cast<int>(size_k);
  p.padded_n = static_cast<int>(padded_n);
  p.groups = static_cast<int>(padded_k / kGroupK);

  const int mr = rows <= 1 ? 1 : rows <= 2 ? 2 : rows <= 4 ? 4 : 8;
  const int max_slice_groups =
      (p.groups + static_cast<int>(cluster) - 1) / static_cast<int>(cluster);
  const size_t x_bytes =
      (static_cast<size_t>(mr) * max_slice_groups * kGroupK * 2 + 15) &
      ~size_t{15};
  const size_t smem =
      x_bytes + static_cast<size_t>(warps + 1) * kTileN * mr * 4;
  KernelFn kernel =
      select_kernel(mr, static_cast<int>(warps), static_cast<int>(cluster));
  STD_TORCH_CHECK(kernel != nullptr, "w8a16_gemv: no kernel");
  if (smem > 48 * 1024) {
    const cudaError_t e = cudaFuncSetAttribute(
        kernel, cudaFuncAttributeMaxDynamicSharedMemorySize,
        static_cast<int>(smem));
    STD_TORCH_CHECK(e == cudaSuccess, "w8a16_gemv: shared memory ",
                    cudaGetErrorString(e));
  }

  const torch::stable::accelerator::DeviceGuard device_guard(
      x.get_device_index());
  cudaLaunchConfig_t cfg = {};
  cfg.gridDim = dim3(static_cast<unsigned>(padded_n / kTileN * cluster));
  cfg.blockDim = dim3(static_cast<unsigned>(warps * 32));
  cfg.dynamicSmemBytes = smem;
  cfg.stream = get_current_cuda_stream();
  cudaLaunchAttribute attr[2];
  attr[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
  attr[0].val.programmaticStreamSerializationAllowed = pdl ? 1 : 0;
  attr[1].id = cudaLaunchAttributeClusterDimension;
  attr[1].val.clusterDim.x = static_cast<unsigned>(cluster);
  attr[1].val.clusterDim.y = 1;
  attr[1].val.clusterDim.z = 1;
  cfg.attrs = attr;
  cfg.numAttrs = cluster == 1 ? 1 : 2;
  const cudaError_t err = cudaLaunchKernelEx(&cfg, kernel, p);
  STD_TORCH_CHECK(err == cudaSuccess,
                  "w8a16_gemv launch failed: ", cudaGetErrorString(err));
}

}  // namespace

STABLE_TORCH_LIBRARY_FRAGMENT(_C_k3decode, k3decode_ops) {
  k3decode_ops.def(
      "w8a16_gemv(Tensor! out, Tensor x, Tensor weight, Tensor scales, "
      "int size_n, int size_k, int cluster, int warps, bool pdl) -> ()");
}

STABLE_TORCH_LIBRARY_IMPL(_C_k3decode, CUDA, k3decode_ops) {
  k3decode_ops.impl("w8a16_gemv", TORCH_BOX(&w8a16_gemv));
}
