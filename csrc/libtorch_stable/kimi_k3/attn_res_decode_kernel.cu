// Kimi-K3 AttnRes mixing for decode-sized batches.
//
// Semantics of vLLM's Triton `_attn_res_kernel` (vllm/models/kimi_k3/nvidia/
// ops/attn_res.py) for bf16 operands:
//   p = prefix;  if delta: p = bf16(p + delta), stored back to prefix
//   if block_write_idx >= 0: blocks[row, block_write_idx] = p
//   sources s = blocks[row, 0 .. num_blocks-1], then p (last)
//   logit_s = (sum_d v_sd * (n_d * q_d)) * rsqrt(sum_d v_sd^2 / hidden + eps)
//   mixed = softmax(logit) . v
//   out = bf16(mixed * rsqrt(sum_d mixed_d^2 / hidden + oeps) * o_d)  (or
//   mixed)
//
// Arithmetic differences, all in the direction of fewer or smaller roundings:
//   * every product v*v and v*(n*q) is exact in fp32 (bf16 x bf16 and
//     bf16 x (bf16*bf16)); each thread sums the 8 products of a 16-byte
//     vector as a pairwise tree (the Triton kernel accumulates 32 products
//     per thread sequentially), then warps, the CTA and the cluster reduce in
//     fixed trees;
//   * logits, softmax weights and the output-norm scale use correctly rounded
//     fp32 division and square root and the accurate expf (the Triton kernel
//     uses the approximate rsqrt and exp2 instructions), with the exact
//     maximum in a single pass (the Triton kernel rescales an online softmax
//     per tile of four sources);
//   * the mix accumulates the sources in order with one fma each.
// The prefix update and the block write are the same bf16 values.
//
// Layouts (`cluster`):
//   1        one CTA per row; each thread owns `slots` 16-byte vectors and
//            reads the block sources twice (statistics, then the mix);
//   2, 4, 8  a thread-block cluster per row; each thread owns one vector of
//            every source in registers, so every source is read once and the
//            row streams through `cluster` SMs. Per-CTA partial sums are
//            exchanged through distributed shared memory; every CTA evaluates
//            the same softmax from the same sums in the same order.
//
// PDL: the constant weights are loaded before `griddepcontrol.wait`; the
// dependent grid is released once every dependent load has been issued.
#include "../torch_utils.h"

#include <torch/csrc/stable/library.h>
#include <torch/headeronly/core/ScalarType.h>

#include <cooperative_groups.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <cstdint>

namespace {

namespace cg = cooperative_groups;

constexpr int kVec = 8;  // bf16 values per 16-byte vector
constexpr int kMaxSources = 12;
constexpr int kMaxWarps = 32;
constexpr int kClusterThreads = 512;

struct AttnResParams {
  __nv_bfloat16* prefix;
  const __nv_bfloat16* delta;
  __nv_bfloat16* blocks;
  const __nv_bfloat16* norm_w;
  const __nv_bfloat16* qk_w;
  const __nv_bfloat16* onorm_w;
  __nv_bfloat16* out;
  int64_t stride_prefix;
  int64_t stride_delta;
  int64_t stride_block_m;
  int64_t stride_block_r;
  int64_t stride_out;
  int hidden;
  int block_write_idx;
  float eps;
  float oeps;
  bool has_delta;
  bool apply_norm;
};

__device__ __forceinline__ void griddep_wait() {
  asm volatile("griddepcontrol.wait;" ::: "memory");
}

__device__ __forceinline__ void griddep_launch_dependents() {
  asm volatile("griddepcontrol.launch_dependents;" :::);
}

// Sum of the 8 products x_i * y_i of one vector as a fixed pairwise tree.
// For the bf16 statistics every product is exact in fp32, so the first level
// fma(x1, y1, x0 * y0) rounds the exact pair sum once.
__device__ __forceinline__ float dot8_pairwise(const float (&x)[kVec],
                                               const float (&y)[kVec]) {
  const float s01 = __fmaf_rn(x[1], y[1], __fmul_rn(x[0], y[0]));
  const float s23 = __fmaf_rn(x[3], y[3], __fmul_rn(x[2], y[2]));
  const float s45 = __fmaf_rn(x[5], y[5], __fmul_rn(x[4], y[4]));
  const float s67 = __fmaf_rn(x[7], y[7], __fmul_rn(x[6], y[6]));
  return __fadd_rn(__fadd_rn(s01, s23), __fadd_rn(s45, s67));
}

// Butterfly sum: fp32 addition is commutative, so every lane ends with the
// same value in a fixed order.
__device__ __forceinline__ float warp_sum(float v) {
#pragma unroll
  for (int off = 16; off > 0; off >>= 1) {
    v = __fadd_rn(v, __shfl_xor_sync(0xffffffffu, v, off));
  }
  return v;
}

__device__ __forceinline__ float warp_max(float v) {
#pragma unroll
  for (int off = 16; off > 0; off >>= 1) {
    v = fmaxf(v, __shfl_xor_sync(0xffffffffu, v, off));
  }
  return v;
}

__device__ __forceinline__ void bf16x8_to_float(const uint4& v,
                                                float (&f)[kVec]) {
  const __nv_bfloat162* h = reinterpret_cast<const __nv_bfloat162*>(&v);
#pragma unroll
  for (int i = 0; i < kVec / 2; ++i) {
    const float2 t = __bfloat1622float2(h[i]);
    f[2 * i] = t.x;
    f[2 * i + 1] = t.y;
  }
}

__device__ __forceinline__ uint4 float_to_bf16x8(const float (&f)[kVec]) {
  uint4 v;
  __nv_bfloat162* h = reinterpret_cast<__nv_bfloat162*>(&v);
#pragma unroll
  for (int i = 0; i < kVec / 2; ++i) {
    h[i] = __floats2bfloat162_rn(f[2 * i], f[2 * i + 1]);
  }
  return v;
}

// n * q of vector v; every product is exact (bf16 x bf16).
__device__ __forceinline__ void load_qk(const AttnResParams& p, int v,
                                        bool active, float (&qw)[kVec]) {
  if (!active) {
#pragma unroll
    for (int i = 0; i < kVec; ++i) qw[i] = 0.f;
    return;
  }
  float n[kVec], q[kVec];
  bf16x8_to_float(reinterpret_cast<const uint4*>(p.norm_w)[v], n);
  bf16x8_to_float(reinterpret_cast<const uint4*>(p.qk_w)[v], q);
#pragma unroll
  for (int i = 0; i < kVec; ++i) qw[i] = __fmul_rn(n[i], q[i]);
}

__device__ __forceinline__ const uint4* block_source(const AttnResParams& p,
                                                     int row, int s) {
  return reinterpret_cast<const uint4*>(
      p.blocks + static_cast<int64_t>(row) * p.stride_block_m +
      static_cast<int64_t>(s) * p.stride_block_r);
}

// The served prefix update and block write for vector v of a row; returns
// the updated prefix vector.
__device__ __forceinline__ uint4 update_prefix(const AttnResParams& p, int row,
                                               int v) {
  uint4* prefix_row = reinterpret_cast<uint4*>(
      p.prefix + static_cast<int64_t>(row) * p.stride_prefix);
  uint4 pv = prefix_row[v];
  if (p.has_delta) {
    const uint4 dv = reinterpret_cast<const uint4*>(
        p.delta + static_cast<int64_t>(row) * p.stride_delta)[v];
    float a[kVec], b[kVec], sum[kVec];
    bf16x8_to_float(pv, a);
    bf16x8_to_float(dv, b);
#pragma unroll
    for (int i = 0; i < kVec; ++i) sum[i] = __fadd_rn(a[i], b[i]);
    pv = float_to_bf16x8(sum);
    prefix_row[v] = pv;
  }
  if (p.block_write_idx >= 0) {
    reinterpret_cast<uint4*>(
        p.blocks + static_cast<int64_t>(row) * p.stride_block_m +
        static_cast<int64_t>(p.block_write_idx) * p.stride_block_r)[v] = pv;
  }
  return pv;
}

// Per-source totals of the per-warp partial sums (sum of squares, dot):
// every source is summed over the warps by a butterfly; lane s < NS returns
// the totals of source s.
template <int NS>
__device__ __forceinline__ float2 cta_source_totals(const float (*part)[NS][2],
                                                    int lane, int nwarps) {
  float2 mine = make_float2(0.f, 0.f);
#pragma unroll
  for (int s = 0; s < NS; ++s) {
    const float ss = warp_sum(lane < nwarps ? part[lane][s][0] : 0.f);
    const float dot = warp_sum(lane < nwarps ? part[lane][s][1] : 0.f);
    if (lane == s) mine = make_float2(ss, dot);
  }
  return mine;
}

// Softmax weights, evaluated by one warp from the totals of source s held by
// lane s < NS.
template <int NS>
__device__ __forceinline__ void softmax_weights(float2 tot, int lane,
                                                const AttnResParams& p,
                                                float* s_weight) {
  float logit = -INFINITY;
  if (lane < NS) {
    const float ms =
        __fadd_rn(__fdiv_rn(tot.x, static_cast<float>(p.hidden)), p.eps);
    logit = __fdiv_rn(tot.y, __fsqrt_rn(ms));
  }
  const float m = warp_max(logit);
  const float e = lane < NS ? expf(__fsub_rn(logit, m)) : 0.f;
  const float den = warp_sum(e);
  if (lane < NS) s_weight[lane] = __fdiv_rn(e, den);
}

__device__ __forceinline__ float norm_scale(float sum_sq,
                                            const AttnResParams& p) {
  const float ms =
      __fadd_rn(__fdiv_rn(sum_sq, static_cast<float>(p.hidden)), p.oeps);
  return __frcp_rn(__fsqrt_rn(ms));
}

__device__ __forceinline__ void store_output(const AttnResParams& p, int row,
                                             int v, const float (&mixed)[kVec],
                                             float scale) {
  float o[kVec];
  if (p.apply_norm) {
    float ow[kVec];
    bf16x8_to_float(reinterpret_cast<const uint4*>(p.onorm_w)[v], ow);
#pragma unroll
    for (int i = 0; i < kVec; ++i)
      o[i] = __fmul_rn(__fmul_rn(mixed[i], scale), ow[i]);
  } else {
#pragma unroll
    for (int i = 0; i < kVec; ++i) o[i] = mixed[i];
  }
  reinterpret_cast<uint4*>(p.out + static_cast<int64_t>(row) *
                                       p.stride_out)[v] = float_to_bf16x8(o);
}

// ---------------------------------------------------------------------------
// One CTA per row. NS: number of sources (num_blocks + 1). SLOTS: 16-byte
// vectors per thread.
template <int NS, int SLOTS>
__global__ void __launch_bounds__(1024 / SLOTS)
    attn_res_decode_kernel(const AttnResParams p) {
  static_assert(NS >= 1 && NS <= kMaxSources, "source count");
  const int row = blockIdx.x;
  const int tid = threadIdx.x;
  const int lane = tid & 31;
  const int warp = tid >> 5;
  const int nwarps = blockDim.x >> 5;
  const int num_vec = p.hidden / kVec;

  __shared__ float s_part[kMaxWarps][NS][2];
  __shared__ float s_weight[NS];
  __shared__ float s_norm[kMaxWarps];
  __shared__ float s_out_scale;

  // Constant operands first: they do not depend on the preceding kernel.
  float qw[SLOTS][kVec];
#pragma unroll
  for (int j = 0; j < SLOTS; ++j) {
    const int v = tid + j * static_cast<int>(blockDim.x);
    load_qk(p, v, NS > 1 && v < num_vec, qw[j]);
  }

  griddep_wait();

  // The updated prefix (the last source) stays in registers; the blocks are
  // read once for the statistics and again (L1 hits) for the mix.
  uint4 pv[SLOTS];
#pragma unroll
  for (int j = 0; j < SLOTS; ++j) {
    const int v = tid + j * static_cast<int>(blockDim.x);
    pv[j] = v < num_vec ? update_prefix(p, row, v) : make_uint4(0u, 0u, 0u, 0u);
  }

  griddep_launch_dependents();

  if constexpr (NS > 1) {
#pragma unroll
    for (int s = 0; s < NS; ++s) {
      float ss = 0.f, dot = 0.f;
#pragma unroll
      for (int j = 0; j < SLOTS; ++j) {
        const int v = tid + j * static_cast<int>(blockDim.x);
        uint4 raw = pv[j];
        if (s < NS - 1) {
          raw = v < num_vec ? block_source(p, row, s)[v]
                            : make_uint4(0u, 0u, 0u, 0u);
        }
        float x[kVec];
        bf16x8_to_float(raw, x);
        ss = __fadd_rn(ss, dot8_pairwise(x, x));
        dot = __fadd_rn(dot, dot8_pairwise(x, qw[j]));
      }
      ss = warp_sum(ss);
      dot = warp_sum(dot);
      if (lane == 0) {
        s_part[warp][s][0] = ss;
        s_part[warp][s][1] = dot;
      }
    }
    __syncthreads();
    if (warp == 0) {
      softmax_weights<NS>(cta_source_totals<NS>(s_part, lane, nwarps), lane, p,
                          s_weight);
    }
    __syncthreads();
  }

  // Mix in source order (blocks, then the prefix).
  float mixed[SLOTS][kVec];
#pragma unroll
  for (int j = 0; j < SLOTS; ++j) {
    const int v = tid + j * static_cast<int>(blockDim.x);
    if constexpr (NS == 1) {
      bf16x8_to_float(pv[j], mixed[j]);
    } else {
#pragma unroll
      for (int i = 0; i < kVec; ++i) mixed[j][i] = 0.f;
#pragma unroll
      for (int s = 0; s < NS; ++s) {
        uint4 raw = pv[j];
        if (s < NS - 1) {
          raw = v < num_vec ? block_source(p, row, s)[v]
                            : make_uint4(0u, 0u, 0u, 0u);
        }
        float x[kVec];
        bf16x8_to_float(raw, x);
        const float w = s_weight[s];
#pragma unroll
        for (int i = 0; i < kVec; ++i)
          mixed[j][i] = __fmaf_rn(w, x[i], mixed[j][i]);
      }
    }
  }

  float scale = 1.f;
  if (p.apply_norm) {
    float sq = 0.f;
#pragma unroll
    for (int j = 0; j < SLOTS; ++j) {
      if (tid + j * static_cast<int>(blockDim.x) < num_vec) {
        sq = __fadd_rn(sq, dot8_pairwise(mixed[j], mixed[j]));
      }
    }
    sq = warp_sum(sq);
    if (lane == 0) s_norm[warp] = sq;
    __syncthreads();
    if (warp == 0) {
      const float tot = warp_sum(lane < nwarps ? s_norm[lane] : 0.f);
      if (lane == 0) s_out_scale = norm_scale(tot, p);
    }
    __syncthreads();
    scale = s_out_scale;
  }

#pragma unroll
  for (int j = 0; j < SLOTS; ++j) {
    const int v = tid + j * static_cast<int>(blockDim.x);
    if (v < num_vec) store_output(p, row, v, mixed[j], scale);
  }
}

// ---------------------------------------------------------------------------
// A cluster of CL CTAs per row, one 16-byte vector of every source per
// thread.
template <int CL>
__device__ __forceinline__ float2 cluster_pairwise_sum(const float2 (&r)[CL]) {
  if constexpr (CL == 2) {
    return make_float2(__fadd_rn(r[0].x, r[1].x), __fadd_rn(r[0].y, r[1].y));
  } else if constexpr (CL == 4) {
    return make_float2(
        __fadd_rn(__fadd_rn(r[0].x, r[1].x), __fadd_rn(r[2].x, r[3].x)),
        __fadd_rn(__fadd_rn(r[0].y, r[1].y), __fadd_rn(r[2].y, r[3].y)));
  } else {
    static_assert(CL == 8, "cluster size 2, 4 or 8");
    return make_float2(
        __fadd_rn(
            __fadd_rn(__fadd_rn(r[0].x, r[1].x), __fadd_rn(r[2].x, r[3].x)),
            __fadd_rn(__fadd_rn(r[4].x, r[5].x), __fadd_rn(r[6].x, r[7].x))),
        __fadd_rn(
            __fadd_rn(__fadd_rn(r[0].y, r[1].y), __fadd_rn(r[2].y, r[3].y)),
            __fadd_rn(__fadd_rn(r[4].y, r[5].y), __fadd_rn(r[6].y, r[7].y))));
  }
}

template <int NS, int CL>
__global__ void __launch_bounds__(kClusterThreads)
    attn_res_decode_cluster_kernel(const AttnResParams p) {
  static_assert(NS >= 1 && NS <= kMaxSources, "source count");
  cg::cluster_group cluster = cg::this_cluster();
  const int rank = static_cast<int>(cluster.block_rank());
  const int row = blockIdx.x / CL;
  const int tid = threadIdx.x;
  const int lane = tid & 31;
  const int warp = tid >> 5;
  const int nwarps = blockDim.x >> 5;
  const int per_cta = p.hidden / kVec / CL;
  const bool active = tid < per_cta;
  const int v = rank * per_cta + tid;

  __shared__ float s_part[kMaxWarps][NS][2];
  __shared__ float2 s_cta[NS];  // read by every CTA of the cluster
  __shared__ float s_weight[NS];
  __shared__ float s_norm[kMaxWarps];
  __shared__ float2 s_cta_norm;  // read by every CTA of the cluster
  __shared__ float s_out_scale;

  float qw[kVec];
  load_qk(p, v, NS > 1 && active, qw);

  griddep_wait();

  // Blocks first (independent loads in flight), then the prefix update.
  uint4 src[NS];
#pragma unroll
  for (int s = 0; s < NS - 1; ++s) {
    src[s] = active ? block_source(p, row, s)[v] : make_uint4(0u, 0u, 0u, 0u);
  }
  src[NS - 1] = active ? update_prefix(p, row, v) : make_uint4(0u, 0u, 0u, 0u);

  griddep_launch_dependents();

  if constexpr (NS > 1) {
#pragma unroll
    for (int s = 0; s < NS; ++s) {
      float x[kVec];
      bf16x8_to_float(src[s], x);
      const float ss = warp_sum(dot8_pairwise(x, x));
      const float dot = warp_sum(dot8_pairwise(x, qw));
      if (lane == 0) {
        s_part[warp][s][0] = ss;
        s_part[warp][s][1] = dot;
      }
    }
    __syncthreads();
    if (warp == 0) {
      const float2 t = cta_source_totals<NS>(s_part, lane, nwarps);
      if (lane < NS) s_cta[lane] = t;
    }
    cluster.sync();
    if (warp == 0) {
      float2 tot = make_float2(0.f, 0.f);
      if (lane < NS) {
        float2 r[CL];
#pragma unroll
        for (int c = 0; c < CL; ++c)
          r[c] = *cluster.map_shared_rank(&s_cta[lane], c);
        tot = cluster_pairwise_sum<CL>(r);
      }
      softmax_weights<NS>(tot, lane, p, s_weight);
    }
    __syncthreads();
  }

  float mixed[kVec];
  if constexpr (NS == 1) {
    bf16x8_to_float(src[0], mixed);
  } else {
#pragma unroll
    for (int i = 0; i < kVec; ++i) mixed[i] = 0.f;
#pragma unroll
    for (int s = 0; s < NS; ++s) {
      float x[kVec];
      bf16x8_to_float(src[s], x);
      const float w = s_weight[s];
#pragma unroll
      for (int i = 0; i < kVec; ++i) mixed[i] = __fmaf_rn(w, x[i], mixed[i]);
    }
  }

  float scale = 1.f;
  if (p.apply_norm) {
    const float sq = warp_sum(active ? dot8_pairwise(mixed, mixed) : 0.f);
    if (lane == 0) s_norm[warp] = sq;
    __syncthreads();
    if (warp == 0) {
      const float tot = warp_sum(lane < nwarps ? s_norm[lane] : 0.f);
      if (lane == 0) s_cta_norm = make_float2(tot, 0.f);
    }
    cluster.sync();
    if (tid == 0) {
      float2 r[CL];
#pragma unroll
      for (int c = 0; c < CL; ++c)
        r[c] = *cluster.map_shared_rank(&s_cta_norm, c);
      s_out_scale = norm_scale(cluster_pairwise_sum<CL>(r).x, p);
    }
    __syncthreads();
    scale = s_out_scale;
  }

  // A CTA's shared memory must outlive the other CTAs' reads of it: arrive
  // after the last remote read, wait before exiting.
  const bool remote_reads = NS > 1 || p.apply_norm;
  if (remote_reads)
    asm volatile("barrier.cluster.arrive.aligned;" ::: "memory");
  if (active) store_output(p, row, v, mixed, scale);
  if (remote_reads) asm volatile("barrier.cluster.wait.aligned;" ::: "memory");
}

// ---------------------------------------------------------------------------
using KernelFn = void (*)(const AttnResParams);

template <int SLOTS>
KernelFn select_cta_kernel(int ns) {
  switch (ns) {
    case 1:
      return attn_res_decode_kernel<1, SLOTS>;
    case 2:
      return attn_res_decode_kernel<2, SLOTS>;
    case 3:
      return attn_res_decode_kernel<3, SLOTS>;
    case 4:
      return attn_res_decode_kernel<4, SLOTS>;
    case 5:
      return attn_res_decode_kernel<5, SLOTS>;
    case 6:
      return attn_res_decode_kernel<6, SLOTS>;
    case 7:
      return attn_res_decode_kernel<7, SLOTS>;
    case 8:
      return attn_res_decode_kernel<8, SLOTS>;
    case 9:
      return attn_res_decode_kernel<9, SLOTS>;
    case 10:
      return attn_res_decode_kernel<10, SLOTS>;
    case 11:
      return attn_res_decode_kernel<11, SLOTS>;
    case 12:
      return attn_res_decode_kernel<12, SLOTS>;
    default:
      return nullptr;
  }
}

template <int CL>
KernelFn select_cluster_kernel(int ns) {
  switch (ns) {
    case 1:
      return attn_res_decode_cluster_kernel<1, CL>;
    case 2:
      return attn_res_decode_cluster_kernel<2, CL>;
    case 3:
      return attn_res_decode_cluster_kernel<3, CL>;
    case 4:
      return attn_res_decode_cluster_kernel<4, CL>;
    case 5:
      return attn_res_decode_cluster_kernel<5, CL>;
    case 6:
      return attn_res_decode_cluster_kernel<6, CL>;
    case 7:
      return attn_res_decode_cluster_kernel<7, CL>;
    case 8:
      return attn_res_decode_cluster_kernel<8, CL>;
    case 9:
      return attn_res_decode_cluster_kernel<9, CL>;
    case 10:
      return attn_res_decode_cluster_kernel<10, CL>;
    case 11:
      return attn_res_decode_cluster_kernel<11, CL>;
    case 12:
      return attn_res_decode_cluster_kernel<12, CL>;
    default:
      return nullptr;
  }
}

void check_bf16_rowmajor(const torch::stable::Tensor& t, const char* name) {
  STD_TORCH_CHECK(t.scalar_type() == torch::headeronly::ScalarType::BFloat16,
                  "attn_res_decode: ", name, " must be bf16");
  STD_TORCH_CHECK(t.stride(-1) == 1, "attn_res_decode: ", name,
                  " needs a unit last stride");
  STD_TORCH_CHECK(reinterpret_cast<uintptr_t>(t.data_ptr()) % 16 == 0,
                  "attn_res_decode: ", name, " must be 16-byte aligned");
}

void attn_res_decode(
    torch::stable::Tensor& output, torch::stable::Tensor& prefix,
    torch::stable::Tensor const& delta, bool has_delta,
    torch::stable::Tensor& blocks, torch::stable::Tensor const& norm_weight,
    torch::stable::Tensor const& qk_weight,
    torch::stable::Tensor const& output_norm_weight, bool apply_norm,
    int64_t num_blocks, int64_t block_write_idx, double eps,
    double output_norm_eps, int64_t slots, int64_t cluster, bool pdl) {
  STD_TORCH_CHECK(prefix.dim() == 2 && output.dim() == 2 && blocks.dim() == 3,
                  "attn_res_decode: prefix/output [rows, hidden], blocks "
                  "[rows, R, hidden]");
  const int64_t rows = prefix.size(0);
  const int64_t hidden = prefix.size(1);
  STD_TORCH_CHECK(hidden % kVec == 0,
                  "attn_res_decode: hidden must be a multiple of 8");
  STD_TORCH_CHECK(output.size(0) == rows && output.size(1) == hidden &&
                      blocks.size(0) == rows && blocks.size(2) == hidden,
                  "attn_res_decode: shape mismatch");
  STD_TORCH_CHECK(num_blocks >= 0 && num_blocks + 1 <= kMaxSources &&
                      num_blocks <= blocks.size(1),
                  "attn_res_decode: unsupported source count");
  STD_TORCH_CHECK(block_write_idx < blocks.size(1),
                  "attn_res_decode: block index");
  STD_TORCH_CHECK(norm_weight.size(0) == hidden &&
                      qk_weight.size(0) == hidden &&
                      (!apply_norm || output_norm_weight.size(0) == hidden),
                  "attn_res_decode: weight size");
  check_bf16_rowmajor(prefix, "prefix");
  check_bf16_rowmajor(output, "output");
  check_bf16_rowmajor(blocks, "blocks");
  if (has_delta) check_bf16_rowmajor(delta, "delta");
  check_bf16_rowmajor(norm_weight, "norm_weight");
  check_bf16_rowmajor(qk_weight, "qk_weight");
  if (apply_norm) check_bf16_rowmajor(output_norm_weight, "output_norm_weight");
  STD_TORCH_CHECK(
      prefix.stride(0) % kVec == 0 && output.stride(0) % kVec == 0 &&
          blocks.stride(0) % kVec == 0 && blocks.stride(1) % kVec == 0 &&
          (!has_delta || delta.stride(0) % kVec == 0),
      "attn_res_decode: row strides must be multiples of 8 elements");
  if (rows == 0) return;

  AttnResParams p;
  p.prefix = static_cast<__nv_bfloat16*>(prefix.data_ptr());
  p.delta =
      has_delta ? static_cast<const __nv_bfloat16*>(delta.data_ptr()) : nullptr;
  p.blocks = static_cast<__nv_bfloat16*>(blocks.data_ptr());
  p.norm_w = static_cast<const __nv_bfloat16*>(norm_weight.data_ptr());
  p.qk_w = static_cast<const __nv_bfloat16*>(qk_weight.data_ptr());
  p.onorm_w =
      apply_norm
          ? static_cast<const __nv_bfloat16*>(output_norm_weight.data_ptr())
          : nullptr;
  p.out = static_cast<__nv_bfloat16*>(output.data_ptr());
  p.stride_prefix = prefix.stride(0);
  p.stride_delta = has_delta ? delta.stride(0) : 0;
  p.stride_block_m = blocks.stride(0);
  p.stride_block_r = blocks.stride(1);
  p.stride_out = output.stride(0);
  p.hidden = static_cast<int>(hidden);
  p.block_write_idx = static_cast<int>(block_write_idx);
  p.eps = static_cast<float>(eps);
  p.oeps = static_cast<float>(output_norm_eps);
  p.has_delta = has_delta;
  p.apply_norm = apply_norm;

  const int num_vec = static_cast<int>(hidden / kVec);
  const int ns = static_cast<int>(num_blocks + 1);
  STD_TORCH_CHECK(cluster == 1 || cluster == 2 || cluster == 4 || cluster == 8,
                  "attn_res_decode: cluster 1, 2, 4 or 8");
  KernelFn kernel = nullptr;
  int threads = 0;
  if (cluster == 1) {
    STD_TORCH_CHECK(slots == 1 || slots == 2 || slots == 4,
                    "attn_res_decode: slots 1, 2 or 4");
    threads = (num_vec + static_cast<int>(slots) - 1) / static_cast<int>(slots);
    threads = ((threads + 31) / 32) * 32;
    STD_TORCH_CHECK(threads <= 1024 / slots,
                    "attn_res_decode: hidden too large for this slot count");
    kernel = slots == 1   ? select_cta_kernel<1>(ns)
             : slots == 2 ? select_cta_kernel<2>(ns)
                          : select_cta_kernel<4>(ns);
  } else {
    STD_TORCH_CHECK(slots == 1, "attn_res_decode: clusters use one slot");
    STD_TORCH_CHECK(num_vec % cluster == 0,
                    "attn_res_decode: hidden / 8 must divide by the cluster");
    threads = ((num_vec / static_cast<int>(cluster) + 31) / 32) * 32;
    STD_TORCH_CHECK(threads <= kClusterThreads,
                    "attn_res_decode: hidden too large for this cluster");
    kernel = cluster == 2   ? select_cluster_kernel<2>(ns)
             : cluster == 4 ? select_cluster_kernel<4>(ns)
                            : select_cluster_kernel<8>(ns);
  }

  const torch::stable::accelerator::DeviceGuard device_guard(
      prefix.get_device_index());
  cudaLaunchConfig_t cfg = {};
  cfg.gridDim = dim3(static_cast<unsigned>(rows * cluster));
  cfg.blockDim = dim3(static_cast<unsigned>(threads));
  cfg.dynamicSmemBytes = 0;
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
                  "attn_res_decode launch failed: ", cudaGetErrorString(err));
}

}  // namespace

STABLE_TORCH_LIBRARY_FRAGMENT(_C_k3decode, k3decode_ops) {
  k3decode_ops.def(
      "attn_res_decode(Tensor! output, Tensor! prefix, Tensor delta, "
      "bool has_delta, Tensor! blocks, Tensor norm_weight, Tensor qk_weight, "
      "Tensor output_norm_weight, bool apply_norm, int num_blocks, "
      "int block_write_idx, float eps, float output_norm_eps, int slots, "
      "int cluster, bool pdl) -> ()");
}

STABLE_TORCH_LIBRARY_IMPL(_C_k3decode, CUDA, k3decode_ops) {
  k3decode_ops.impl("attn_res_decode", TORCH_BOX(&attn_res_decode));
}
