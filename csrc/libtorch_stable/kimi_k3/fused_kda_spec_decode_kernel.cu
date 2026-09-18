/*
 * Fused Kimi-K3 KDA speculative-decode step, v4: one thread-block cluster of
 * kSplit=4 CTAs per (sequence, head); CTA `split` owns state rows
 * [32*split, 32*split+32) (8 warps x 4 rows, 4 k-values per lane).
 *
 * Why a cluster: v3 (one CTA per head, research/dense-feedback-20260918)
 * measured 3.45 us per token of which 1.35 us was the 64 KB per-token state
 * store leaving one SM and the rest the 16-row-per-warp reduction chains.
 * Splitting the head over 4 SMs divides both; the only cross-row coupling,
 * the gated RMS output norm's sum of squares, is exchanged once after the
 * token loop through distributed shared memory (DSMEM), so the token loop
 * has no block-level barrier at all.
 *
 * Semantics are those of the served chain (causal conv update with the
 * rolling speculative window, q/k L2 norm, gate, gated delta recurrence with
 * per-token state slots, gated RMS output norm); see v3's header and
 * feedback/kda_spec_semantics.py. Numerics: fp32 throughout, accurate expf.
 */

#include "../torch_utils.h"

#include <torch/csrc/stable/library.h>
#include <torch/headeronly/core/ScalarType.h>

#include <cooperative_groups.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <cstdint>
#include <cstdlib>
#include <optional>

namespace cg = cooperative_groups;

namespace {

constexpr int kD = 128;
constexpr int kW = 4;
constexpr int kThreads = 256;
constexpr int kWarps = kThreads / 32;
constexpr int kSplit = 4;                           // CTAs per head (cluster)
constexpr int kRowsPerCta = kD / kSplit;            // 32
constexpr int kRowsPerWarp = kRowsPerCta / kWarps;  // 4
constexpr int kMaxTokens = 8;
constexpr int kHistLen = kW - 1 + kMaxTokens;
constexpr float kQkEps = 1.0e-6f;
constexpr float kSoftplusThreshold = 20.0f;

struct SpecStrides {
  int64_t x_row;
  int64_t g_row;
  int64_t beta_row;
  int64_t g2_row;
  int64_t out_row;
  int64_t conv_slot;
  int64_t conv_col;
  int64_t state_slot;
  int64_t idx_row;
};

__device__ __forceinline__ float ld_bf16(const __nv_bfloat16* p, int64_t i) {
  return __bfloat162float(__ldg(p + i));
}
__device__ __forceinline__ float sigmoid_acc(float x) {
  return 1.0f / (1.0f + expf(-x));
}
__device__ __forceinline__ float silu_acc(float x) {
  return x / (1.0f + expf(-x));
}
__device__ __forceinline__ float sigmoid_fast(float x) {
  return __fdividef(1.0f, 1.0f + __expf(-x));
}
__device__ __forceinline__ float silu_fast(float x) {
  return __fdividef(x, 1.0f + __expf(-x));
}
__device__ __forceinline__ float softplus_acc(float x) {
  return x > kSoftplusThreshold ? x : log1pf(expf(x));
}
__device__ __forceinline__ void cp_async_16(void* smem_ptr,
                                            const void* gmem_ptr) {
  const uint32_t smem_addr =
      static_cast<uint32_t>(__cvta_generic_to_shared(smem_ptr));
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" ::"r"(smem_addr),
               "l"(gmem_ptr));
}
__device__ __forceinline__ void cp_async_commit() {
  asm volatile("cp.async.commit_group;\n" ::);
}
__device__ __forceinline__ void cp_async_wait_all() {
  asm volatile("cp.async.wait_all;\n" ::: "memory");
}

template <int N>
__device__ __forceinline__ void warp_sum_n(float (&v)[N]) {
#pragma unroll
  for (int o = 16; o > 0; o >>= 1) {
#pragma unroll
    for (int i = 0; i < N; ++i) v[i] += __shfl_xor_sync(0xffffffffu, v[i], o);
  }
}

template <bool kUseLowerBound, bool kDebugSkipStore>
__global__ void __cluster_dims__(1, 1, kSplit) __launch_bounds__(kThreads, 2)
    kda_spec_decode_kernel(
        const __nv_bfloat16* __restrict__ x, const float* __restrict__ w_t,
        const float* __restrict__ bias, __nv_bfloat16* __restrict__ conv_state,
        const float* __restrict__ a_log,
        const __nv_bfloat16* __restrict__ raw_g,
        const float* __restrict__ dt_bias,
        const __nv_bfloat16* __restrict__ raw_beta,
        const __nv_bfloat16* __restrict__ g2, const float* __restrict__ norm_w,
        const int* __restrict__ state_indices,
        const int* __restrict__ cu_seqlens,
        const int* __restrict__ num_accepted, float* __restrict__ state,
        __nv_bfloat16* __restrict__ out, int H, float lower_bound, float scale,
        float norm_eps, SpecStrides st, int debug_stage, int debug_flags) {
  const int tid = threadIdx.x;
  const int lane = tid & 31;
  const int warp = tid >> 5;
  const int n = blockIdx.x;
  const int h = blockIdx.y;
  const int split = blockIdx.z;  // == cluster block rank (cluster along z)
  const int row0 = split * kRowsPerCta;
  const int dim = H * kD;
  cg::cluster_group cluster = cg::this_cluster();

#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
  cudaGridDependencySynchronize();
  // PTX griddepcontrol: the dependent grid may *launch* now; its own
  // griddepcontrol.wait still blocks until this grid has completed and its
  // memory is visible, so triggering first hides the launch latency of the
  // next kernel behind all of our work (the served Triton conv kernel does
  // the same).
  cudaTriggerProgrammaticLaunchCompletion();
#endif
  // Every CTA of the cluster reads the same sequence metadata, so the early
  // exits below are cluster-uniform (no CTA can be left waiting at a
  // cluster barrier).
  const int qs = cu_seqlens[n];
  const int qe = cu_seqlens[n + 1];
  const int T = qe - qs;
  if (T <= 0) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
    cudaTriggerProgrammaticLaunchCompletion();
#endif
    return;
  }
  const int off = num_accepted[n] - 1;
  const int conv_slot = state_indices[n * st.idx_row];
  const int init_slot = state_indices[n * st.idx_row + off];
  if (conv_slot <= 0 || init_slot <= 0 || T > kMaxTokens || off < 0) {
    if (tid < kRowsPerCta) {
      for (int t = 0; t < T; ++t) {
        out[(int64_t)(qs + t) * st.out_row + h * kD + row0 + tid] =
            __float2bfloat16(0.0f);
      }
    }
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
    cudaTriggerProgrammaticLaunchCompletion();
#endif
    return;
  }
  if (debug_stage == 0) return;

  __shared__ float s_q[kMaxTokens][kD];
  __shared__ float s_k[kMaxTokens][kD];
  __shared__ float s_v[kMaxTokens][kD];
  __shared__ float s_decay[kMaxTokens][kD];
  __shared__ float s_og[kMaxTokens][kD];
  __shared__ float s_o[kMaxTokens][kRowsPerCta];
  __shared__ float s_part[kMaxTokens][kWarps];
  __shared__ float s_cta_sumsq[kMaxTokens];
  __shared__ float s_beta[kMaxTokens];
  __shared__ int s_slot[kMaxTokens];

  // ---- this CTA's 32 state rows: 4 per warp, 4 consecutive k per lane ----
  float hreg[kRowsPerWarp][4];
  {
    const float* src = state + (int64_t)init_slot * st.state_slot +
                       (int64_t)h * kD * kD + lane * 4;
#pragma unroll
    for (int r = 0; r < kRowsPerWarp; ++r) {
      const int row = row0 + warp * kRowsPerWarp + r;
      const float4 v4 =
          __ldg(reinterpret_cast<const float4*>(src + (int64_t)row * kD));
      hreg[r][0] = v4.x;
      hreg[r][1] = v4.y;
      hreg[r][2] = v4.z;
      hreg[r][3] = v4.w;
    }
  }
  if (tid < kMaxTokens)
    s_slot[tid] = (tid < T) ? state_indices[n * st.idx_row + tid] : 0;
  const float wn = (tid < kRowsPerCta) ? __ldg(norm_w + row0 + tid) : 0.0f;

  // ---- prologue inputs staged into shared memory by all threads at once --
  // Each 16-byte chunk is one cp.async; the whole head's inputs (T token rows
  // of q|k|v, gate and output-gate rows, the three prior conv columns) cost
  // the CTA one memory latency regardless of T.
  __shared__ __align__(16) __nv_bfloat16 s_x[kMaxTokens][3 * kD];
  __shared__ __align__(16) __nv_bfloat16 s_gin[kMaxTokens][kD];
  __shared__ __align__(16) __nv_bfloat16 s_g2in[kMaxTokens][kD];
  __shared__ __align__(16) __nv_bfloat16 s_prior[kW - 1][3 * kD];
  {
    constexpr int kSeg = kD / 8;  // 16-byte chunks per 128-channel segment
    const __nv_bfloat16* cs_base =
        conv_state + (int64_t)conv_slot * st.conv_slot;
    // prior conv columns: 3 columns x 3 parts x kSeg chunks = 144 chunks
    for (int i = tid; i < (kW - 1) * 3 * kSeg; i += kThreads) {
      const int col = i / (3 * kSeg);
      const int rem = i - col * (3 * kSeg);
      const int part = rem / kSeg;
      const int chunk = rem - part * kSeg;
      cp_async_16(&s_prior[col][part * kD + chunk * 8],
                  cs_base + (int64_t)(off + col) * st.conv_col + part * dim +
                      h * kD + chunk * 8);
    }
    // token rows: T x (3 parts x kSeg) chunks of x, T x kSeg of gate and g2
    for (int i = tid; i < T * 3 * kSeg; i += kThreads) {
      const int t = i / (3 * kSeg);
      const int rem = i - t * (3 * kSeg);
      const int part = rem / kSeg;
      const int chunk = rem - part * kSeg;
      cp_async_16(
          &s_x[t][part * kD + chunk * 8],
          x + (int64_t)(qs + t) * st.x_row + part * dim + h * kD + chunk * 8);
    }
    for (int i = tid; i < T * kSeg; i += kThreads) {
      const int t = i / kSeg;
      const int chunk = i - t * kSeg;
      cp_async_16(&s_gin[t][chunk * 8],
                  raw_g + (int64_t)(qs + t) * st.g_row + h * kD + chunk * 8);
      cp_async_16(&s_g2in[t][chunk * 8],
                  g2 + (int64_t)(qs + t) * st.g2_row + h * kD + chunk * 8);
    }
    cp_async_commit();
  }

  // ---- prologue (every CTA of the cluster, redundantly): conv + gates ----
  {
    const int c = tid & (kD - 1);
    const bool group_a = tid < kD;
    const int p0 = group_a ? 0 : 2;
    const int ch0 = p0 * dim + h * kD + c;
    const int ch1 = 1 * dim + h * kD + c;
    __nv_bfloat16* cs0 = conv_state + (int64_t)conv_slot * st.conv_slot + ch0;
    __nv_bfloat16* cs1 = conv_state + (int64_t)conv_slot * st.conv_slot + ch1;
    float hist0[kHistLen];
    float hist1[kHistLen];
    float graw[kMaxTokens];
    float g2raw[kMaxTokens];
    const float* wp0 = w_t + (int64_t)p0 * kW * dim + h * kD + c;
    const float* wp1 = w_t + (int64_t)1 * kW * dim + h * kD + c;
    float w0[kW];
    float w1[kW];
#pragma unroll
    for (int j = 0; j < kW; ++j) {
      w0[j] = __ldg(wp0 + j * dim);
      w1[j] = group_a ? __ldg(wp1 + j * dim) : 0.0f;
    }
    const float b0 = bias == nullptr ? 0.0f : __ldg(bias + ch0);
    const float b1 = (bias == nullptr || !group_a) ? 0.0f : __ldg(bias + ch1);
    const float a = expf(__ldg(a_log + h));
    const float db = __ldg(dt_bias + h * kD + c);
    cp_async_wait_all();
    __syncthreads();
#pragma unroll
    for (int i = 0; i < kW - 1; ++i) {
      hist0[i] = __bfloat162float(s_prior[i][p0 * kD + c]);
      hist1[i] = group_a ? __bfloat162float(s_prior[i][1 * kD + c]) : 0.0f;
    }
#pragma unroll
    for (int t = 0; t < kMaxTokens; ++t) {
      const bool live = t < T;
      hist0[kW - 1 + t] = live ? __bfloat162float(s_x[t][p0 * kD + c]) : 0.0f;
      hist1[kW - 1 + t] =
          (live && group_a) ? __bfloat162float(s_x[t][1 * kD + c]) : 0.0f;
      graw[t] = (live && !group_a) ? __bfloat162float(s_gin[t][c]) : 0.0f;
      g2raw[t] = (live && !group_a) ? __bfloat162float(s_g2in[t][c]) : 0.0f;
    }
    if (tid >= kThreads - kMaxTokens) {
      const int t = tid - (kThreads - kMaxTokens);
      if (t < T)
        s_beta[t] =
            sigmoid_acc(ld_bf16(raw_beta, (int64_t)(qs + t) * st.beta_row + h));
    }
#pragma unroll
    for (int t = 0; t < kMaxTokens; ++t) {
      if (t < T) {
        float acc0 = b0;
        acc0 = fmaf(hist0[t + 0], w0[0], acc0);
        acc0 = fmaf(hist0[t + 1], w0[1], acc0);
        acc0 = fmaf(hist0[t + 2], w0[2], acc0);
        acc0 = fmaf(hist0[t + 3], w0[3], acc0);
        acc0 = (debug_flags & 2) ? silu_fast(acc0) : silu_acc(acc0);
        if (group_a) {
          float acc1 = b1;
          acc1 = fmaf(hist1[t + 0], w1[0], acc1);
          acc1 = fmaf(hist1[t + 1], w1[1], acc1);
          acc1 = fmaf(hist1[t + 2], w1[2], acc1);
          acc1 = fmaf(hist1[t + 3], w1[3], acc1);
          s_q[t][c] = acc0;
          s_k[t][c] = (debug_flags & 2) ? silu_fast(acc1) : silu_acc(acc1);
        } else {
          s_v[t][c] = acc0;
          float lg;
          if constexpr (kUseLowerBound) {
            lg = lower_bound * ((debug_flags & 2)
                                    ? sigmoid_fast(a * (graw[t] + db))
                                    : sigmoid_acc(a * (graw[t] + db)));
          } else {
            lg = -a * softplus_acc(graw[t] + db);
          }
          s_decay[t][c] = (debug_flags & 2) ? __expf(lg) : expf(lg);
          s_og[t][c] = (debug_flags & 2) ? sigmoid_fast(g2raw[t])
                                         : sigmoid_acc(g2raw[t]);
        }
      }
    }
    // new conv state = hist[1 : 3+T] -> columns 0 .. 1+T; written once per
    // head by the cluster's first CTA (values are exact bf16 already)
    if (split == 0) {
#pragma unroll
      for (int i = 0; i < kHistLen - 1; ++i) {
        if (i < kW - 2 + T) {
          cs0[(int64_t)i * st.conv_col] = __float2bfloat16(hist0[1 + i]);
          if (group_a)
            cs1[(int64_t)i * st.conv_col] = __float2bfloat16(hist1[1 + i]);
        }
      }
    }
  }
  __syncthreads();

  // ---- q/k L2 norm once per token: warp w handles token t = w ----------
  for (int t = warp; t < T; t += kWarps) {
    const float4 qv = *reinterpret_cast<const float4*>(&s_q[t][lane * 4]);
    const float4 kv = *reinterpret_cast<const float4*>(&s_k[t][lane * 4]);
    float ss[2];
    ss[0] = fmaf(qv.x, qv.x, fmaf(qv.y, qv.y, fmaf(qv.z, qv.z, qv.w * qv.w)));
    ss[1] = fmaf(kv.x, kv.x, fmaf(kv.y, kv.y, fmaf(kv.z, kv.z, kv.w * kv.w)));
    warp_sum_n<2>(ss);
    const float rq = (1.0f / sqrtf(ss[0] + kQkEps)) * scale;
    const float rk = 1.0f / sqrtf(ss[1] + kQkEps);
    *reinterpret_cast<float4*>(&s_q[t][lane * 4]) =
        make_float4(qv.x * rq, qv.y * rq, qv.z * rq, qv.w * rq);
    *reinterpret_cast<float4*>(&s_k[t][lane * 4]) =
        make_float4(kv.x * rk, kv.y * rk, kv.z * rk, kv.w * rk);
  }
  __syncthreads();
  if (debug_stage == 1) return;

  // ---- recurrence: 4 rows per warp, no block barrier inside the loop -----
  for (int t = 0; t < T; ++t) {
    const float4 qv = *reinterpret_cast<const float4*>(&s_q[t][lane * 4]);
    const float4 kv = *reinterpret_cast<const float4*>(&s_k[t][lane * 4]);
    const float4 dv = *reinterpret_cast<const float4*>(&s_decay[t][lane * 4]);
    const float q4[4] = {qv.x, qv.y, qv.z, qv.w};
    const float k4[4] = {kv.x, kv.y, kv.z, kv.w};
    const float d4[4] = {dv.x, dv.y, dv.z, dv.w};
    const float beta = s_beta[t];
    float dot_hk[kRowsPerWarp];
#pragma unroll
    for (int i = 0; i < kRowsPerWarp; ++i) {
      float* hr = hreg[i];
      hr[0] *= d4[0];
      hr[1] *= d4[1];
      hr[2] *= d4[2];
      hr[3] *= d4[3];
      dot_hk[i] = fmaf(hr[0], k4[0],
                       fmaf(hr[1], k4[1], fmaf(hr[2], k4[2], hr[3] * k4[3])));
    }
    if (!(debug_flags & 1)) warp_sum_n<kRowsPerWarp>(dot_hk);
    float dot_hq[kRowsPerWarp];
#pragma unroll
    for (int i = 0; i < kRowsPerWarp; ++i) {
      const int row = row0 + warp * kRowsPerWarp + i;
      const float v_new = (s_v[t][row] - dot_hk[i]) * beta;
      float* hr = hreg[i];
      hr[0] = fmaf(k4[0], v_new, hr[0]);
      hr[1] = fmaf(k4[1], v_new, hr[1]);
      hr[2] = fmaf(k4[2], v_new, hr[2]);
      hr[3] = fmaf(k4[3], v_new, hr[3]);
      dot_hq[i] = fmaf(hr[0], q4[0],
                       fmaf(hr[1], q4[1], fmaf(hr[2], q4[2], hr[3] * q4[3])));
    }
    if (!(debug_flags & 1)) warp_sum_n<kRowsPerWarp>(dot_hq);
    float sumsq = 0.0f;
#pragma unroll
    for (int i = 0; i < kRowsPerWarp; ++i) {
      if (lane == 0) s_o[t][warp * kRowsPerWarp + i] = dot_hq[i];
      sumsq = fmaf(dot_hq[i], dot_hq[i], sumsq);
    }
    if (lane == 0) s_part[t][warp] = sumsq;
    const int fslot = s_slot[t];
    if (!kDebugSkipStore && fslot > 0) {
      float* dst = state + (int64_t)fslot * st.state_slot +
                   (int64_t)h * kD * kD + lane * 4;
#pragma unroll
      for (int r = 0; r < kRowsPerWarp; ++r) {
        const int row = row0 + warp * kRowsPerWarp + r;
        __stcg(reinterpret_cast<float4*>(dst + (int64_t)row * kD),
               make_float4(hreg[r][0], hreg[r][1], hreg[r][2], hreg[r][3]));
      }
    }
  }
  __syncthreads();
  if (tid < T) {
    float acc = 0.0f;
#pragma unroll
    for (int w = 0; w < kWarps; ++w) acc += s_part[tid][w];
    s_cta_sumsq[tid] = acc;
  }
  // ---- output norm: the four CTAs of the head exchange their partial sums --
  cluster.sync();
  if (tid < kRowsPerCta) {
    float totals[kMaxTokens];
#pragma unroll
    for (int t = 0; t < kMaxTokens; ++t) totals[t] = 0.0f;
#pragma unroll
    for (int r = 0; r < kSplit; ++r) {
      const float* peer = cluster.map_shared_rank(s_cta_sumsq, r);
#pragma unroll
      for (int t = 0; t < kMaxTokens; ++t) {
        if (t < T) totals[t] += peer[t];
      }
    }
#pragma unroll
    for (int t = 0; t < kMaxTokens; ++t) {
      if (t < T) {
        const float rstd =
            1.0f / sqrtf(totals[t] / static_cast<float>(kD) + norm_eps);
        const float y = s_o[t][tid] * rstd * wn * s_og[t][row0 + tid];
        out[(int64_t)(qs + t) * st.out_row + h * kD + row0 + tid] =
            __float2bfloat16(y);
      }
    }
  }
  // no CTA may exit while a peer can still read its shared memory
  cluster.sync();
}

template <bool kUseLowerBound, bool kDebugSkipStore>
void launch(const __nv_bfloat16* x, const float* w_t, const float* bias,
            __nv_bfloat16* conv_state, const float* a_log,
            const __nv_bfloat16* raw_g, const float* dt_bias,
            const __nv_bfloat16* raw_beta, const __nv_bfloat16* g2,
            const float* norm_w, const int* state_indices,
            const int* cu_seqlens, const int* num_accepted, float* state,
            __nv_bfloat16* out, int N, int H, float lower_bound, float scale,
            float norm_eps, SpecStrides strides, int debug_stage,
            int debug_flags, cudaStream_t stream) {
  auto kernel = &kda_spec_decode_kernel<kUseLowerBound, kDebugSkipStore>;
  cudaLaunchConfig_t config{};
  config.gridDim = dim3(N, H, kSplit);
  config.blockDim = dim3(kThreads);
  config.dynamicSmemBytes = 0;
  config.stream = stream;
  cudaLaunchAttribute attrs[1];
  attrs[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
  attrs[0].val.programmaticStreamSerializationAllowed = 1;
  config.attrs = attrs;
  config.numAttrs = 1;
  cudaLaunchKernelEx(&config, kernel, x, w_t, bias, conv_state, a_log, raw_g,
                     dt_bias, raw_beta, g2, norm_w, state_indices, cu_seqlens,
                     num_accepted, state, out, H, lower_bound, scale, norm_eps,
                     strides, debug_stage, debug_flags);
}

void fused_kda_spec_decode(
    torch::stable::Tensor const& x, torch::stable::Tensor const& weight,
    std::optional<torch::stable::Tensor> bias,
    torch::stable::Tensor& conv_state, torch::stable::Tensor const& raw_g,
    torch::stable::Tensor const& raw_beta, torch::stable::Tensor const& a_log,
    torch::stable::Tensor const& dt_bias,
    torch::stable::Tensor const& state_indices,
    torch::stable::Tensor const& cu_seqlens,
    torch::stable::Tensor const& num_accepted, torch::stable::Tensor& state,
    torch::stable::Tensor& out, std::optional<double> lower_bound,
    torch::stable::Tensor const& output_gate,
    torch::stable::Tensor const& norm_weight, double norm_eps) {
  using torch::headeronly::ScalarType;
  STD_TORCH_CHECK(
      x.is_cuda() && x.scalar_type() == ScalarType::BFloat16 && x.dim() == 2,
      "x must be a CUDA bfloat16 [T, 3*H*128] tensor");
  const int64_t T_total = x.size(0);
  const int64_t width3 = x.size(1);
  STD_TORCH_CHECK(width3 % (3 * kD) == 0, "x must have 3*H*128 columns");
  const int H = static_cast<int>(width3 / (3 * kD));
  const int64_t dim = static_cast<int64_t>(H) * kD;
  STD_TORCH_CHECK(x.stride(1) == 1, "x must be contiguous in its last dim");
  STD_TORCH_CHECK(
      weight.is_cuda() && weight.scalar_type() == ScalarType::Float &&
          weight.dim() == 3 && weight.is_contiguous() && weight.size(0) == 3 &&
          weight.size(1) == kW && weight.size(2) == dim,
      "weight must be a contiguous float32 [3, 4, H*128] tensor");
  const float* bias_ptr = nullptr;
  if (bias.has_value()) {
    STD_TORCH_CHECK(bias->is_cuda() &&
                        bias->scalar_type() == ScalarType::Float &&
                        bias->is_contiguous() && bias->numel() == 3 * dim,
                    "bias must be a contiguous float32 [3*H*128] tensor");
    bias_ptr = static_cast<const float*>(bias->data_ptr());
  }
  STD_TORCH_CHECK(conv_state.is_cuda() &&
                      conv_state.scalar_type() == ScalarType::BFloat16 &&
                      conv_state.dim() == 3 && conv_state.size(1) == 3 * dim &&
                      conv_state.stride(1) == 1,
                  "conv_state must be a bfloat16 [slots, 3*H*128, C] view with "
                  "unit channel stride");
  const int64_t conv_cols = conv_state.size(2);
  STD_TORCH_CHECK(
      raw_g.is_cuda() && raw_g.scalar_type() == ScalarType::BFloat16 &&
          raw_g.dim() == 4 && raw_g.size(0) == 1 && raw_g.size(1) == T_total &&
          raw_g.size(2) == H && raw_g.size(3) == kD && raw_g.stride(3) == 1 &&
          raw_g.stride(2) == kD,
      "raw_g must be a bfloat16 [1, T, H, 128] tensor with contiguous head "
      "rows");
  STD_TORCH_CHECK(raw_beta.is_cuda() &&
                      raw_beta.scalar_type() == ScalarType::BFloat16 &&
                      raw_beta.dim() == 3 && raw_beta.size(0) == 1 &&
                      raw_beta.size(1) == T_total && raw_beta.size(2) == H &&
                      raw_beta.stride(2) == 1,
                  "raw_beta must be a bfloat16 [1, T, H] tensor");
  STD_TORCH_CHECK(a_log.is_cuda() && a_log.scalar_type() == ScalarType::Float &&
                      a_log.is_contiguous() && a_log.numel() == H,
                  "A_log must be a contiguous float32 [H] tensor");
  STD_TORCH_CHECK(dt_bias.is_cuda() &&
                      dt_bias.scalar_type() == ScalarType::Float &&
                      dt_bias.is_contiguous() && dt_bias.numel() == dim,
                  "dt_bias must be a contiguous float32 [H*128] tensor");
  STD_TORCH_CHECK(state_indices.is_cuda() &&
                      state_indices.scalar_type() == ScalarType::Int &&
                      state_indices.dim() == 2 && state_indices.stride(1) == 1,
                  "state_indices must be an int32 [N, num_spec+1] tensor");
  const int N = static_cast<int>(state_indices.size(0));
  const int64_t num_slots_per_seq = state_indices.size(1);
  STD_TORCH_CHECK(num_slots_per_seq <= kMaxTokens, "at most ", kMaxTokens,
                  " tokens per sequence");
  STD_TORCH_CHECK(conv_cols >= kW - 1 + num_slots_per_seq - 1,
                  "conv_state has too few columns for the speculative window");
  STD_TORCH_CHECK(cu_seqlens.is_cuda() &&
                      cu_seqlens.scalar_type() == ScalarType::Int &&
                      cu_seqlens.is_contiguous() && cu_seqlens.numel() == N + 1,
                  "cu_seqlens must be a contiguous int32 [N+1] tensor");
  STD_TORCH_CHECK(num_accepted.is_cuda() &&
                      num_accepted.scalar_type() == ScalarType::Int &&
                      num_accepted.is_contiguous() && num_accepted.numel() == N,
                  "num_accepted must be a contiguous int32 [N] tensor");
  STD_TORCH_CHECK(state.is_cuda() && state.scalar_type() == ScalarType::Float &&
                      state.dim() == 4 && state.size(1) == H &&
                      state.size(2) == kD && state.size(3) == kD &&
                      state.stride(1) == kD * kD && state.stride(2) == kD &&
                      state.stride(3) == 1,
                  "state must be a float32 [slots, H, 128, 128] tensor with "
                  "contiguous slots");
  STD_TORCH_CHECK(out.is_cuda() && out.scalar_type() == ScalarType::BFloat16 &&
                      out.dim() == 4 && out.size(0) == 1 &&
                      out.size(1) == T_total && out.size(2) == H &&
                      out.size(3) == kD && out.is_contiguous(),
                  "out must be a contiguous bfloat16 [1, T, H, 128] tensor");
  STD_TORCH_CHECK(output_gate.is_cuda() &&
                      output_gate.scalar_type() == ScalarType::BFloat16 &&
                      output_gate.dim() == 3 &&
                      output_gate.size(0) == T_total &&
                      output_gate.size(1) == H && output_gate.size(2) == kD &&
                      output_gate.stride(2) == 1 && output_gate.stride(1) == kD,
                  "output_gate must be a bfloat16 [T, H, 128] tensor with "
                  "contiguous head rows");
  STD_TORCH_CHECK(norm_weight.is_cuda() &&
                      norm_weight.scalar_type() == ScalarType::Float &&
                      norm_weight.is_contiguous() && norm_weight.numel() == kD,
                  "norm_weight must be a contiguous float32 [128] tensor");
  STD_TORCH_CHECK(norm_eps >= 0.0, "norm_eps must be non-negative");

  const SpecStrides strides{
      x.stride(0),           raw_g.stride(1), raw_beta.stride(1),
      output_gate.stride(0), out.stride(1),   conv_state.stride(0),
      conv_state.stride(2),  state.stride(0), state_indices.stride(0),
  };
  const bool use_lb = lower_bound.has_value();
  const float lb = use_lb ? static_cast<float>(*lower_bound) : 0.0f;
  constexpr float kScale = 0.08838834764831845f;  // 128 ** -0.5

  torch::stable::accelerator::DeviceGuard const device_guard(
      x.get_device_index());
  cudaStream_t const stream = get_current_cuda_stream(x.get_device_index());
  // K3KDA_DEBUG_STAGE=0|1 truncates the kernel (timing experiments only).
  static const int debug_stage =
      std::getenv("K3KDA_DEBUG_STAGE")
          ? std::atoi(std::getenv("K3KDA_DEBUG_STAGE"))
          : 2;
  static const int debug_flags =
      std::getenv("K3KDA_DEBUG_FLAGS")
          ? std::atoi(std::getenv("K3KDA_DEBUG_FLAGS"))
          : 0;
  auto args = [&](auto fn) {
    fn(static_cast<const __nv_bfloat16*>(x.data_ptr()),
       static_cast<const float*>(weight.data_ptr()), bias_ptr,
       static_cast<__nv_bfloat16*>(conv_state.data_ptr()),
       static_cast<const float*>(a_log.data_ptr()),
       static_cast<const __nv_bfloat16*>(raw_g.data_ptr()),
       static_cast<const float*>(dt_bias.data_ptr()),
       static_cast<const __nv_bfloat16*>(raw_beta.data_ptr()),
       static_cast<const __nv_bfloat16*>(output_gate.data_ptr()),
       static_cast<const float*>(norm_weight.data_ptr()),
       static_cast<const int*>(state_indices.data_ptr()),
       static_cast<const int*>(cu_seqlens.data_ptr()),
       static_cast<const int*>(num_accepted.data_ptr()),
       static_cast<float*>(state.data_ptr()),
       static_cast<__nv_bfloat16*>(out.data_ptr()), N, H, lb, kScale,
       static_cast<float>(norm_eps), strides, debug_stage, debug_flags, stream);
  };
  // K3KDA_DEBUG_SKIP_STORE=1 drops the per-token state stores (timing
  // experiments only; outputs stay valid, the cache does not).
  static const bool debug_skip_store =
      std::getenv("K3KDA_DEBUG_SKIP_STORE") != nullptr;
  if (N > 0) {
    if (debug_skip_store) {
      if (use_lb)
        args(launch<true, true>);
      else
        args(launch<false, true>);
    } else {
      if (use_lb)
        args(launch<true, false>);
      else
        args(launch<false, false>);
    }
  }
  cudaError_t const error = cudaGetLastError();
  STD_TORCH_CHECK(error == cudaSuccess,
                  "Kimi K3 fused KDA spec-decode launch failed: ",
                  cudaGetErrorString(error));
}

}  // namespace

STABLE_TORCH_LIBRARY_FRAGMENT(_C_k3kda, fused_kda_spec_decode_ops) {
  fused_kda_spec_decode_ops.def(
      "fused_kda_spec_decode("
      "Tensor x, Tensor weight, Tensor? bias, Tensor! conv_state, "
      "Tensor raw_g, Tensor raw_beta, Tensor A_log, Tensor dt_bias, "
      "Tensor state_indices, Tensor cu_seqlens, Tensor num_accepted, "
      "Tensor! state, Tensor! out, float? lower_bound, "
      "Tensor output_gate, Tensor norm_weight, float norm_eps) -> ()");
}

STABLE_TORCH_LIBRARY_IMPL(_C_k3kda, CUDA, fused_kda_spec_decode_ops) {
  fused_kda_spec_decode_ops.impl("fused_kda_spec_decode",
                                 TORCH_BOX(&fused_kda_spec_decode));
}
