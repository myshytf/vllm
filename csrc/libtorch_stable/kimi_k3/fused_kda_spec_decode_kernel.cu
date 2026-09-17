/*
 * Fused Kimi-K3 KDA speculative-decode step (side extension `_C_k3kda`).
 *
 * Build: this file is NOT part of the `_C` CMake target. It is compiled as a
 * standalone stable-ABI extension against the served PyTorch (see
 * kimi-k3-production/research/dense-feedback-20260918/kernel/build.sh:
 * nvcc -gencode arch=compute_120,code=sm_120 ... -o _C_k3kda.abi3.so) and
 * loaded by vllm/models/kimi_k3/nvidia/kda.py from
 * VLLM_K3_KDA_SPEC_FUSED_LIB. Keeping the source here makes the runtime
 * change reviewable in one tree; `torch_utils.h` is the sibling header.
 *
 * One launch per KDA layer replaces the served three-kernel chain for the
 * pure speculative-decode batch (`_causal_conv1d_update_kernel` +
 * `fused_recurrent_kda_fwd_kernel` + `layer_norm_gated_fwd_kernel`):
 *
 *   for each (sequence n, head h) — one 256-thread CTA:
 *     1. causal conv1d over the rolling speculative window
 *        (history = 3 prior conv-state columns starting at num_accepted-1,
 *        followed by the T new tokens; new state = history[1:3+T] written to
 *        columns 0..1+T), computed once per token in fp32 (the served chain
 *        rounds the conv output to bf16 before the recurrence);
 *     2. q/k L2 normalisation, gate exp and beta sigmoid once per token
 *        (the served Triton kernel recomputes them in every V-block program:
 *        32x at BV=4);
 *     3. the gated delta recurrence with the 128x128 fp32 state resident in
 *        registers across the T tokens (16 rows per warp, 4 k-values per
 *        lane), each token's state stored to its own slot exactly as the
 *        served kernel does;
 *     4. the gated RMS output norm on the unrounded fp32 recurrence output.
 *
 * Numerics: fp32 throughout with correctly rounded sqrt/div and accurate
 * expf; reductions are warp shuffle trees over per-lane fp32 partials. The
 * result is not bit-identical to the served chain (different reduction
 * order, no bf16 intermediates); the fp64 feedback harness
 * (feedback/kda_spec_check.py) measures both against the exact chain.
 *
 * Semantics mirrored from vllm/models/kimi_k3/nvidia/kda.py spec path and
 * vllm/model_executor/layers/mamba/ops/causal_conv1d.py (varlen + spec,
 * KERNEL_WIDTH 4). Differences: a sequence whose conv slot or initial state
 * slot is the null slot (<= 0) gets zeros written to all of its output rows
 * (the served recurrent kernel zeroes only the first row and leaves the
 * others stale); nothing else is written for it.
 */

#include "../torch_utils.h"

#include <torch/csrc/stable/library.h>
#include <torch/headeronly/core/ScalarType.h>

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <cstdint>
#include <optional>

namespace {

constexpr int kD = 128;  // head dim (K == V)
constexpr int kW = 4;    // conv width
constexpr int kThreads = 256;
constexpr int kWarps = kThreads / 32;
constexpr int kRowsPerWarp = kD / kWarps;  // 16 state rows per warp
constexpr int kMaxTokens = 8;              // >= num_spec + 1 (served: 5)
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
  return __bfloat162float(p[i]);
}

__device__ __forceinline__ float sigmoid_acc(float x) {
  return 1.0f / (1.0f + expf(-x));
}

__device__ __forceinline__ float silu_acc(float x) {
  return x / (1.0f + expf(-x));
}

__device__ __forceinline__ float softplus_acc(float x) {
  return x > kSoftplusThreshold ? x : log1pf(expf(x));
}

template <int N>
__device__ __forceinline__ void warp_sum_n(float (&v)[N]) {
#pragma unroll
  for (int o = 16; o > 0; o >>= 1) {
#pragma unroll
    for (int i = 0; i < N; ++i) v[i] += __shfl_xor_sync(0xffffffffu, v[i], o);
  }
}

template <bool kUseLowerBound>
__global__ void __launch_bounds__(kThreads, 2) kda_spec_decode_kernel(
    const __nv_bfloat16* __restrict__ x,      // [T_total, 3*H*D]
    const float* __restrict__ w_t,            // [3][kW][H*D]
    const float* __restrict__ bias,           // [3*H*D] or nullptr
    __nv_bfloat16* __restrict__ conv_state,   // [slots][3*H*D][C] (SD layout)
    const float* __restrict__ a_log,          // [H]
    const __nv_bfloat16* __restrict__ raw_g,  // [T_total][H][D]
    const float* __restrict__ dt_bias,        // [H*D]
    const __nv_bfloat16* __restrict__ raw_beta,  // [T_total][H]
    const __nv_bfloat16* __restrict__ g2,        // [T_total][H][D]
    const float* __restrict__ norm_w,            // [D]
    const int* __restrict__ state_indices,       // [N][num_spec+1]
    const int* __restrict__ cu_seqlens,          // [N+1]
    const int* __restrict__ num_accepted,        // [N]
    float* __restrict__ state,                   // [slots][H][D][D]
    __nv_bfloat16* __restrict__ out,             // [T_total][H][D]
    int H, float lower_bound, float scale, float norm_eps, SpecStrides st) {
  const int tid = threadIdx.x;
  const int lane = tid & 31;
  const int warp = tid >> 5;
  const int n = blockIdx.x;
  const int h = blockIdx.y;
  const int dim = H * kD;

#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
  cudaGridDependencySynchronize();
#endif

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
    // Null slot (cudagraph padding row with a real length, or a defensive
    // path): defined output, no state writes.
    if (tid < kD) {
      for (int t = 0; t < T; ++t) {
        out[(int64_t)(qs + t) * st.out_row + h * kD + tid] =
            __float2bfloat16(0.0f);
      }
    }
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
    cudaTriggerProgrammaticLaunchCompletion();
#endif
    return;
  }

  __shared__ float s_q[kMaxTokens][kD];
  __shared__ float s_k[kMaxTokens][kD];
  __shared__ float s_v[kMaxTokens][kD];
  __shared__ float s_decay[kMaxTokens][kD];
  __shared__ float s_og[kMaxTokens][kD];
  __shared__ float s_beta[kMaxTokens];
  __shared__ float s_o[kD];
  __shared__ float s_red[2 * kWarps];

  // ---- state load: 16 rows per warp, 4 consecutive k per lane ----------
  float hreg[kRowsPerWarp][4];
  {
    const float* src = state + (int64_t)init_slot * st.state_slot +
                       (int64_t)h * kD * kD + lane * 4;
#pragma unroll
    for (int r = 0; r < kRowsPerWarp; ++r) {
      const int row = warp * kRowsPerWarp + r;
      const float4 v4 =
          *reinterpret_cast<const float4*>(src + (int64_t)row * kD);
      hreg[r][0] = v4.x;
      hreg[r][1] = v4.y;
      hreg[r][2] = v4.z;
      hreg[r][3] = v4.w;
    }
  }

  // ---- phase A: conv + gates (threads 0..127: q,k channel c; 128..255: v) --
  {
    const int c = tid & (kD - 1);
    const int part_lo = (tid < kD) ? 0 : 2;  // q(0)+k(1) or v(2)
    const int part_hi = (tid < kD) ? 1 : 2;
    for (int part = part_lo; part <= part_hi; ++part) {
      const int ch = part * dim + h * kD + c;  // channel in packed qkv
      __nv_bfloat16* cs = conv_state + (int64_t)conv_slot * st.conv_slot + ch;
      float hist[kHistLen];
#pragma unroll
      for (int i = 0; i < kW - 1; ++i) {
        hist[i] = __bfloat162float(cs[(int64_t)(off + i) * st.conv_col]);
      }
#pragma unroll
      for (int t = 0; t < kMaxTokens; ++t) {
        hist[kW - 1 + t] =
            (t < T) ? ld_bf16(x, (int64_t)(qs + t) * st.x_row + ch) : 0.0f;
      }
      const float b = bias == nullptr ? 0.0f : bias[ch];
      const float* wp = w_t + (int64_t)part * kW * dim + ch;
      const float w0 = wp[0 * dim], w1 = wp[1 * dim], w2 = wp[2 * dim],
                  w3 = wp[3 * dim];
#pragma unroll
      for (int t = 0; t < kMaxTokens; ++t) {
        if (t < T) {
          float acc = b;
          acc = fmaf(hist[t + 0], w0, acc);
          acc = fmaf(hist[t + 1], w1, acc);
          acc = fmaf(hist[t + 2], w2, acc);
          acc = fmaf(hist[t + 3], w3, acc);
          acc = silu_acc(acc);
          if (part == 0)
            s_q[t][c] = acc;
          else if (part == 1)
            s_k[t][c] = acc;
          else
            s_v[t][c] = acc;
        }
      }
      // new conv state = hist[1 : 3+T] -> columns 0 .. 1+T (bf16 values are
      // exact: history columns and x are bf16 already)
#pragma unroll
      for (int i = 0; i < kHistLen - 1; ++i) {
        if (i < kW - 2 + T)
          cs[(int64_t)i * st.conv_col] = __float2bfloat16(hist[1 + i]);
      }
    }
    if (tid >= kD) {
      // gate exp per k channel and output-gate sigmoid per v channel
      const float a = expf(a_log[h]);
      const float db = dt_bias[h * kD + c];
      for (int t = 0; t < T; ++t) {
        const float g =
            ld_bf16(raw_g, (int64_t)(qs + t) * st.g_row + h * kD + c) + db;
        float lg;
        if constexpr (kUseLowerBound) {
          lg = lower_bound * sigmoid_acc(a * g);
        } else {
          lg = -a * softplus_acc(g);
        }
        s_decay[t][c] = expf(lg);
        s_og[t][c] = sigmoid_acc(
            ld_bf16(g2, (int64_t)(qs + t) * st.g2_row + h * kD + c));
      }
    }
    if (tid == kThreads - 1) {
      for (int t = 0; t < T; ++t) {
        s_beta[t] =
            sigmoid_acc(ld_bf16(raw_beta, (int64_t)(qs + t) * st.beta_row + h));
      }
    }
  }
  __syncthreads();
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
  // All reads of the previous kernels' outputs are done (their values live in
  // shared memory / registers); the dependent grid may start its prologue.
  cudaTriggerProgrammaticLaunchCompletion();
#endif

  // ---- q/k L2 norm once per token: warp w handles tokens t = w, w+8, ... ----
  for (int t = warp; t < T; t += kWarps) {
    float q4[4], k4[4];
    const float4 qv = *reinterpret_cast<const float4*>(&s_q[t][lane * 4]);
    const float4 kv = *reinterpret_cast<const float4*>(&s_k[t][lane * 4]);
    q4[0] = qv.x;
    q4[1] = qv.y;
    q4[2] = qv.z;
    q4[3] = qv.w;
    k4[0] = kv.x;
    k4[1] = kv.y;
    k4[2] = kv.z;
    k4[3] = kv.w;
    float ss[2];
    ss[0] = fmaf(q4[0], q4[0],
                 fmaf(q4[1], q4[1], fmaf(q4[2], q4[2], q4[3] * q4[3])));
    ss[1] = fmaf(k4[0], k4[0],
                 fmaf(k4[1], k4[1], fmaf(k4[2], k4[2], k4[3] * k4[3])));
    warp_sum_n<2>(ss);
    const float rq = (1.0f / sqrtf(ss[0] + kQkEps)) * scale;
    const float rk = 1.0f / sqrtf(ss[1] + kQkEps);
    *reinterpret_cast<float4*>(&s_q[t][lane * 4]) =
        make_float4(q4[0] * rq, q4[1] * rq, q4[2] * rq, q4[3] * rq);
    *reinterpret_cast<float4*>(&s_k[t][lane * 4]) =
        make_float4(k4[0] * rk, k4[1] * rk, k4[2] * rk, k4[3] * rk);
  }
  __syncthreads();

  // ---- phase B: recurrence, one token at a time ---------------------------
  for (int t = 0; t < T; ++t) {
    const float4 qv = *reinterpret_cast<const float4*>(&s_q[t][lane * 4]);
    const float4 kv = *reinterpret_cast<const float4*>(&s_k[t][lane * 4]);
    const float4 dv = *reinterpret_cast<const float4*>(&s_decay[t][lane * 4]);
    const float q4[4] = {qv.x, qv.y, qv.z, qv.w};
    const float k4[4] = {kv.x, kv.y, kv.z, kv.w};
    const float d4[4] = {dv.x, dv.y, dv.z, dv.w};
    const float beta = s_beta[t];
    float o_sumsq = 0.0f;

#pragma unroll
    for (int r0 = 0; r0 < kRowsPerWarp; r0 += 8) {
      float dot_hk[8];
#pragma unroll
      for (int i = 0; i < 8; ++i) {
        float* hr = hreg[r0 + i];
        hr[0] *= d4[0];
        hr[1] *= d4[1];
        hr[2] *= d4[2];
        hr[3] *= d4[3];
        dot_hk[i] = fmaf(hr[0], k4[0],
                         fmaf(hr[1], k4[1], fmaf(hr[2], k4[2], hr[3] * k4[3])));
      }
      warp_sum_n<8>(dot_hk);
      float dot_hq[8];
#pragma unroll
      for (int i = 0; i < 8; ++i) {
        const int row = warp * kRowsPerWarp + r0 + i;
        const float v_new = (s_v[t][row] - dot_hk[i]) * beta;
        float* hr = hreg[r0 + i];
        hr[0] = fmaf(k4[0], v_new, hr[0]);
        hr[1] = fmaf(k4[1], v_new, hr[1]);
        hr[2] = fmaf(k4[2], v_new, hr[2]);
        hr[3] = fmaf(k4[3], v_new, hr[3]);
        dot_hq[i] = fmaf(hr[0], q4[0],
                         fmaf(hr[1], q4[1], fmaf(hr[2], q4[2], hr[3] * q4[3])));
      }
      warp_sum_n<8>(dot_hq);
#pragma unroll
      for (int i = 0; i < 8; ++i) {
        const int row = warp * kRowsPerWarp + r0 + i;
        if (lane == 0) s_o[row] = dot_hq[i];
        o_sumsq = fmaf(dot_hq[i], dot_hq[i], o_sumsq);
      }
    }
    // per-token state store to this token's slot (served semantics)
    const int fslot = state_indices[n * st.idx_row + t];
    if (fslot > 0) {
      float* dst = state + (int64_t)fslot * st.state_slot +
                   (int64_t)h * kD * kD + lane * 4;
#pragma unroll
      for (int r = 0; r < kRowsPerWarp; ++r) {
        const int row = warp * kRowsPerWarp + r;
        __stcg(reinterpret_cast<float4*>(dst + (int64_t)row * kD),
               make_float4(hreg[r][0], hreg[r][1], hreg[r][2], hreg[r][3]));
      }
    }
    if (lane == 0)
      s_red[warp] = o_sumsq;  // identical across lanes after the tree
    __syncthreads();
    if (tid < kD) {
      float sumsq = 0.0f;
#pragma unroll
      for (int w = 0; w < kWarps; ++w) sumsq += s_red[w];
      const float rstd =
          1.0f / sqrtf(sumsq / static_cast<float>(kD) + norm_eps);
      const float y = s_o[tid] * rstd * norm_w[tid] * s_og[t][tid];
      out[(int64_t)(qs + t) * st.out_row + h * kD + tid] = __float2bfloat16(y);
    }
    __syncthreads();
  }
}

template <bool kUseLowerBound>
void launch(const __nv_bfloat16* x, const float* w_t, const float* bias,
            __nv_bfloat16* conv_state, const float* a_log,
            const __nv_bfloat16* raw_g, const float* dt_bias,
            const __nv_bfloat16* raw_beta, const __nv_bfloat16* g2,
            const float* norm_w, const int* state_indices,
            const int* cu_seqlens, const int* num_accepted, float* state,
            __nv_bfloat16* out, int N, int H, float lower_bound, float scale,
            float norm_eps, SpecStrides strides, cudaStream_t stream) {
  auto kernel = &kda_spec_decode_kernel<kUseLowerBound>;
  cudaLaunchConfig_t config{};
  config.gridDim = dim3(N, H);
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
                     strides);
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
       static_cast<float>(norm_eps), strides, stream);
  };
  if (N > 0) {
    if (use_lb)
      args(launch<true>);
    else
      args(launch<false>);
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
