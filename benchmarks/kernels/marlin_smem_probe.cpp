// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
// Component-only launch adapter: reuse the deployed Marlin device function
// while varying only its dynamic shared-memory reservation. No kernel is
// recompiled here. The caller owns valid, correctly packed device buffers.

#include <cuda_runtime_api.h>

#include "core/scalar_type.hpp"

namespace marlin {
using Kernel = void (*)(const int4*, const int4*, int4*, int4*, const int4*,
                        const float*, const int4*, const float*, const int4*,
                        const int*, int, int, int, int, int, int*, bool, bool,
                        bool, int);

// This is the selector exported by the deployed vLLM stable extension.
Kernel get_marlin_kernel(vllm::ScalarType, vllm::ScalarType, vllm::ScalarType,
                         vllm::ScalarType, int, int, int, bool, bool, bool, int,
                         int, bool, int);
}  // namespace marlin

static marlin::Kernel select_kernel(int threads, int n_blocks, int k_blocks) {
  if (!((threads == 128 && n_blocks == 4 && k_blocks == 8) ||
        (threads == 256 && n_blocks == 8 && k_blocks == 8))) {
    return nullptr;
  }
  return marlin::get_marlin_kernel(
      vllm::kBFloat16, vllm::kFE4M3fn, vllm::kBFloat16, vllm::kFE8M0fnu, 1,
      n_blocks, k_blocks, true, false, false, 2, threads, false, 4);
}

extern "C" int marlin_probe_name(int threads, int n_blocks, int k_blocks,
                                 const char** name) {
  auto kernel = select_kernel(threads, n_blocks, k_blocks);
  if (!kernel) return -1;
  return static_cast<int>(
      cudaFuncGetName(name, reinterpret_cast<void*>(kernel)));
}

extern "C" int marlin_probe_launch(void* a, void* b, void* c, void* c_tmp,
                                   void* scales, void* locks, int m, int n,
                                   int k, int lda, int threads, int n_blocks,
                                   int k_blocks, int grid, int shared_bytes,
                                   int original_shared_bytes, void* stream) {
  auto kernel = select_kernel(threads, n_blocks, k_blocks);
  if (!kernel || m < 1 || m > 8 || n % (16 * n_blocks) || k % (16 * k_blocks) ||
      lda < k || lda % 8 || grid < 1) {
    return -1;
  }
  // Exact footprint of these BF16/MXFP8 M8 specializations in
  // marlin_template.h: B/reduction alias, four A stages, four E8M0 stages.
  const int tile_n = n_blocks * 16;
  const int tile_k = k_blocks * 16;
  const int required =
      4 * tile_k * tile_n + 4 * 8 * tile_k * 2 + 4 * (k_blocks / 2) * tile_n;
  if (shared_bytes < required || shared_bytes > original_shared_bytes)
    return -2;
  auto result = cudaFuncSetAttribute(
      reinterpret_cast<void*>(kernel),
      cudaFuncAttributeMaxDynamicSharedMemorySize, original_shared_bytes);
  if (result != cudaSuccess) return static_cast<int>(result);
  void* absent = nullptr;
  int groups = k / 32;
  bool bias = false;
  bool atomic = false;
  bool fp32_reduce = true;
  void* args[] = {&a,      &b,      &c,           &c_tmp,
                  &absent, &absent, &scales,      &absent,
                  &absent, &absent, &groups,      &m,
                  &n,      &k,      &lda,         &locks,
                  &bias,   &atomic, &fp32_reduce, &original_shared_bytes};
  return static_cast<int>(cudaLaunchKernel(
      reinterpret_cast<void*>(kernel), dim3(grid), dim3(threads), args,
      shared_bytes, static_cast<cudaStream_t>(stream)));
}
