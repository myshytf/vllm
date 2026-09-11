# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Marlin MXFP8 W8A16 for small M, exact BF16 GEMM for large M.

Marlin's W8A16 kernel is weight-bandwidth-bound and efficient at decode
batch sizes, but at prefill batch sizes (thousands of rows) it reaches only
a fraction of the BF16 tensor-core throughput. This kernel keeps Marlin for
``M <= large_m_threshold`` and, above it, reconstructs the exact BF16 weight
matrix from the Marlin-packed MXFP8 payload on the fly and runs a cuBLAS
BF16 GEMM. Both paths use the same operands (BF16 activations, exactly
reconstructed BF16 weights) with FP32 accumulation; they differ only in
summation order. No extra resident memory: the reconstruction is a transient
of one layer shard per call.
"""

from __future__ import annotations

import functools

import torch

from vllm import envs
from vllm.logger import init_logger
from vllm.model_executor.layers.quantization.utils.marlin_utils import (
    GPTQ_MARLIN_TILE,
    get_scale_perms,
    marlin_repacked_nk,
)
from vllm.model_executor.layers.quantization.utils.marlin_utils_test import (
    get_weight_perm,
)

from .marlin import MarlinMxfp8LinearKernel

logger = init_logger(__name__)

MXFP8_GROUP = 32
_FP8_PACK = 4  # fp8 values per int32


@functools.cache
def _inverse_perms(device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    """Inverse of Marlin's 8-bit weight permutation and of the scale permutation."""
    weight_perm = get_weight_perm(num_bits=8)
    inv_weight = torch.argsort(weight_perm).to(device=device)
    scale_perm, _ = get_scale_perms()
    inv_scale = torch.argsort(torch.tensor(scale_perm)).to(device=device)
    return inv_weight, inv_scale


def unrepack_marlin_fp8_weight(
    marlin_qweight: torch.Tensor, size_n: int, size_k: int
) -> torch.Tensor:
    """Exact inverse of ``pack_fp8_to_int32`` + ``gptq_marlin_repack`` (8-bit).

    Returns the ``[size_n, size_k]`` float8_e4m3fn weight (padding removed).
    """
    padded_n, padded_k = marlin_repacked_nk(marlin_qweight, num_bits=8)
    inv_weight, _ = _inverse_perms(marlin_qweight.device)
    tile = GPTQ_MARLIN_TILE
    # int32 [K/16, N*16/4] -> uint8 [K/16, N*16] (little-endian byte order equals
    # the reference packing q_packed |= q_w[:, i::4] << 8*i).
    rows = padded_k // tile
    q = marlin_qweight.contiguous().view(torch.uint8).view(rows, padded_n * tile)
    # undo the 1024-wide in-row permutation
    q = q.reshape(-1, inv_weight.numel())[:, inv_weight].reshape(rows, padded_n * tile)
    # undo the 16x16 tile transpose: [K/16, N/16, 16, 16] -> [K/16, 16, N/16, 16]
    q = q.reshape(rows, padded_n // tile, tile, tile).permute(0, 2, 1, 3)
    q = q.reshape(padded_k, padded_n)
    # gptq orientation [K, N] -> weight [N, K]; drop padding
    return q.t()[:size_n, :size_k].contiguous().view(torch.float8_e4m3fn)


def unrepack_marlin_mxfp8_scales(
    marlin_scales: torch.Tensor, size_n: int, size_k: int, dtype: torch.dtype
) -> torch.Tensor:
    """Exact inverse of the MXFP8 Marlin scale layout.

    Returns ``[size_n, size_k // 32]`` scale factors (powers of two) in ``dtype``.
    """
    _, inv_scale = _inverse_perms(marlin_scales.device)
    s = marlin_scales.to(dtype)
    padded_rows, padded_n = s.shape  # [padded_k // 32, padded_n]
    # undo mxfp8_marlin_process_scales: view(-1, 4)[:, [0, 2, 1, 3]] (an involution)
    s = s.reshape(-1, 4)[:, [0, 2, 1, 3]].reshape(padded_rows, padded_n)
    # undo marlin_permute_scales: reshape(-1, 64)[:, scale_perm]
    s = s.reshape(-1, inv_scale.numel())[:, inv_scale].reshape(padded_rows, padded_n)
    return s.t()[:size_n, : size_k // MXFP8_GROUP].contiguous()


def reconstruct_bf16_weight(
    marlin_qweight: torch.Tensor,
    marlin_scales: torch.Tensor,
    size_n: int,
    size_k: int,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Exact ``[size_n, size_k]`` weight in ``dtype`` from the Marlin payload."""
    w = unrepack_marlin_fp8_weight(marlin_qweight, size_n, size_k).to(dtype)
    s = unrepack_marlin_mxfp8_scales(marlin_scales, size_n, size_k, dtype)
    # power-of-two scales: the product is exact in bf16/fp16 within range
    return (w.view(size_n, size_k // MXFP8_GROUP, MXFP8_GROUP) * s.unsqueeze(-1)).view(
        size_n, size_k
    )


class MarlinMxfp8HybridLinearKernel(MarlinMxfp8LinearKernel):
    """Marlin W8A16 below the threshold, exact BF16 GEMM above it."""

    @classmethod
    def large_m_threshold(cls) -> int:
        return envs.VLLM_MXFP8_MARLIN_LARGE_M_THRESHOLD

    def apply_weights(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        rows = x.numel() // x.shape[-1]
        if rows <= self.large_m_threshold():
            return super().apply_weights(layer, x, bias)
        size_n = layer.output_size_per_partition
        size_k = layer.input_size_per_partition
        weight = reconstruct_bf16_weight(
            layer.weight, layer.weight_scale, size_n, size_k, x.dtype
        )
        return torch.nn.functional.linear(x, weight, bias)
