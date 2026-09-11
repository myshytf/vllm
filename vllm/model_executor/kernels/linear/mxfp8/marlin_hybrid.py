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
from vllm.triton_utils import tl, triton

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


def reconstruct_bf16_weight_torch(
    marlin_qweight: torch.Tensor,
    marlin_scales: torch.Tensor,
    size_n: int,
    size_k: int,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Exact ``[size_n, size_k]`` weight in ``dtype`` from the payload (torch ops)."""
    w = unrepack_marlin_fp8_weight(marlin_qweight, size_n, size_k).to(dtype)
    s = unrepack_marlin_mxfp8_scales(marlin_scales, size_n, size_k, dtype)
    # power-of-two scales: the product is exact in bf16/fp16 within range
    return (w.view(size_n, size_k // MXFP8_GROUP, MXFP8_GROUP) * s.unsqueeze(-1)).view(
        size_n, size_k
    )


@functools.cache
def _scale_inverse_table(device: torch.device) -> torch.Tensor:
    """64-entry table: original n_local -> position inside a Marlin 64-scale chunk.

    Inverts ``marlin_permute_scales`` (scale_perm over 64 columns) followed by
    ``mxfp8_marlin_process_scales`` (the [0, 2, 1, 3] swap inside groups of 4).
    """
    scale_perm, _ = get_scale_perms()
    inv = torch.argsort(torch.tensor(scale_perm))  # orig n -> position after perm
    swap = torch.tensor([0, 2, 1, 3])
    pos = (inv // 4) * 4 + swap[inv % 4]
    return pos.to(dtype=torch.int32, device=device)


@triton.jit
def _marlin_fp8_unrepack_dequant_kernel(
    packed_ptr,
    inv_ptr,
    scale_pos_ptr,
    marlin_scale_ptr,
    out_ptr,
    padded_n,
    size_n,
    size_k,
    K_GROUPS: tl.constexpr,
    TILE_ELEMS: tl.constexpr,
):
    """Program = K_GROUPS consecutive 16-row k groups x one 64-column n group.

    Every 16 x 64 Marlin tile is a contiguous 1 KB span of the packed payload;
    lanes are ordered n-major / k-minor so that each n row stores 16*K_GROUPS
    contiguous BF16 values. Scales are gathered from the Marlin scale layout
    in-kernel (one e8m0 byte per n, per 32-wide k group).
    """
    kg = tl.program_id(0)
    nc = tl.program_id(1)
    lane = tl.arange(0, TILE_ELEMS)
    n_local = lane // 16
    k_in = lane % 16
    # tile-internal index of (n_local, k_in): [4 n-tiles][16 k][16 n]
    g = (n_local // 16) * 256 + k_in * 16 + (n_local % 16)
    src = tl.load(inv_ptr + g)
    n_global = nc * 64 + n_local
    n_mask = n_global < size_n
    scale_pos = tl.load(scale_pos_ptr + n_local)
    for i in tl.static_range(K_GROUPS):
        kr = kg * K_GROUPS + i
        tile_base = kr.to(tl.int64) * (padded_n * 16) + nc * TILE_ELEMS
        raw = tl.load(packed_ptr + tile_base + src)
        k_global = kr * 16 + k_in
        # scale row = 32-wide k group; 16 | 32 so it is constant per tile
        s_row = (kr // 2).to(tl.int64) * padded_n + nc * 64
        e8m0 = tl.load(marlin_scale_ptr + s_row + scale_pos, mask=n_mask, other=127)
        scale = tl.exp2(e8m0.to(tl.float32) - 127.0)
        val = raw.to(tl.float8e4nv, bitcast=True).to(tl.float32) * scale
        mask = n_mask & (k_global < size_k)
        tl.store(
            out_ptr + n_global.to(tl.int64) * size_k + k_global,
            val.to(tl.bfloat16),
            mask=mask,
        )


def reconstruct_bf16_weight_triton(
    marlin_qweight: torch.Tensor,
    marlin_scales: torch.Tensor,
    size_n: int,
    size_k: int,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Exact ``[size_n, size_k]`` weight in ``dtype`` with one fused gather kernel."""
    if dtype != torch.bfloat16:
        return reconstruct_bf16_weight_torch(
            marlin_qweight, marlin_scales, size_n, size_k, dtype
        )
    padded_n, padded_k = marlin_repacked_nk(marlin_qweight, num_bits=8)
    inv_weight, _ = _inverse_perms(marlin_qweight.device)
    out = torch.empty(size_n, size_k, dtype=dtype, device=marlin_qweight.device)
    packed = marlin_qweight.contiguous().view(torch.uint8).view(-1)
    scales_u8 = marlin_scales.contiguous().view(torch.uint8).view(-1)
    k_groups = 4 if (padded_k // GPTQ_MARLIN_TILE) % 4 == 0 else 2
    grid = (padded_k // GPTQ_MARLIN_TILE // k_groups, padded_n // 64)
    _marlin_fp8_unrepack_dequant_kernel[grid](
        packed,
        inv_weight.to(torch.int32),
        _scale_inverse_table(marlin_qweight.device),
        scales_u8,
        out,
        padded_n,
        size_n,
        size_k,
        K_GROUPS=k_groups,
        TILE_ELEMS=1024,
    )
    return out


def reconstruct_bf16_weight(
    marlin_qweight: torch.Tensor,
    marlin_scales: torch.Tensor,
    size_n: int,
    size_k: int,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Exact ``[size_n, size_k]`` weight in ``dtype`` from the Marlin payload."""
    if envs.VLLM_MXFP8_HYBRID_RECONSTRUCT == "triton":
        return reconstruct_bf16_weight_triton(
            marlin_qweight, marlin_scales, size_n, size_k, dtype
        )
    return reconstruct_bf16_weight_torch(
        marlin_qweight, marlin_scales, size_n, size_k, dtype
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
