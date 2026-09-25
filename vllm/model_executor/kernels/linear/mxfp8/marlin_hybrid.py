# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Marlin MXFP8 W8A16 for small M, exact BF16 GEMM for large M.

With ``VLLM_K3_W8A16_GEMV_LIB`` set, decode batches of at most 8 rows of the
selected layers use ``_C_k3decode.w8a16_gemv`` instead of Marlin: it reads the
same Marlin payload, computes the same exact products and reduces them in a
fixed order on the CUDA cores.

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
from collections.abc import Callable

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
from vllm.platforms import current_platform
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


@triton.jit
def _marlin_fp8_unrepack_dequant_kernel(
    packed_ptr,
    marlin_scale_ptr,
    out_ptr,
    padded_n: tl.constexpr,
    size_n: tl.constexpr,
    size_k: tl.constexpr,
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
    # Exact inverse of get_weight_perm(8), expressed as a bit permutation.
    # The 1024-entry lookup and its per-call int64 -> int32 cast disappear.
    src = (
        (k_in >> 3)
        | ((k_in & 1) << 1)
        | ((n_local >> 3) << 2)
        | ((k_in & 6) << 4)
        | ((n_local & 7) << 7)
    )
    n_global = nc * 64 + n_local
    n_mask = n_global < size_n
    scale_pos = (
        ((n_local & 16) >> 4)
        | ((n_local & 8) >> 2)
        | ((n_local & 32) >> 3)
        | ((n_local & 7) << 3)
    )
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
    out = torch.empty(size_n, size_k, dtype=dtype, device=marlin_qweight.device)
    packed = marlin_qweight.contiguous().view(torch.uint8).view(-1)
    scales_u8 = marlin_scales.contiguous().view(torch.uint8).view(-1)
    k_groups = 4 if (padded_k // GPTQ_MARLIN_TILE) % 4 == 0 else 2
    grid = (padded_k // GPTQ_MARLIN_TILE // k_groups, padded_n // 64)
    _marlin_fp8_unrepack_dequant_kernel[grid](
        packed,
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


@functools.cache
def load_w8a16_gemv_op() -> Callable[..., None] | None:
    """``_C_k3decode.w8a16_gemv`` from ``VLLM_K3_W8A16_GEMV_LIB``, or None
    (Marlin)."""
    path = envs.VLLM_K3_W8A16_GEMV_LIB
    if not path:
        return None
    torch.ops.load_library(path)
    return torch.ops._C_k3decode.w8a16_gemv


@functools.cache
def _multiprocessor_count(device_index: int) -> int:
    return torch.cuda.get_device_properties(device_index).multi_processor_count


def can_w8a16_gemv(
    layer: torch.nn.Module,
    x: torch.Tensor,
    bias: torch.Tensor | None,
    *,
    require_op: bool = True,
) -> bool:
    """Whether ``w8a16_gemv`` serves this call: at most
    ``VLLM_K3_W8A16_GEMV_MAX_ROWS`` (<= 8) bf16 rows with 16-byte aligned rows,
    no bias, and a layer prefix selected by ``VLLM_K3_W8A16_GEMV_LAYERS``."""
    if require_op and load_w8a16_gemv_op() is None:
        return False
    rows = x.numel() // x.shape[-1] if x.shape[-1] else 0
    if bias is not None or not 0 < rows <= min(8, envs.VLLM_K3_W8A16_GEMV_MAX_ROWS):
        return False
    if x.dtype != torch.bfloat16 or x.shape[-1] != layer.input_size_per_partition:
        return False
    selected = [s for s in envs.VLLM_K3_W8A16_GEMV_LAYERS.split(",") if s]
    prefix = getattr(layer, "prefix", "")
    if selected and not any(s in prefix for s in selected):
        return False
    x2 = x.reshape(rows, x.shape[-1])
    if not (x2.stride(-1) == 1 and x2.stride(0) % 8 == 0 and x2.data_ptr() % 16 == 0):
        return False
    if not require_op:
        return True
    padded_n, padded_k = marlin_repacked_nk(layer.weight, num_bits=8)
    cluster, warps = w8a16_gemv_layout(padded_n, padded_k, x.device.index or 0)
    return w8a16_gemv_smem_bytes(rows, padded_k, cluster, warps) <= _GEMV_MAX_SMEM


# Dynamic shared memory the GEMV may request per CTA (sm_120 allows 99 KiB).
_GEMV_MAX_SMEM = 96 * 1024


def w8a16_gemv_smem_bytes(rows: int, padded_k: int, cluster: int, warps: int) -> int:
    """Dynamic shared memory of one ``w8a16_gemv`` CTA: the CTA's K slice of
    the activations (bf16, rows padded to 1/2/4/8) plus the per-warp and
    per-CTA column sums."""
    padded_rows = 1 if rows <= 1 else 2 if rows <= 2 else 4 if rows <= 4 else 8
    slice_groups = -(-(padded_k // 32) // cluster)
    x_bytes = -(-(padded_rows * slice_groups * 32 * 2) // 16) * 16
    return x_bytes + (warps + 1) * 64 * padded_rows * 4


@functools.cache
def _parse_gemv_table(table: str) -> dict[tuple[int, int], tuple[int, int]]:
    layouts = {}
    for entry in filter(None, table.split(",")):
        shape, layout = entry.split(":")
        n, k = shape.split("x")
        cluster, warps = layout.split("x")
        layouts[(int(n), int(k))] = (int(cluster), int(warps))
    return layouts


def w8a16_gemv_layout(
    padded_n: int, padded_k: int, device_index: int
) -> tuple[int, int]:
    """(CTAs per 64-column group, warps per CTA): the
    ``VLLM_K3_W8A16_GEMV_TABLE`` entry of this shape, else
    ``VLLM_K3_W8A16_GEMV_CLUSTER`` (0 = the smallest K split that gives at
    least two CTAs per SM, at most 8) and ``VLLM_K3_W8A16_GEMV_WARPS``."""
    table = _parse_gemv_table(envs.VLLM_K3_W8A16_GEMV_TABLE)
    layout = table.get((padded_n, padded_k))
    if layout is not None:
        return layout
    warps = envs.VLLM_K3_W8A16_GEMV_WARPS
    if envs.VLLM_K3_W8A16_GEMV_CLUSTER:
        return envs.VLLM_K3_W8A16_GEMV_CLUSTER, warps
    column_groups = padded_n // 64
    target = 2 * _multiprocessor_count(device_index)
    for cluster in (1, 2, 4):
        if column_groups * cluster >= target:
            return cluster, warps
    return 8, warps


def apply_w8a16_gemv(layer: torch.nn.Module, x: torch.Tensor) -> torch.Tensor:
    op = load_w8a16_gemv_op()
    assert op is not None
    size_n = layer.output_size_per_partition
    size_k = layer.input_size_per_partition
    x2 = x.reshape(-1, size_k)
    out = torch.empty(x2.shape[0], size_n, dtype=x.dtype, device=x.device)
    padded_n, padded_k = marlin_repacked_nk(layer.weight, num_bits=8)
    cluster, warps = w8a16_gemv_layout(padded_n, padded_k, x.device.index or 0)
    op(
        out,
        x2,
        layer.weight,
        layer.weight_scale.view(torch.uint8),
        size_n,
        size_k,
        cluster,
        warps,
        current_platform.is_arch_support_pdl(),
    )
    return out.reshape(*x.shape[:-1], size_n)


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
            if can_w8a16_gemv(layer, x, bias):
                return apply_w8a16_gemv(layer, x)
            return super().apply_weights(layer, x, bias)
        size_n = layer.output_size_per_partition
        size_k = layer.input_size_per_partition
        weight = reconstruct_bf16_weight(
            layer.weight, layer.weight_scale, size_n, size_k, x.dtype
        )
        return torch.nn.functional.linear(x, weight, bias)
