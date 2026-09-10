# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Move native 656-byte MLA cache records before expanding them to BF16.

The wire representation retains all 512 E4M3 latent bytes, four FP32 scales,
and 64 BF16 RoPE values. Transport views use the existing byte-sized dtype
accepted by the DCP publisher; no fields are converted on the wire.
"""

import functools

import torch

from vllm.triton_utils import tl, triton


@triton.jit(do_not_specialize=["TABLE_STRIDE", "BLOCK_STRIDE", "TOKEN_STRIDE"])
def _gather_records(
    cache,
    output,
    block_table,
    token_to_seq,
    workspace_starts,
    seq_starts,
    BLOCK_SIZE: tl.constexpr,
    TABLE_STRIDE,
    BLOCK_STRIDE,
    TOKEN_STRIDE,
):
    row = tl.program_id(0)
    request = tl.load(token_to_seq + row).to(tl.int64)
    offset = (
        row.to(tl.int64)
        - tl.load(workspace_starts + request).to(tl.int64)
        + tl.load(seq_starts + request).to(tl.int64)
    )
    page = tl.load(block_table + request * TABLE_STRIDE + offset // BLOCK_SIZE)
    source = page.to(tl.int64) * BLOCK_STRIDE + (offset % BLOCK_SIZE) * TOKEN_STRIDE
    column = tl.arange(0, 1024)
    data = tl.load(cache + source + column, column < 656, other=0)
    tl.store(output + row.to(tl.int64) * 656 + column, data, column < 656)


@triton.jit
def _unpack_planes(packed_c, scales, packed_rope, output_c, output_rope):
    row = tl.program_id(0).to(tl.int64)
    column = tl.arange(0, 512)
    encoded = tl.load(packed_c + row * 528 + column)
    latent = encoded.to(tl.float8e4nv, bitcast=True).to(tl.float32)
    scale = tl.load(scales + row * 132 + column // 128)
    value = (latent * scale).to(tl.bfloat16)
    tl.store(output_c + row * 512 + column, value)
    rope_column = tl.arange(0, 64)
    rope = tl.load(packed_rope + row * 64 + rope_column)
    tl.store(output_rope + row * 64 + rope_column, rope)


def gather_packed_records(
    cache: torch.Tensor,
    output: torch.Tensor,
    block_table: torch.Tensor,
    token_to_seq: torch.Tensor,
    workspace_starts: torch.Tensor,
    seq_starts: torch.Tensor,
) -> None:
    """Gather paged records without interpreting or changing their bytes."""
    if (
        cache.dtype != torch.uint8
        or cache.ndim != 3
        or cache.shape[-1] != 656
        or cache.stride(-1) != 1
        or output.dtype != torch.uint8
        or output.ndim != 2
        or output.shape[1] != 656
        or not output.is_contiguous()
        or not cache.is_cuda
        or output.device != cache.device
    ):
        raise ValueError(
            "packed KV gather requires CUDA uint8 cache and contiguous [T,656] output"
        )
    if token_to_seq.numel() != output.shape[0]:
        raise ValueError("packed KV gather token map must cover every output row")
    if output.shape[0] == 0:
        return
    with torch.accelerator.device_index(cache.device.index):
        _gather_records[(output.shape[0],)](
            cache,
            output,
            block_table,
            token_to_seq,
            workspace_starts,
            seq_starts,
            cache.shape[1],
            block_table.stride(0),
            cache.stride(0),
            cache.stride(1),
            num_warps=4,
        )


def unpack_packed_planes(
    packed_c: torch.Tensor,
    packed_rope: torch.Tensor,
    output_c: torch.Tensor,
    output_rope: torch.Tensor,
) -> None:
    """Expand native FP8 values with FP32 multiplication and one BF16 rounding."""
    rows = packed_c.shape[0]
    if (
        packed_c.shape != (rows, 528)
        or packed_rope.shape != (rows, 128)
        or packed_c.element_size() != 1
        or packed_rope.element_size() != 1
        or output_c.shape != (rows, 512)
        or output_rope.shape != (rows, 64)
        or output_c.dtype != torch.bfloat16
        or output_rope.dtype != torch.bfloat16
        or not all(
            t.is_contiguous() for t in (packed_c, packed_rope, output_c, output_rope)
        )
        or not packed_c.is_cuda
        or any(
            t.device != packed_c.device for t in (packed_rope, output_c, output_rope)
        )
    ):
        raise ValueError(
            "packed KV unpack requires byte planes [T,528]/[T,128] "
            "and BF16 outputs [T,512]/[T,64]"
        )
    if rows == 0:
        return
    raw_c = packed_c.view(torch.uint8)
    scales = raw_c[:, 512:528].view(torch.float32)
    with torch.accelerator.device_index(packed_c.device.index):
        _unpack_planes[(rows,)](
            raw_c,
            scales,
            packed_rope.view(torch.uint16),
            output_c,
            output_rope.view(torch.uint16),
            num_warps=4,
        )


@functools.cache
def warmup_packed_transport(block_size: int, device: torch.device) -> None:
    """Compile and load both operations before a request can use packed KV."""
    cache = torch.zeros((1, block_size, 656), dtype=torch.uint8, device=device)
    cache[0, 0, 512:528].view(torch.float32).fill_(1)
    wire = torch.empty((1, 656), dtype=torch.uint8, device=device)
    index = torch.zeros(1, dtype=torch.int32, device=device)
    gather_packed_records(cache, wire, index.view(1, 1), index, index, index)
    c = wire[:, :528].contiguous()
    rope = wire[:, 528:].contiguous()
    unpack_packed_planes(
        c,
        rope,
        torch.empty((1, 512), dtype=torch.bfloat16, device=device),
        torch.empty((1, 64), dtype=torch.bfloat16, device=device),
    )
