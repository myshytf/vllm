# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for CUDA kernels in cache_kernels.cu."""

import pytest
import torch

try:
    from vllm import _custom_ops as ops
except ImportError:
    pytest.skip(
        "Could not import vllm._custom_ops. (pip install -e .)", allow_module_level=True
    )


def _fill_packed_kimi_rows(rows: torch.Tensor, *, edge_scales=False):
    """Populate finite E4M3 values, exact scale bytes and arbitrary RoPE bits."""
    raw = torch.arange(512, device=rows.device, dtype=torch.int32) % 256
    raw = torch.where((raw & 127) == 127, raw - 1, raw).to(torch.uint8)
    rows[..., :512].copy_(raw)
    values = (
        [1.0e-40, 1.0e-30, 1.0e30, 1.0]
        if edge_scales
        else [2.0**-10, 0.1234567, 3.75, 2.0**10]
    )
    scale = torch.tensor(values, device=rows.device, dtype=torch.float32)
    rows[..., 512:528].copy_(scale.view(torch.uint8))
    rope = torch.arange(64, device=rows.device, dtype=torch.int32) * 1013
    rows[..., 528:].view(torch.uint16).copy_(rope.to(torch.uint16))


@pytest.mark.skipif(torch.accelerator.device_count() < 1, reason="Need CUDA device")
@pytest.mark.parametrize("edge_scales", [False, True])
def test_kimi_packed_transport_matches_cuda_upconversion(edge_scales):
    """Moving the packed bytes must reproduce the serving CUDA decoder exactly."""
    from vllm.v1.attention.ops.kimi_packed_kv_transport import (
        gather_packed_records,
        unpack_packed_planes,
        warmup_packed_transport,
    )

    warmup_packed_transport(64, torch.device("cuda", 0))
    # Padding between records verifies physical cache strides independently
    # of the 656-byte semantic record size.
    storage = torch.empty((6, 64, 672), dtype=torch.uint8, device="cuda")
    cache = storage[..., :656]
    _fill_packed_kimi_rows(cache, edge_scales=edge_scales)
    table = torch.tensor([[2, 5, 1], [4, 3, 0]], dtype=torch.int32, device="cuda")
    cu = torch.tensor([0, 5, 12], dtype=torch.int32, device="cuda")
    starts = torch.tensor([62, 125], dtype=torch.int32, device="cuda")
    token_to_seq = torch.tensor([0] * 5 + [1] * 7, dtype=torch.int32, device="cuda")
    wire = torch.empty((12, 656), dtype=torch.uint8, device="cuda")
    reference = torch.empty((12, 576), dtype=torch.bfloat16, device="cuda")
    ops.cp_gather_and_upconvert_fp8_kv_cache(
        cache, reference, table, cu, 2, seq_starts=starts
    )
    gather_packed_records(cache, wire, table, token_to_seq, cu, starts)
    expected = torch.cat((cache[2, 62:], cache[5, :3], cache[3, 61:], cache[0, :4]))
    assert torch.equal(wire, expected)
    c = wire[:, :528].contiguous().view(torch.float8_e4m3fn)
    rope = wire[:, 528:].contiguous().view(torch.float8_e4m3fn)
    out_c = torch.empty((12, 512), device="cuda", dtype=torch.bfloat16)
    out_rope = torch.empty((12, 64), device="cuda", dtype=torch.bfloat16)
    unpack_packed_planes(c, rope, out_c, out_rope)
    assert torch.equal(out_c.view(torch.uint16), reference[:, :512].view(torch.uint16))
    assert torch.equal(
        out_rope.view(torch.uint16), reference[:, 512:].view(torch.uint16)
    )
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        gather_packed_records(cache, wire, table, token_to_seq, cu, starts)
        c.copy_(wire[:, :528].view(torch.float8_e4m3fn))
        rope.copy_(wire[:, 528:].view(torch.float8_e4m3fn))
        unpack_packed_planes(c, rope, out_c, out_rope)
    for _ in range(5):
        out_c.fill_(float("nan"))
        graph.replay()
        torch.accelerator.synchronize()
        assert torch.equal(
            out_c.view(torch.uint16), reference[:, :512].view(torch.uint16)
        )
        assert torch.equal(
            out_rope.view(torch.uint16), reference[:, 512:].view(torch.uint16)
        )


@pytest.mark.skipif(torch.accelerator.device_count() < 1, reason="Need CUDA device")
def test_kimi_packed_gather_preserves_page_offsets_above_int32():
    """Use mapped pinned storage to test >2-GiB offsets without spare GPU VRAM."""
    from vllm.utils.torch_utils import get_accelerator_view_from_cpu_tensor
    from vllm.v1.attention.ops.kimi_packed_kv_transport import gather_packed_records

    block_bytes = 64 * 656
    page = (2**31 // block_bytes) + 3
    pool = torch.empty(
        (page + 1, 64, 656), dtype=torch.uint8, device="cpu", pin_memory=True
    )
    _fill_packed_kimi_rows(pool[page])
    cache = get_accelerator_view_from_cpu_tensor(pool)
    assert cache.is_cuda and page * cache.stride(0) > 2**31
    table = torch.tensor([[page]], dtype=torch.int32, device="cuda")
    zero = torch.zeros(1, dtype=torch.int32, device="cuda")
    starts = torch.tensor([7], dtype=torch.int32, device="cuda")
    wire = torch.empty((1, 656), dtype=torch.uint8, device="cuda")
    gather_packed_records(cache, wire, table, zero, zero, starts)
    assert torch.equal(wire.cpu()[0], pool[page, 7])


@pytest.mark.skipif(torch.accelerator.device_count() < 1, reason="Need CUDA device")
def test_gather_cache_oob():
    """
    Tests for OOB read in gather_and_maybe_dequant_cache (Issue #27909).
    This test constructs a boundary case identified in the issue where
    seq_starts causes the block_table offset to read out of bounds.
    """

    block_size = 64
    # The kernel only supports the MLA entry sizes.
    entry_size = 576

    block_table = torch.tensor([[1, 2]], dtype=torch.int32, device="cuda")

    # This will result in offset = 128 / block_size = 128 / 64 = 2
    # This will cause the kernel to try to read from
    # block_table[0, 2], but its size is only 2.
    seq_starts = torch.tensor([128], dtype=torch.int32, device="cuda")

    seq_len = 65
    cu_seq_lens = torch.tensor([0, seq_len], dtype=torch.int32, device="cuda")
    token_to_seq = torch.zeros(seq_len, dtype=torch.int32, device="cuda")

    # src_cache: [num_blocks, block_size, entry_size]
    num_blocks = 5
    src_cache = torch.randn(
        (num_blocks, block_size, entry_size), dtype=torch.float16, device="cuda"
    )

    dst = torch.empty((seq_len, entry_size), dtype=torch.float16, device="cuda")

    scale = torch.tensor([1.0], dtype=torch.float32, device="cuda")

    # Calling the C++ function gather_and_maybe_dequant_cache
    ops.gather_and_maybe_dequant_cache(
        src_cache,
        dst,
        block_table,
        cu_seq_lens,
        token_to_seq,
        seq_len,
        "auto",  # kv_cache_dtype
        scale,
        seq_starts,
    )

    torch.accelerator.synchronize()
    assert True


if __name__ == "__main__":
    pytest.main([__file__])
