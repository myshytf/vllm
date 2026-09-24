# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Exactness of the Marlin MXFP8 payload inversion used by the hybrid kernel."""

import os

import pytest
import torch

from vllm.model_executor.kernels.linear.mxfp8.marlin_hybrid import (
    reconstruct_bf16_weight,
    unrepack_marlin_fp8_weight,
    unrepack_marlin_mxfp8_scales,
)
from vllm.model_executor.layers.quantization.utils.marlin_utils import (
    marlin_pad_qweight,
    marlin_pad_scales,
    marlin_padded_nk,
    marlin_permute_scales,
)
from vllm.model_executor.layers.quantization.utils.marlin_utils_fp8 import (
    mxfp8_marlin_process_scales,
    pack_fp8_to_int32,
)
from vllm.model_executor.layers.quantization.utils.marlin_utils_test import (
    get_weight_perm,
    marlin_weights,
)


def _reference_repack(weight: torch.Tensor, scales_u8: torch.Tensor, dtype):
    """Python reference of prepare_mxfp8_layer_for_marlin (CPU)."""
    size_n, size_k = weight.shape
    padded_n, padded_k = marlin_padded_nk(size_n, size_k, 32)
    qweight = pack_fp8_to_int32(weight, size_k_first=False).T.contiguous()
    qweight = marlin_pad_qweight(qweight, size_n, size_k, padded_n, padded_k)
    # unpacked gptq layout [K, N] as uint8 values for the reference permuter
    # int32 (k // 4, n) holds k = 4r + byte (little-endian): bytes are the
    # innermost view dimension, so put them next to r before flattening K.
    unpacked = qweight.contiguous().view(torch.uint8).view(padded_k // 4, padded_n, 4)
    unpacked = unpacked.permute(0, 2, 1).reshape(padded_k, padded_n)
    marlin_q = marlin_weights(
        unpacked.to(torch.int32), padded_k, padded_n, 8, get_weight_perm(8)
    )
    s = scales_u8.view(torch.float8_e8m0fnu).to(dtype).T.contiguous()
    s = marlin_pad_scales(s, size_n, size_k, padded_n, padded_k, 32)
    s = marlin_permute_scales(s, padded_k, padded_n, 32)
    return marlin_q, mxfp8_marlin_process_scales(s)


def test_reconstruction_bit_permutations_cover_the_marlin_layout():
    """Every tile lane and scale column selects the original packed byte."""
    from vllm.model_executor.layers.quantization.utils.marlin_utils import (
        get_scale_perms,
    )

    lane = torch.arange(1024)
    n, k = lane // 16, lane % 16
    source = (
        (k >> 3) | ((k & 1) << 1) | ((n >> 3) << 2) | ((k & 6) << 4) | ((n & 7) << 7)
    )
    tiled = (n // 16) * 256 + k * 16 + n % 16
    assert torch.equal(source, torch.argsort(get_weight_perm(8))[tiled])

    n = torch.arange(64)
    scale = ((n & 16) >> 4) | ((n & 8) >> 2) | ((n & 32) >> 3) | ((n & 7) << 3)
    inverse = torch.argsort(torch.tensor(get_scale_perms()[0]))
    expected = (inverse // 4) * 4 + torch.tensor([0, 2, 1, 3])[inverse % 4]
    assert torch.equal(scale, expected)


@pytest.mark.parametrize("size_n,size_k", [(256, 512), (1365, 7168), (7168, 704)])
def test_unrepack_matches_reference(size_n, size_k, monkeypatch):
    # This part of the oracle uses CPU packing; GPU reconstruction is covered
    # below independently of the serving image's backend preference.
    monkeypatch.setenv("VLLM_MXFP8_HYBRID_RECONSTRUCT", "torch")
    torch.manual_seed(0)
    weight = (torch.randn(size_n, size_k) / 4).to(torch.float8_e4m3fn)
    scales = torch.randint(118, 132, (size_n, size_k // 32), dtype=torch.uint8)
    marlin_q, marlin_s = _reference_repack(weight, scales, torch.bfloat16)

    w_back = unrepack_marlin_fp8_weight(marlin_q, size_n, size_k)
    assert torch.equal(w_back.view(torch.uint8), weight.view(torch.uint8))

    s_back = unrepack_marlin_mxfp8_scales(marlin_s, size_n, size_k, torch.bfloat16)
    s_ref = scales.view(torch.float8_e8m0fnu).to(torch.bfloat16)
    assert torch.equal(s_back, s_ref)

    full = reconstruct_bf16_weight(marlin_q, marlin_s, size_n, size_k, torch.bfloat16)
    ref = weight.to(torch.bfloat16).view(size_n, size_k // 32, 32) * s_ref.unsqueeze(-1)
    assert torch.equal(full, ref.view(size_n, size_k))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs the CUDA repack")
@pytest.mark.parametrize("size_n,size_k", [(256, 512), (1408, 7168), (7168, 704)])
def test_triton_reconstruction_matches_torch_on_cuda_repack(size_n, size_k):
    from vllm.model_executor.kernels.linear.mxfp8.marlin_hybrid import (
        reconstruct_bf16_weight_torch,
        reconstruct_bf16_weight_triton,
    )
    from vllm.model_executor.layers.quantization.utils.marlin_utils_fp8 import (
        prepare_mxfp8_layer_for_marlin,
    )
    from vllm.utils.torch_utils import set_default_torch_dtype

    torch.manual_seed(0)
    dev = torch.device("cuda")
    weight = (torch.randn(size_n, size_k, device=dev) / 4).to(torch.float8_e4m3fn)
    scales = torch.randint(
        118, 132, (size_n, size_k // 32), dtype=torch.uint8, device=dev
    )
    ref = (
        weight.to(torch.bfloat16).view(size_n, size_k // 32, 32)
        * scales.view(torch.float8_e8m0fnu).to(torch.bfloat16).unsqueeze(-1)
    ).view(size_n, size_k)
    layer = torch.nn.Module()
    layer.weight = torch.nn.Parameter(weight.clone(), requires_grad=False)
    layer.weight_scale = torch.nn.Parameter(scales.clone(), requires_grad=False)
    layer.output_size_per_partition = size_n
    layer.input_size_per_partition = size_k
    with set_default_torch_dtype(torch.bfloat16):
        prepare_mxfp8_layer_for_marlin(layer)

    w_torch = reconstruct_bf16_weight_torch(
        layer.weight, layer.weight_scale, size_n, size_k, torch.bfloat16
    )
    w_triton = reconstruct_bf16_weight_triton(
        layer.weight, layer.weight_scale, size_n, size_k, torch.bfloat16
    )
    assert torch.equal(w_torch, ref)
    assert torch.equal(w_triton, ref)


def test_gemv_lane_chunks_cover_each_tile_once():
    """`w8a16_gemv` lane l reads 16-byte chunks l and l + 32 of every tile;
    chunk byte j holds column (l >> 3 & 3) | h << 2 | (j >> 2) << 3 |
    (l & 1) << 5 and row 2 * (l >> 1 & 3) + (0, 8, 1, 9)[j & 3], and the
    scale word at (l & 1) << 2 | (column & 7) << 3 holds the scales of the
    chunk's four columns in byte order q = 0, 2, 1, 3."""
    lane, h, j = torch.meshgrid(
        torch.arange(32), torch.arange(2), torch.arange(16), indexing="ij"
    )
    n = ((lane >> 3) & 3) | (h << 2) | ((j >> 2) << 3) | ((lane & 1) << 5)
    k = 2 * ((lane >> 1) & 3) + torch.tensor([0, 8, 1, 9])[j & 3]
    byte = (lane + 32 * h) * 16 + j
    source = (
        (k >> 3) | ((k & 1) << 1) | ((n >> 3) << 2) | ((k & 6) << 4) | ((n & 7) << 7)
    )
    assert torch.equal(byte, source)
    assert torch.equal(torch.sort((n * 16 + k).flatten()).values, torch.arange(1024))

    q = j >> 2
    word = ((lane & 1) << 2) | ((n & 7) << 3)
    scale_byte = word + ((q >> 1) | ((q & 1) << 1))
    scale = ((n & 16) >> 4) | ((n & 8) >> 2) | ((n & 32) >> 3) | ((n & 7) << 3)
    assert torch.equal(scale_byte, scale)


@pytest.mark.parametrize(
    "change", [None, "library", "rows", "bias", "dtype", "layer", "misaligned"]
)
def test_gemv_gate(monkeypatch, change):
    """Only bias-free decode batches (at most 8 bf16 rows, 16-byte aligned) of
    the selected layers reach `w8a16_gemv`, and only when its library is
    set."""
    from vllm.model_executor.kernels.linear.mxfp8.marlin_hybrid import (
        can_w8a16_gemv,
    )

    monkeypatch.setenv("VLLM_K3_W8A16_GEMV_LIB", "")
    layer = torch.nn.Module()
    layer.input_size_per_partition = 256
    layer.prefix = (
        "model.layers.0.mlp.shared_experts.down_proj"
        if change == "layer"
        else "model.layers.0.self_attn.o_proj"
    )
    x = torch.zeros(16 if change == "rows" else 4, 256, dtype=torch.bfloat16)
    bias = torch.zeros(8) if change == "bias" else None
    if change == "dtype":
        x = x.float()
    elif change == "misaligned":
        x = torch.zeros(4 * 256 + 1, dtype=torch.bfloat16)[1:].view(4, 256)

    accepted = can_w8a16_gemv(layer, x, bias, require_op=change == "library")

    assert accepted == (change is None)


@pytest.mark.skipif(
    not os.environ.get("VLLM_K3_W8A16_GEMV_LIB"),
    reason="VLLM_K3_W8A16_GEMV_LIB names no _C_k3decode library",
)
@pytest.mark.parametrize("rows", [1, 3, 4, 8])
@pytest.mark.parametrize(
    "size_n,size_k", [(4224, 7168), (7168, 1408), (2112, 1536), (320, 704)]
)
@pytest.mark.parametrize("cluster", [1, 4, 8])
def test_gemv_matches_float64(monkeypatch, rows, size_n, size_k, cluster):
    """`w8a16_gemv` reads the CUDA-repacked payload: every output is within one
    bf16 ulp of the float64 product of the exact weights, and repeated calls
    are bitwise identical."""
    from vllm.model_executor.kernels.linear.mxfp8.marlin_hybrid import (
        apply_w8a16_gemv,
        can_w8a16_gemv,
    )
    from vllm.model_executor.layers.quantization.utils.marlin_utils_fp8 import (
        prepare_mxfp8_layer_for_marlin,
    )
    from vllm.utils.torch_utils import set_default_torch_dtype

    monkeypatch.setenv("VLLM_K3_W8A16_GEMV_CLUSTER", str(cluster))
    torch.manual_seed(0)
    dev = torch.device("cuda")
    weight = (torch.randn(size_n, size_k, device=dev) / 4).to(torch.float8_e4m3fn)
    scales = torch.randint(
        118, 132, (size_n, size_k // 32), dtype=torch.uint8, device=dev
    )
    exact = (
        weight.double().view(size_n, size_k // 32, 32)
        * scales.view(torch.float8_e8m0fnu).double().unsqueeze(-1)
    ).view(size_n, size_k)
    layer = torch.nn.Module()
    layer.weight = torch.nn.Parameter(weight.clone(), requires_grad=False)
    layer.weight_scale = torch.nn.Parameter(scales.clone(), requires_grad=False)
    layer.output_size_per_partition = size_n
    layer.input_size_per_partition = size_k
    layer.prefix = "model.layers.0.self_attn.o_proj"
    with set_default_torch_dtype(torch.bfloat16):
        prepare_mxfp8_layer_for_marlin(layer)
    x = torch.randn(rows, size_k, device=dev, dtype=torch.bfloat16)
    assert can_w8a16_gemv(layer, x, None)

    out = apply_w8a16_gemv(layer, x)
    again = apply_w8a16_gemv(layer, x)

    expected = x.double() @ exact.T
    torch.testing.assert_close(out.double(), expected, atol=1e-3, rtol=2**-8)
    assert torch.equal(out, again)


def test_gemv_layout_table_overrides_the_default(monkeypatch):
    from vllm.model_executor.kernels.linear.mxfp8.marlin_hybrid import (
        w8a16_gemv_layout,
    )

    monkeypatch.setenv("VLLM_K3_W8A16_GEMV_TABLE", "4224x7168:8x4,7168x1408:2x8")
    monkeypatch.setenv("VLLM_K3_W8A16_GEMV_CLUSTER", "4")
    monkeypatch.setenv("VLLM_K3_W8A16_GEMV_WARPS", "2")

    assert w8a16_gemv_layout(7168, 1408, 0) == (2, 8)
    assert w8a16_gemv_layout(2112, 1536, 0) == (4, 2)


def test_gemv_shared_memory_bound():
    """A one-CTA K split of a 7168-wide input at 8 rows needs more shared
    memory than a CTA may request, so the gate falls back to Marlin; two CTAs
    per column group fit."""
    from vllm.model_executor.kernels.linear.mxfp8.marlin_hybrid import (
        _GEMV_MAX_SMEM,
        w8a16_gemv_smem_bytes,
    )

    assert w8a16_gemv_smem_bytes(8, 7168, 1, 4) > _GEMV_MAX_SMEM
    assert w8a16_gemv_smem_bytes(8, 7168, 2, 4) <= _GEMV_MAX_SMEM
    assert w8a16_gemv_smem_bytes(4, 1408, 4, 4) == 2816 + 5 * 64 * 4 * 4
