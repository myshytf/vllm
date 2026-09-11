# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Exactness of the Marlin MXFP8 payload inversion used by the hybrid kernel."""

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


@pytest.mark.parametrize(
    "size_n,size_k", [(256, 512), (1365, 7168), (7168, 683 * 32 // 32 * 32)]
)
def test_unrepack_matches_reference(size_n, size_k):
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
    prepare_mxfp8_layer_for_marlin(layer)

    w_torch = reconstruct_bf16_weight_torch(
        layer.weight, layer.weight_scale, size_n, size_k, torch.bfloat16
    )
    w_triton = reconstruct_bf16_weight_triton(
        layer.weight, layer.weight_scale, size_n, size_k, torch.bfloat16
    )
    assert torch.equal(w_torch, ref)
    assert torch.equal(w_triton, ref)
