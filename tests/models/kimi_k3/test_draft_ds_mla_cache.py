# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Page geometry of the windowed Kimi-K3 draft KV group under fp8_ds_mla.

The served K3 target stores its MLA latent as 656-byte fp8_ds_mla records.
The colocated DFlash2 draft is a windowed (SlidingWindowMLASpec) group in the
same hybrid pool. These tests pin that a draft group declared fp8_ds_mla gets
the same 656-byte record, hence exactly the target's page, so the unifier
pads nothing; and that the plain-fp8 draft (576 bytes per token) is padded to
the target's page, as it is served today.
"""

import pytest

from vllm.models.kimi_k3.nvidia import mla
from vllm.utils.torch_utils import kv_cache_dtype_str_to_dtype
from vllm.v1.core.kv_cache_utils import unify_kv_cache_spec_page_size
from vllm.v1.kv_cache_interface import (
    MLAAttentionSpec,
    SlidingWindowMLASpec,
    get_kv_quant_mode,
)

_BLOCK = 1536
_WINDOW = 4608
_K3_HEAD = 576  # kv_lora_rank 512 + rope 64
_DS_MLA_RECORD = 656


def _target_spec(cache_dtype: str = "fp8_ds_mla") -> MLAAttentionSpec:
    return MLAAttentionSpec(  # type: ignore[call-arg]
        block_size=_BLOCK,
        num_kv_heads=1,
        head_size=_K3_HEAD,
        dtype=kv_cache_dtype_str_to_dtype(cache_dtype, None),
        cache_dtype_str=cache_dtype,
        kv_quant_mode=get_kv_quant_mode(cache_dtype),
        indexes_kv_by_block_stride=True,
    )


def _draft_spec(cache_dtype: str) -> SlidingWindowMLASpec:
    return SlidingWindowMLASpec(
        block_size=_BLOCK,
        num_kv_heads=1,
        head_size=_K3_HEAD,
        dtype=kv_cache_dtype_str_to_dtype(cache_dtype, None),
        cache_dtype_str=cache_dtype,
        kv_quant_mode=get_kv_quant_mode(cache_dtype),
        sliding_window=_WINDOW,
        dcp_replicated=True,
        non_causal_multi_token_decode=True,
        indexes_kv_by_block_stride=True,
    )


def test_windowed_ds_mla_draft_uses_the_656_byte_record() -> None:
    spec = _draft_spec("fp8_ds_mla")

    assert spec.page_size_bytes == _BLOCK * _DS_MLA_RECORD
    assert spec.page_size_bytes == _target_spec().page_size_bytes


def test_windowed_plain_fp8_draft_keeps_576_bytes_per_token() -> None:
    assert _draft_spec("fp8").page_size_bytes == _BLOCK * _K3_HEAD


def test_ds_mla_draft_and_target_unify_without_padding() -> None:
    unified = unify_kv_cache_spec_page_size(
        {"target": _target_spec(), "draft": _draft_spec("fp8_ds_mla")}
    )

    assert unified["draft"].page_size_padded is None
    assert unified["target"].page_size_padded is None
    assert unified["draft"].block_size == _BLOCK
    assert unified["draft"].page_size_bytes == unified["target"].page_size_bytes


def test_plain_fp8_draft_is_padded_to_the_target_page() -> None:
    unified = unify_kv_cache_spec_page_size(
        {"target": _target_spec(), "draft": _draft_spec("fp8")}
    )

    assert unified["target"].page_size_padded is None
    assert unified["draft"].page_size_padded == _BLOCK * _DS_MLA_RECORD
    assert unified["draft"].block_size == _BLOCK


@pytest.mark.parametrize("cache_dtype", ["fp8", "fp8_ds_mla"])
def test_bounded_draft_layer_declares_its_cache_dtype(
    monkeypatch: pytest.MonkeyPatch, cache_dtype: str
) -> None:
    monkeypatch.delenv("VLLM_DCP_SHARD_DRAFT", raising=False)
    attention = object.__new__(mla.MultiHeadLatentAttention)
    attention.kv_cache_dtype = cache_dtype
    attention.head_size = _K3_HEAD
    attention.non_causal_multi_token_decode = True
    attention.draft_kv_window = _WINDOW

    class _Config:
        class cache_config:
            block_size = _BLOCK

        model_config = None

        class parallel_config:
            decode_context_parallel_size = 9

    spec = mla.MultiHeadLatentAttention.get_kv_cache_spec(attention, _Config())

    assert isinstance(spec, SlidingWindowMLASpec)
    assert spec.cache_dtype_str == cache_dtype
    assert spec.dtype == kv_cache_dtype_str_to_dtype(cache_dtype, None)
    expected = _DS_MLA_RECORD if cache_dtype == "fp8_ds_mla" else _K3_HEAD
    assert spec.page_size_bytes == _BLOCK * expected
    assert spec.dcp_replicated
