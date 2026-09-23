# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Draft-only packed-reader split counts and partial types for B12X_MLA.

A draft group (non-causal query block per request, e.g. the Kimi-K3 DFlash2
draft) takes its packed split count per plan row capacity from
VLLM_K3_DRAFT_PACKED_MLA_SPLITS and its partial type from
VLLM_K3_DRAFT_PACKED_MLA_PARTIAL_DTYPE; every other group keeps the
capacity-based plan so the target's launches stay unchanged.
"""

from types import SimpleNamespace

import pytest
import torch

from vllm.v1.attention.backends.mla import b12x_mla

_DRAFT = SimpleNamespace(non_causal_multi_token_decode=True)
_TARGET = SimpleNamespace(non_causal_multi_token_decode=False)
_WINDOW_TOKENS = 6143  # 96 chunks of 64 tokens -> 64 splits by default


@pytest.fixture
def splits_env(monkeypatch: pytest.MonkeyPatch):
    def set_env(splits: str = "", partial: str = "") -> None:
        monkeypatch.setenv("VLLM_K3_DRAFT_PACKED_MLA_SPLITS", splits)
        monkeypatch.setenv("VLLM_K3_DRAFT_PACKED_MLA_PARTIAL_DTYPE", partial)

    return set_env


def test_split_table_parses_sorted_pairs(splits_env) -> None:
    splits_env("32:6, 8:24,16:12")

    assert b12x_mla._draft_packed_split_table() == ((8, 24), (16, 12), (32, 6))


@pytest.mark.parametrize("raw", ["8", "8:0", "x:4", "8:2,8:3", "0:4", "8:-1"])
def test_split_table_rejects_malformed_pairs(splits_env, raw: str) -> None:
    splits_env(raw)

    with pytest.raises(ValueError, match="rows:splits"):
        b12x_mla._draft_packed_split_table()


@pytest.mark.parametrize(
    ("row_cap", "splits"),
    [(1, 24), (8, 24), (9, 12), (16, 12), (28, 6), (32, 6), (60, 6)],
)
def test_row_cap_takes_the_smallest_covering_entry(row_cap: int, splits: int) -> None:
    table = ((8, 24), (16, 12), (32, 6))

    assert b12x_mla._draft_packed_splits(row_cap, table) == splits


def test_target_groups_get_no_overrides(splits_env) -> None:
    splits_env("8:24,32:6", "fp32")

    assert b12x_mla._packed_plan_overrides(_TARGET, 8) == {}
    assert b12x_mla._packed_plan_overrides(SimpleNamespace(), 8) == {}


def test_draft_groups_take_table_and_partial_type(splits_env) -> None:
    splits_env("8:24,32:6", "fp32")

    assert b12x_mla._packed_plan_overrides(_DRAFT, 16) == {
        "max_chunks_per_row": 6,
        "partial_dtype_name": "fp32",
    }


def test_draft_partial_type_is_validated(splits_env) -> None:
    splits_env("", "fp16")

    with pytest.raises(ValueError, match="VLLM_K3_DRAFT_PACKED_MLA_PARTIAL_DTYPE"):
        b12x_mla._packed_plan_overrides(_DRAFT, 8)


def test_unset_draft_environment_keeps_the_capacity_plan(splits_env) -> None:
    splits_env()

    assert b12x_mla._packed_plan_overrides(_DRAFT, 8) == {}


@pytest.mark.parametrize(
    ("spec", "policy", "expected"),
    [
        (_DRAFT, "balanced", "balanced"),
        (_DRAFT, "static", "static"),
        (_DRAFT, "", None),
        (_TARGET, "balanced", None),
        (SimpleNamespace(), "balanced", None),
    ],
)
def test_split_policy_override_applies_to_draft_groups_only(
    monkeypatch: pytest.MonkeyPatch, spec, policy: str, expected
) -> None:
    monkeypatch.setenv("VLLM_K3_DRAFT_PACKED_MLA_SPLIT_POLICY", policy)

    assert b12x_mla._packed_split_policy_override(spec) == expected


def test_split_policy_override_is_validated(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("VLLM_K3_DRAFT_PACKED_MLA_SPLIT_POLICY", "dynamic")

    with pytest.raises(ValueError, match="VLLM_K3_DRAFT_PACKED_MLA_SPLIT_POLICY"):
        b12x_mla._packed_split_policy_override(_DRAFT)


class _Caps:
    __dataclass_fields__ = {"partial_dtype": None}

    def __init__(self, **kwargs) -> None:
        self.kwargs = kwargs


def _plan(monkeypatch: pytest.MonkeyPatch, **overrides):
    monkeypatch.setenv("VLLM_K3_PACKED_MLA_PARTIAL_DTYPE", "bf16")
    monkeypatch.setattr(
        b12x_mla,
        "_load_sparse_mla",
        lambda: SimpleNamespace(Caps=_Caps, plan=lambda caps: caps),
    )
    config = SimpleNamespace(
        parallel_config=SimpleNamespace(decode_context_parallel_size=1),
        scheduler_config=SimpleNamespace(max_num_seqs=4),
    )
    caps = b12x_mla._create_packed_dense_mla_plan(
        config,
        torch.device("cpu"),
        page_size=1536,
        num_q_heads=8,
        max_total_q=8,
        dcp_size=1,
        max_cache_tokens=_WINDOW_TOKENS,
        **overrides,
    )
    return caps.kwargs


def test_packed_plan_defaults_to_the_capacity_split_count(monkeypatch) -> None:
    kwargs = _plan(monkeypatch)

    assert kwargs["max_chunks_per_row"] == 64
    assert kwargs["partial_dtype"] == torch.bfloat16


@pytest.mark.parametrize(("requested", "planned"), [(24, 24), (1, 1), (200, 64)])
def test_packed_plan_takes_the_requested_split_count(
    monkeypatch, requested: int, planned: int
) -> None:
    kwargs = _plan(monkeypatch, max_chunks_per_row=requested)

    assert kwargs["max_chunks_per_row"] == planned


def test_packed_plan_takes_the_partial_type_override(monkeypatch) -> None:
    kwargs = _plan(monkeypatch, partial_dtype_name="fp32")

    assert kwargs["partial_dtype"] == torch.float32
