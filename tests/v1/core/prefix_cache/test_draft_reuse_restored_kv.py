# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Prefix-cache geometry when the DFlash draft reuses restored draft KV.

Without the EAGLE last-block drop the target cache publishes its recurrent
checkpoint at the prompt's last hash boundary, so a later prompt that extends
the same prefix (the usual agentic turn) recomputes one hash unit less than
with the drop. Legacy geometry with the drop is unchanged.
"""

from types import SimpleNamespace

import pytest
import torch

from tests.v1.core.test_prefix_caching import make_kv_cache_manager, make_request
from vllm import envs
from vllm.utils.hashing import sha256
from vllm.v1.core.kv_cache_utils import _annotate_eagle_groups, init_none_hash
from vllm.v1.core.sched.scheduler import Scheduler, use_eagle_for_target_cache
from vllm.v1.kv_cache_interface import (
    FullAttentionSpec,
    KVCacheConfig,
    KVCacheGroupSpec,
    MambaSpec,
    SlidingWindowSpec,
    get_mamba_prefill_checkpoint_position,
)
from vllm.v1.worker.gpu.states import draft_reuses_restored_kv


@pytest.fixture(autouse=True)
def _none_hash():
    init_none_hash(sha256)


@pytest.mark.parametrize(
    ("num_tokens", "drop", "expected"),
    [
        (23, False, 22),
        (23, True, 20),
        (24, False, 24),
        (24, True, 22),
        (2, False, 2),
        (1, False, 0),
        (0, True, 0),
    ],
)
def test_checkpoint_position_is_the_last_hash_boundary(num_tokens, drop, expected):
    assert get_mamba_prefill_checkpoint_position(num_tokens, 2, drop) == expected


def _dcp9_config(unit: int, draft_is_eagle: bool) -> KVCacheConfig:
    return KVCacheConfig(
        num_blocks=64,
        kv_cache_tensors=[],
        kv_cache_groups=[
            KVCacheGroupSpec(
                ["target_attention"],
                FullAttentionSpec(
                    block_size=unit,
                    num_kv_heads=1,
                    head_size=1,
                    dtype=torch.float32,
                ),
            ),
            KVCacheGroupSpec(
                ["target_recurrent"],
                MambaSpec(
                    block_size=6,
                    shapes=(1, 1),
                    dtypes=(torch.float32,),
                    mamba_cache_mode="align",
                    num_prefill_checkpoint_blocks=1,
                ),
            ),
            KVCacheGroupSpec(
                ["draft_attention"],
                SlidingWindowSpec(
                    block_size=unit,
                    num_kv_heads=1,
                    head_size=1,
                    dtype=torch.float32,
                    sliding_window=6,
                    dcp_replicated=True,
                ),
                is_eagle_group=draft_is_eagle,
            ),
        ],
    )


def _prefill(manager, tokens, unit, eagle):
    producer = make_request("producer", tokens, unit, sha256)
    scheduler = SimpleNamespace(
        cache_config=SimpleNamespace(block_size=unit),
        mamba_block_size=6,
        max_num_scheduled_tokens=6,
        scheduler_config=SimpleNamespace(long_prefill_token_threshold=0),
        use_eagle=True,
        use_eagle_for_target_cache=eagle,
        mamba_eagle_block_sizes=(unit,) if eagle else (),
        hash_block_size=unit,
        mamba_partial_cache_hit=True,
    )
    chunks = []
    while producer.num_computed_tokens < len(tokens):
        remaining = len(tokens) - producer.num_computed_tokens
        chunk = Scheduler._mamba_block_aligned_split(
            scheduler, producer, min(6, remaining)
        )
        assert chunk > 0
        chunks.append(chunk)
        assert manager.allocate_slots(producer, chunk) is not None
        producer.num_computed_tokens += chunk
        manager.new_step_starts()
    manager.free(producer)
    manager.new_step_starts()
    return chunks


@pytest.mark.parametrize(
    ("producer_tokens", "consumer_tokens", "eagle", "expected_chunks", "expected_hit"),
    [
        # Draft reuse: the checkpoint sits on the last hash boundary and a
        # same-length replay reaches it.
        (23, 23, False, [6, 6, 6, 4, 1], 22),
        # A prompt ending on a hash boundary publishes that boundary; the
        # next, longer turn reuses it.
        (24, 25, False, [6, 6, 6, 6], 24),
        # Legacy EAGLE drop keeps its geometry.
        (23, 23, True, [6, 6, 6, 2, 3], 20),
        (24, 25, True, [6, 6, 6, 4, 2], 22),
    ],
)
def test_dcp9_next_turn_reaches_the_published_checkpoint(
    producer_tokens, consumer_tokens, eagle, expected_chunks, expected_hit
):
    unit = 2
    manager = make_kv_cache_manager(
        kv_cache_config=_dcp9_config(unit, draft_is_eagle=eagle),
        max_model_len=128,
        enable_caching=True,
        scheduler_block_size=18,
        hash_block_size=unit,
        dcp_world_size=9,
        use_eagle=eagle,
    )
    tokens = list(range(producer_tokens))
    assert _prefill(manager, tokens, unit, eagle) == expected_chunks

    consumer = make_request("consumer", list(range(consumer_tokens)), unit, sha256)
    blocks, hit, _ = manager.get_computed_blocks(consumer)
    assert hit == expected_hit
    assert (
        manager.allocate_slots(consumer, consumer_tokens - hit, hit, blocks) is not None
    )
    copies, _ = manager.take_kv_cache_block_copies()
    assert {copy.kv_cache_group_id for copy in copies} <= {0, 1}


def _spec_config(method: str):
    return SimpleNamespace(
        method=method,
        use_eagle=lambda: True,
        use_dflash=lambda: method == "dflash",
    )


def _groups_with_draft_layer():
    draft_spec = SimpleNamespace(non_causal_multi_token_decode=True)
    target_spec = SimpleNamespace(non_causal_multi_token_decode=False)
    groups = [
        KVCacheGroupSpec(["target"], target_spec),
        KVCacheGroupSpec(["draft"], draft_spec),
    ]
    return {"target": target_spec, "draft": draft_spec}, groups


def test_annotation_skips_the_dflash_draft_group_when_reusing(monkeypatch):
    monkeypatch.setattr(envs, "VLLM_K3_DRAFT_REUSE_RESTORED_KV", True)
    spec, groups = _groups_with_draft_layer()
    config = SimpleNamespace(speculative_config=_spec_config("dflash"))
    _annotate_eagle_groups(config, spec, groups)
    assert [g.is_eagle_group for g in groups] == [False, False]
    assert not use_eagle_for_target_cache(config.speculative_config, groups)


def test_annotation_keeps_the_eagle_drop_without_reuse(monkeypatch):
    monkeypatch.setattr(envs, "VLLM_K3_DRAFT_REUSE_RESTORED_KV", False)
    spec, groups = _groups_with_draft_layer()
    config = SimpleNamespace(speculative_config=_spec_config("dflash"))
    _annotate_eagle_groups(config, spec, groups)
    assert [g.is_eagle_group for g in groups] == [False, True]
    assert use_eagle_for_target_cache(config.speculative_config, groups)


def test_annotation_ignores_the_reuse_flag_for_other_drafters(monkeypatch):
    monkeypatch.setattr(envs, "VLLM_K3_DRAFT_REUSE_RESTORED_KV", True)
    spec, groups = _groups_with_draft_layer()
    config = SimpleNamespace(speculative_config=_spec_config("dspark"))
    _annotate_eagle_groups(config, spec, groups)
    assert [g.is_eagle_group for g in groups] == [False, True]


def test_runtime_disable_file_hides_restored_tokens_again(monkeypatch, tmp_path):
    monkeypatch.setattr(envs, "VLLM_K3_DRAFT_REUSE_RESTORED_KV", True)
    disable_file = tmp_path / "draft-reuse-disable"
    monkeypatch.setattr(
        envs, "VLLM_K3_DRAFT_REUSE_RESTORED_KV_DISABLE_FILE", str(disable_file)
    )
    assert draft_reuses_restored_kv()
    disable_file.write_text("")
    assert not draft_reuses_restored_kv()
    disable_file.unlink()
    assert draft_reuses_restored_kv()

    monkeypatch.setattr(envs, "VLLM_K3_DRAFT_REUSE_RESTORED_KV", False)
    assert not draft_reuses_restored_kv()
