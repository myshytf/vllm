# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Batched spec-decode metadata staging for the Kimi-K3 KDA metadata builders.

Production runs one ``KimiK3KDAMetadataBuilder`` per Mamba KV-cache group
(14 for Kimi-K3) and every decode step staged each builder's CUDA-graph
buffers with its own Triton launch. The hybrid model state now stages all
groups in one launch
(``MambaSpecDecodeGPUContext.stage_spec_decode_metadata_all_groups``)
and the builders return the staged views (``mamba_prestaged_spec_decode``).
These tests pin (a) the batch predicate against the builder's own split,
(b) the builder fast path's output, and (c) on CUDA, that the batched kernel
writes exactly what the per-group launches write.
"""

from types import SimpleNamespace

import numpy as np
import pytest
import torch

from tests.v1.attention.utils import (
    BatchSpec,
    create_common_attn_metadata,
    create_vllm_config,
)
from vllm.config import SpeculativeConfig
from vllm.config.compilation import CUDAGraphMode
from vllm.models.kimi_k3.nvidia.kda_metadata import (
    KimiK3KDAMetadata,
    KimiK3KDAMetadataBuilder,
    stage_spec_decode_metadata,
)
from vllm.utils import torch_utils
from vllm.v1.attention.backends import utils as attn_backend_utils
from vllm.v1.attention.backends.utils import NULL_BLOCK_ID
from vllm.v1.kv_cache_interface import MambaSpec
from vllm.v1.worker.gpu.model_states.mamba_hybrid import MambaHybridModelState
from vllm.v1.worker.mamba_utils import MambaSpecDecodeGPUContext

BLOCK_SIZE = 16
CPU = torch.device("cpu")


def _make_builder(
    num_speculative_tokens: int,
    device: torch.device = CPU,
    max_num_seqs: int = 4,
) -> KimiK3KDAMetadataBuilder:
    vllm_config = create_vllm_config(
        model_name="Qwen/Qwen3.5-0.8B",
        block_size=BLOCK_SIZE,
        max_num_seqs=max_num_seqs,
    )
    vllm_config.speculative_config = SpeculativeConfig(
        method="ngram",
        num_speculative_tokens=num_speculative_tokens,
    )
    vllm_config.compilation_config.cudagraph_mode = CUDAGraphMode.FULL_AND_PIECEWISE
    vllm_config.cache_config.mamba_cache_mode = "align"
    return KimiK3KDAMetadataBuilder(
        kv_cache_spec=MambaSpec(
            block_size=BLOCK_SIZE,
            shapes=((16, 64),),
            dtypes=(torch.float16,),
            num_speculative_blocks=num_speculative_tokens,
        ),
        layer_names=["layer.0"],
        vllm_config=vllm_config,
        device=device,
    )


def _qsl(query_lens: list[int]) -> torch.Tensor:
    return torch.tensor(np.concatenate([[0], np.cumsum(query_lens)]), dtype=torch.int32)


@pytest.mark.parametrize(
    ("query_lens", "drafts", "num_reqs", "max_bs", "expected"),
    [
        # pure spec decode, two real rows
        ([3, 3], [2, 2], 2, 8, (2, 6, 2)),
        # pure spec decode with two FULL-graph padding rows (query len 0, -1)
        ([3, 3, 0, 0], [2, 2, -1, -1], 4, 8, (2, 6, 4)),
        # one row without a draft (-1) but with a scheduled token: mixed batch
        ([3, 1], [2, -1], 2, 8, None),
        # spec rows without any scheduled draft token: whole batch non-spec
        ([1, 1], [0, 0], 2, 8, None),
        # prefill chunk among spec rows
        ([50, 3], [-1, 2], 2, 8, None),
        # too many rows or tokens for the capture size
        ([3, 3], [2, 2], 2, 1, None),
        ([3, 3], [2, 2], 2, 5, None),
        # query_start_loc not sized num_reqs + 1
        ([3, 3, 3], [2, 2], 2, 8, None),
    ],
)
def test_pure_spec_decode_predicate(query_lens, drafts, num_reqs, max_bs, expected):
    got = MambaHybridModelState.pure_spec_decode_full_graph_batch(
        _qsl(query_lens),
        torch.tensor(drafts, dtype=torch.int32),
        num_reqs,
        max_bs,
    )
    assert got == expected


def test_predicate_matches_builder_split_on_pure_and_mixed_batches(monkeypatch):
    """The predicate accepts exactly the batches whose builder split is
    num_prefills == num_decodes == 0 with num_spec_decodes > 0."""
    # The mixed-batch paths pin host memory for their H2D copies (a CUDA
    # requirement); on CPU-only hosts turn pinning off so the split logic
    # itself is exercised.
    monkeypatch.setattr(torch_utils, "PIN_MEMORY", False)
    monkeypatch.setattr(attn_backend_utils, "PIN_MEMORY", False)
    builder = _make_builder(num_speculative_tokens=2)
    cases = [
        (BatchSpec(seq_lens=[50, 30], query_lens=[3, 3]), [2, 2], [False, False]),
        (
            BatchSpec(seq_lens=[100, 65, 20], query_lens=[50, 1, 3]),
            [-1, -1, 2],
            [True, False, False],
        ),
        (BatchSpec(seq_lens=[40, 30], query_lens=[1, 1]), [0, 0], [False, False]),
        (BatchSpec(seq_lens=[40, 33], query_lens=[1, 3]), [-1, 2], [False, False]),
    ]
    for batch, drafts, is_prefilling in cases:
        m = create_common_attn_metadata(batch, BLOCK_SIZE, CPU)
        m.is_prefilling = torch.tensor(is_prefilling)
        drafts_t = torch.tensor(drafts, dtype=torch.int32)
        builder.mamba_aligned_state_indices = torch.arange(
            m.num_reqs * 3, dtype=torch.int32
        ).reshape(m.num_reqs, 3)
        builder.mamba_prestaged_spec_decode = None
        builder.use_full_cuda_graph = False  # keep the CPU build off Triton
        md = builder.build(
            0,
            m,
            num_accepted_tokens=torch.ones(m.num_reqs, dtype=torch.int32),
            num_decode_draft_tokens_cpu=drafts_t,
        )
        pure = md.num_prefills == 0 and md.num_decodes == 0 and md.num_spec_decodes > 0
        predicate = MambaHybridModelState.pure_spec_decode_full_graph_batch(
            m.query_start_loc_cpu, drafts_t, m.num_reqs, 64
        )
        assert (predicate is not None) == pure, (batch, drafts)
        if predicate is not None:
            assert predicate == (
                md.num_spec_decodes,
                md.num_spec_decode_tokens,
                m.num_reqs,
            )


def test_prestaged_build_returns_staged_views_without_recomputing():
    builder = _make_builder(num_speculative_tokens=2)
    assert builder.use_full_cuda_graph
    batch = BatchSpec(seq_lens=[50, 30, 0, 0], query_lens=[3, 3, 0, 0])
    m = create_common_attn_metadata(batch, BLOCK_SIZE, CPU)
    m.is_prefilling = torch.tensor([False] * 4)
    # Poison the sources the general path would read: the fast path must not
    # touch them.
    builder.mamba_aligned_state_indices = None
    builder.mamba_prestaged_spec_decode = (2, 6, 4)
    builder.spec_state_indices_tensor.fill_(7)
    builder.spec_query_start_loc.fill_(9)
    builder.num_accepted_tokens.fill_(5)
    md = builder.build(
        0,
        m,
        num_accepted_tokens=torch.ones(4, dtype=torch.int32),
        num_decode_draft_tokens_cpu=torch.tensor([2, 2, -1, -1], dtype=torch.int32),
    )
    assert isinstance(md, KimiK3KDAMetadata)
    assert (md.num_prefills, md.num_decodes) == (0, 0)
    assert (md.num_spec_decodes, md.num_spec_decode_tokens) == (2, 6)
    assert md.num_actual_tokens == m.num_actual_tokens
    assert (
        md.spec_state_indices_tensor.data_ptr()
        == builder.spec_state_indices_tensor.data_ptr()
    )
    assert md.spec_state_indices_tensor.shape == (4, 3)
    assert md.spec_query_start_loc.data_ptr() == builder.spec_query_start_loc.data_ptr()
    assert md.spec_query_start_loc.shape == (5,)
    assert md.num_accepted_tokens.data_ptr() == builder.num_accepted_tokens.data_ptr()
    assert md.num_accepted_tokens.shape == (4,)
    assert torch.equal(
        md.spec_state_indices_tensor, torch.full((4, 3), 7, dtype=torch.int32)
    )
    for name in (
        "non_spec_query_start_loc",
        "non_spec_state_indices_tensor",
        "spec_sequence_masks",
        "spec_token_indx",
        "non_spec_token_indx",
        "has_initial_state",
        "nums_dict",
        "batch_ptr",
        "token_chunk_offset_ptr",
        "checkpoint",
    ):
        assert getattr(md, name) is None, name


def test_prestaged_build_ignored_when_batch_does_not_match():
    """A stale prestaged marker (different padded batch) or a missing draft
    vector falls back to the general path."""
    builder = _make_builder(num_speculative_tokens=2)
    builder.use_full_cuda_graph = False  # general path without Triton on CPU
    batch = BatchSpec(seq_lens=[50, 30], query_lens=[3, 3])
    m = create_common_attn_metadata(batch, BLOCK_SIZE, CPU)
    m.is_prefilling = torch.tensor([False, False])
    builder.mamba_aligned_state_indices = torch.arange(6, dtype=torch.int32).reshape(
        2, 3
    )
    builder.mamba_prestaged_spec_decode = (2, 6, 4)  # padded to 4, batch has 2
    md = builder.build(
        0,
        m,
        num_accepted_tokens=torch.ones(2, dtype=torch.int32),
        num_decode_draft_tokens_cpu=torch.tensor([2, 2], dtype=torch.int32),
    )
    # general path: the state indices come from the aligned view, not the buffers
    assert (
        md.spec_state_indices_tensor.data_ptr()
        != builder.spec_state_indices_tensor.data_ptr()
    )
    torch.testing.assert_close(
        md.spec_state_indices_tensor, builder.mamba_aligned_state_indices[:2, :3]
    )


def _fake_builders(num_groups, max_bs, num_state_slots, device):
    return [
        (
            g,
            SimpleNamespace(
                spec_state_indices_tensor=torch.full(
                    (max_bs, num_state_slots), -3, dtype=torch.int32, device=device
                ),
                spec_query_start_loc=torch.full(
                    (max_bs + 1,), -3, dtype=torch.int32, device=device
                ),
                num_accepted_tokens=torch.full(
                    (max_bs,), -3, dtype=torch.int32, device=device
                ),
            ),
        )
        for g in range(num_groups)
    ]


def _fake_ctx(num_groups, max_num_reqs, aligned_slots, device):
    ctx = MambaSpecDecodeGPUContext.__new__(MambaSpecDecodeGPUContext)
    ctx.num_groups = num_groups
    ctx.aligned_state_indices = torch.randint(
        1,
        1000,
        (num_groups, max_num_reqs, aligned_slots),
        dtype=torch.int32,
        device=device,
    )
    ctx.staged_spec_state_ptrs = None
    ctx.staged_spec_query_start_loc_ptrs = None
    ctx.staged_spec_num_accepted_ptrs = None
    ctx.staged_spec_state_strides = (0, 0)
    ctx.staged_spec_num_state_slots = 0
    ctx.staged_spec_max_batch = 0
    ctx.staged_spec_registered_ids = ()
    return ctx


def test_register_rejects_incomplete_or_mismatched_builder_sets():
    device = CPU
    ctx = _fake_ctx(3, 8, 3, device)
    builders = _fake_builders(3, 8, 3, device)
    # CPU tensors never qualify (the kernel is a CUDA launch).
    assert not ctx.register_spec_decode_staging(builders)
    assert ctx.staged_spec_state_ptrs is None
    # A missing group never qualifies either.
    assert not ctx.register_spec_decode_staging(builders[:2])


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize(
    ("num_groups", "num_spec_decodes", "batch_size", "num_state_slots"),
    [(14, 4, 4, 6), (14, 3, 8, 6), (3, 1, 1, 3), (5, 24, 32, 4), (14, 33, 65, 6)],
)
def test_batched_staging_matches_per_group_launches(
    num_groups, num_spec_decodes, batch_size, num_state_slots
):
    device = torch.device("cuda")
    max_bs = 72
    aligned_slots = num_state_slots + 1  # the aligned buffer may carry more slots
    ctx = _fake_ctx(num_groups, max_bs, aligned_slots, device)
    builders = _fake_builders(num_groups, max_bs, num_state_slots, device)
    assert ctx.register_spec_decode_staging(builders)
    assert ctx.register_spec_decode_staging(builders)  # idempotent

    query_start_loc = torch.cumsum(
        torch.randint(1, 7, (max_bs + 1,), dtype=torch.int32, device=device), 0
    ).to(torch.int32)
    query_start_loc[0] = 0
    num_accepted_tokens = torch.randint(
        1, 7, (max_bs,), dtype=torch.int32, device=device
    )

    ctx.stage_spec_decode_metadata_all_groups(
        query_start_loc,
        num_accepted_tokens,
        num_spec_decodes=num_spec_decodes,
        batch_size=batch_size,
    )
    torch.accelerator.synchronize()

    for g, b in builders:
        expected_state = torch.full(
            (batch_size, num_state_slots), -3, dtype=torch.int32, device=device
        )
        expected_qsl = torch.full(
            (batch_size + 1,), -3, dtype=torch.int32, device=device
        )
        expected_acc = torch.full((batch_size,), -3, dtype=torch.int32, device=device)
        stage_spec_decode_metadata(
            ctx.aligned_state_indices[g, :num_spec_decodes, :num_state_slots],
            query_start_loc[: num_spec_decodes + 1],
            num_accepted_tokens[:num_spec_decodes],
            expected_state,
            expected_qsl,
            expected_acc,
            num_spec_decodes=num_spec_decodes,
        )
        torch.accelerator.synchronize()
        torch.testing.assert_close(
            b.spec_state_indices_tensor[:batch_size], expected_state, rtol=0, atol=0
        )
        torch.testing.assert_close(
            b.spec_query_start_loc[: batch_size + 1], expected_qsl, rtol=0, atol=0
        )
        torch.testing.assert_close(
            b.num_accepted_tokens[:batch_size], expected_acc, rtol=0, atol=0
        )
        # rows past batch_size untouched
        assert torch.all(b.spec_state_indices_tensor[batch_size:] == -3)
        assert torch.all(b.spec_query_start_loc[batch_size + 1 :] == -3)
        assert torch.all(b.num_accepted_tokens[batch_size:] == -3)
        # padding rows carry the same sentinels as the per-group kernel
        if batch_size > num_spec_decodes:
            assert torch.all(
                b.spec_state_indices_tensor[num_spec_decodes:batch_size]
                == NULL_BLOCK_ID
            )
            assert torch.all(b.num_accepted_tokens[num_spec_decodes:batch_size] == 1)
            assert torch.all(
                b.spec_query_start_loc[num_spec_decodes + 1 : batch_size + 1]
                == query_start_loc[num_spec_decodes]
            )
