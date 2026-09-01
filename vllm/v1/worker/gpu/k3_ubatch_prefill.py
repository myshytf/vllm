# SPDX-License-Identifier: Apache-2.0
"""Split one prefill chunk into two row halves and run them as consecutive
sub-steps (Kimi-K3, eager chunked prefill).

Stage 1 of the TP all-reduce / compute overlap design
(research/prefill-w4a16-20260902/DESIGN-intra-request-ubatch-prefill-20260902.md):
the halves run sequentially on the current stream, so every cross-half
dependency (K/V written by the first half, the recurrent KDA state carried
in the request's mamba slot) is satisfied by stream order and the outputs
are the same values the unsplit forward produces. This stage only proves
the split; the overlap (two streams, yield points at the collectives)
comes on top of it.

Eligibility: a single request, no draft tokens, eager (non-graph) dispatch,
at least ``VLLM_K3_UBATCH_PREFILL_MIN_TOKENS`` scheduled tokens, enabled
with ``VLLM_K3_UBATCH_PREFILL=1``. Anything else runs the normal path.
"""
from __future__ import annotations

import dataclasses
import os
import threading

import numpy as np
import torch

from vllm.forward_context import (
    create_forward_context,
    override_forward_context,
    set_forward_context,
)
from vllm.v1.worker.gpu.attn_utils import build_slot_mappings_by_layer
from vllm.v1.worker.gpu.cp_utils import prepare_dcp_local_seq_lens
from vllm.v1.worker.gpu.input_batch import InputBatch


def ubatch_prefill_enabled() -> bool:
    return os.getenv("VLLM_K3_UBATCH_PREFILL", "0") == "1"


def ubatch_prefill_min_tokens() -> int:
    return int(os.getenv("VLLM_K3_UBATCH_PREFILL_MIN_TOKENS", "1024"))


def ubatch_prefill_overlap() -> bool:
    """Stage 2: run the halves on two threads with the TP all-reduces on a
    shared comm stream (see communication_op._ubatch_comm_region)."""
    return os.getenv("VLLM_K3_UBATCH_PREFILL_OVERLAP", "0") == "1"


_COMM_STREAM: torch.cuda.Stream | None = None
_READY_BARRIER: threading.Barrier | None = None


def _comm_resources(device: torch.device):
    global _COMM_STREAM, _READY_BARRIER
    if _COMM_STREAM is None:
        _COMM_STREAM = torch.cuda.Stream(device=device)
        _READY_BARRIER = threading.Barrier(3)
    return _COMM_STREAM, _READY_BARRIER


def eligible(input_batch: InputBatch) -> bool:
    return (
        input_batch.num_reqs == 1
        and input_batch.num_draft_tokens == 0
        and input_batch.num_tokens == input_batch.num_tokens_after_padding
        and input_batch.num_tokens >= ubatch_prefill_min_tokens()
        and bool(input_batch.is_prefilling_np[0])
    )


def _half_batch(
    runner,
    input_batch: InputBatch,
    row_start: int,
    row_end: int,
) -> InputBatch:
    """The rows [row_start, row_end) of a single-request batch as a batch of
    their own, positioned as if the earlier rows were already computed."""
    rows = row_end - row_start
    device = input_batch.query_start_loc.device
    computed = input_batch.num_computed_tokens_np.copy()
    computed[0] += row_start
    computed_prefill = input_batch.num_computed_prefill_tokens_np.copy()
    computed_prefill[0] += row_start
    seq_len = int(computed[0] + rows)
    seq_lens = torch.full((input_batch.num_reqs_after_padding,), seq_len,
                          dtype=torch.int32, device=device)
    seq_lens_cpu = torch.full((input_batch.num_reqs_after_padding,), seq_len,
                              dtype=torch.int32)
    qsl_np = np.array([0, rows], dtype=np.int32)
    qsl = torch.from_numpy(qsl_np).to(device, non_blocking=True)
    dcp_local = None
    if input_batch.dcp_local_seq_lens is not None:
        dcp_local = torch.empty_like(input_batch.dcp_local_seq_lens)
        prepare_dcp_local_seq_lens(
            dcp_local,
            seq_lens,
            input_batch.num_reqs,
            runner.dcp_size,
            runner.dcp_rank,
            runner.cp_interleave,
        )
    return dataclasses.replace(
        input_batch,
        num_scheduled_tokens=np.array([rows], dtype=np.int32),
        max_query_len=rows,
        num_tokens=rows,
        num_tokens_after_padding=rows,
        query_start_loc=qsl,
        query_start_loc_np=qsl_np,
        seq_lens=seq_lens,
        seq_lens_cpu_upper_bound=seq_lens_cpu,
        max_seq_len_upper_bound=seq_len,
        dcp_local_seq_lens=dcp_local,
        num_computed_tokens_np=computed,
        num_computed_prefill_tokens_np=computed_prefill,
        input_ids=input_batch.input_ids[row_start:row_end],
        positions=input_batch.positions[row_start:row_end],
        is_padding=input_batch.is_padding[row_start:row_end],
        max_req_tokens=None,
    )


def run_split_prefill(
    runner,
    scheduler_output,
    input_batch: InputBatch,
    model_inputs: dict,
    *,
    cudagraph_runtime_mode,
    num_tokens_across_dp,
    batch_descriptor,
    skip_compiled: bool,
):
    """Run the chunk as two consecutive half forwards; returns the
    concatenated model output in the same form as one forward."""
    rows = input_batch.num_tokens
    # First half is the larger one so the second half fits the upper half
    # of the retained AttnRes workspace (see KimiLinearModel).
    split = rows - rows // 2
    halves = ((0, split), (split, rows))
    outputs = []
    prepared = []
    num_computed_gpu = runner.req_states.num_computed_tokens.gpu
    req_state_idx = int(input_batch.idx_mapping_np[0])
    for start, end in halves:
        hb = _half_batch(runner, input_batch, start, end)
        block_tables, slot_mappings = runner.prepare_attn(hb)
        # The recurrent-state pre-copy reads the request's computed-token
        # count from the GPU request table; the second half starts where the
        # first ended, so hand it a shifted copy instead of the step-start
        # value.
        if start == 0:
            computed_for_half = num_computed_gpu
        else:
            computed_for_half = num_computed_gpu.clone()
            computed_for_half[req_state_idx] += start
        runner.model_state.preprocess_state(
            hb, block_tables, runner.kv_cache_config, computed_for_half,
        )
        slot_mappings_by_layer = build_slot_mappings_by_layer(
            slot_mappings, runner.kv_cache_config
        )
        attn_metadata = runner.model_state.prepare_attn(
            hb, cudagraph_runtime_mode, block_tables, slot_mappings,
            runner.attn_groups, runner.kv_cache_config, for_capture=False,
        )
        half_inputs = dict(model_inputs)
        half_inputs["input_ids"] = (
            None if model_inputs.get("input_ids") is None
            else model_inputs["input_ids"][start:end]
        )
        half_inputs["positions"] = model_inputs["positions"][start:end]
        if model_inputs.get("inputs_embeds") is not None:
            half_inputs["inputs_embeds"] = model_inputs["inputs_embeds"][start:end]
        half_inputs.update(runner.model_state.prepare_inputs(hb, runner.req_states))
        prepared.append((hb, attn_metadata, slot_mappings_by_layer, half_inputs))

    if not ubatch_prefill_overlap():
        for hb, attn_metadata, slot_mappings_by_layer, half_inputs in prepared:
            with set_forward_context(
                attn_metadata,
                runner.vllm_config,
                num_tokens=hb.num_tokens_after_padding,
                cudagraph_runtime_mode=cudagraph_runtime_mode,
                num_tokens_across_dp=num_tokens_across_dp,
                batch_descriptor=dataclasses.replace(
                    batch_descriptor, num_tokens=hb.num_tokens_after_padding
                ),
                slot_mapping=slot_mappings_by_layer,
                skip_compiled=skip_compiled,
                is_padding=hb.is_padding,
            ):
                outputs.append(runner.model(**half_inputs))
    else:
        outputs = _run_overlapped(runner, prepared, cudagraph_runtime_mode,
                                  batch_descriptor, skip_compiled)
    first = outputs[0]
    if isinstance(first, tuple):
        hidden = torch.cat([o[0] for o in outputs], dim=0)
        aux_lists = [o[1] for o in outputs]
        aux = [torch.cat(parts, dim=0) for parts in zip(*aux_lists)]
        return hidden, aux
    return torch.cat(outputs, dim=0)


def _run_overlapped(runner, prepared, cudagraph_runtime_mode, batch_descriptor,
                    skip_compiled):
    """Two threads, one per half, alternating at every TP all-reduce.

    Both halves compute on the current (compute) stream in CPU-issue order,
    which keeps every cross-half dependency (K/V and recurrent state written
    by the first half) satisfied by stream order; the all-reduces go to one
    shared comm stream (communication_op._ubatch_comm_region), so each
    half's collective overlaps the other half's compute.
    """
    from vllm.v1.worker.ubatching import make_ubatch_contexts

    device = prepared[0][0].query_start_loc.device
    compute_stream = torch.cuda.current_stream(device)
    comm_stream, ready_barrier = _comm_resources(device)
    forward_contexts = [
        create_forward_context(
            attn_metadata,
            runner.vllm_config,
            cudagraph_runtime_mode=cudagraph_runtime_mode,
            batch_descriptor=dataclasses.replace(
                batch_descriptor, num_tokens=hb.num_tokens_after_padding
            ),
            slot_mapping=slot_mappings_by_layer,
            skip_compiled=skip_compiled,
            is_padding=hb.is_padding,
        )
        for hb, attn_metadata, slot_mappings_by_layer, _ in prepared
    ]
    ctxs = make_ubatch_contexts(
        num_micro_batches=len(prepared),
        compute_stream=compute_stream,
        comm_stream=comm_stream,
        forward_contexts=forward_contexts,
        ready_barrier=ready_barrier,
    )
    results: list = []
    errors: list = []

    @torch.inference_mode()
    def _thread(ctx, half_inputs):
        try:
            with ctx:
                out = runner.model(**half_inputs)
            results.append((ctx.id, out))
        except BaseException as exc:  # surfaced after join
            errors.append(exc)
            ctx.cpu_signal_event.set()

    with override_forward_context(None):
        threads = [
            threading.Thread(target=_thread, args=(ctx, item[3]))
            for ctx, item in zip(ctxs, prepared)
        ]
        for th in threads:
            th.start()
        ready_barrier.wait()
        ctxs[0].cpu_wait_event.set()
        for th in threads:
            th.join()
    if errors:
        raise errors[0]
    return [out for _, out in sorted(results, key=lambda r: r[0])]
