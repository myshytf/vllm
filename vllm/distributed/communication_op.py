# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from typing import Any

import torch
import torch.distributed

from .parallel_state import get_tp_group


def _ubatch_comm_region():
    """Ubatch-aware wrapper for a TP collective.

    When this thread runs one half of a split prefill (``UBatchContext``
    active), the collective is issued on the shared comm stream after the
    half's compute so far has been recorded, and the CPU then yields to the
    other half so its compute fills the GPU while the collective runs; on
    resumption the compute stream waits for the collective. Outside a ubatch
    context this is a no-op.
    """
    import threading

    from vllm.v1.worker import ubatching
    from vllm.v1.worker.ubatching import (
        dbo_switch_to_comm_sync,
        dbo_yield_and_switch_from_comm_to_compute,
    )

    class _Region:
        def __enter__(self):
            # Only threads that own a ubatch context take part; any other
            # thread (e.g. a background store thread) keeps the plain path.
            self.active = threading.get_ident() in ubatching._THREAD_ID_TO_CONTEXT
            if self.active:
                dbo_switch_to_comm_sync()
            return self

        def __exit__(self, exc_type, exc, tb):
            if self.active and exc_type is None:
                dbo_yield_and_switch_from_comm_to_compute()
            return False

    return _Region()


def tensor_model_parallel_all_reduce(input_: torch.Tensor) -> torch.Tensor:
    """All-reduce the input tensor across model parallel group."""
    with _ubatch_comm_region():
        return get_tp_group().all_reduce(input_)


def tensor_model_parallel_all_reduce_in_place(input_: torch.Tensor) -> torch.Tensor:
    """All-reduce a dead input tensor without allocating an output tensor."""
    with _ubatch_comm_region():
        return get_tp_group().all_reduce_in_place(input_)


def tensor_model_parallel_all_gather(
    input_: torch.Tensor, dim: int = -1
) -> torch.Tensor:
    """All-gather the input tensor across model parallel group."""
    return get_tp_group().all_gather(input_, dim)


def tensor_model_parallel_all_gatherv(
    input_: torch.Tensor, sizes: list[int], dim: int = 0
) -> torch.Tensor:
    """All-gather variable-length tensor slices across the model-parallel group."""
    tp_group = get_tp_group()
    if tp_group.world_size == 1:
        return input_
    return tp_group.all_gatherv(input_, dim=dim, sizes=sizes)


def tensor_model_parallel_reduce_scatter(
    input_: torch.Tensor, dim: int = -1
) -> torch.Tensor:
    """Reduce-Scatter the input tensor across model parallel group."""
    return get_tp_group().reduce_scatter(input_, dim)


def tensor_model_parallel_gather(
    input_: torch.Tensor, dst: int = 0, dim: int = -1
) -> torch.Tensor | None:
    """Gather the input tensor across model parallel group."""
    return get_tp_group().gather(input_, dst, dim)


def broadcast_tensor_dict(
    tensor_dict: dict[Any, torch.Tensor | Any] | None = None, src: int = 0
):
    if not torch.distributed.is_initialized():
        return tensor_dict
    return get_tp_group().broadcast_tensor_dict(tensor_dict, src)
