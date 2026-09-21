# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Gating of the split all-reduce (B12X DMA ring reduce-scatter, the caller's
work on its owned rows, all-gather) through the communicator layers."""

from unittest.mock import MagicMock

import pytest
import torch

from vllm.distributed import communication_op
from vllm.distributed.device_communicators import custom_all_reduce as car_mod
from vllm.distributed.device_communicators.base_device_communicator import (
    DeviceCommunicatorBase,
)


class _Group:
    def __init__(self):
        self.calls = []

    def split_owned_rows(self, x):
        self.calls.append(("rows", x))
        return [(0, 4)]

    def all_reduce_in_place_split(self, x, between, *, borrow_output=False):
        self.calls.append(("split", x, between, borrow_output))
        return x


def test_public_entry_forwards_to_the_group(monkeypatch):
    group = _Group()
    monkeypatch.setattr(communication_op, "get_tp_group", lambda: group)
    monkeypatch.setattr(communication_op, "_ubatch_active", lambda: False)
    monkeypatch.setattr(communication_op, "_piecewise", lambda: None)
    x = torch.zeros(8, 4)
    hook = lambda out: None  # noqa: E731
    assert communication_op.tensor_model_parallel_split_owned_rows(x) == [(0, 4)]
    assert (
        communication_op.tensor_model_parallel_all_reduce_in_place_split(
            x, hook, borrow_output=True
        )
        is x
    )
    assert group.calls == [("rows", x), ("split", x, hook, True)]


@pytest.mark.parametrize("region", ["ubatch", "piecewise"])
def test_public_entry_declines_split_prefill_regions(monkeypatch, region):
    group = _Group()
    monkeypatch.setattr(communication_op, "get_tp_group", lambda: group)
    monkeypatch.setattr(communication_op, "_ubatch_active", lambda: region == "ubatch")
    monkeypatch.setattr(
        communication_op,
        "_piecewise",
        lambda: object() if region == "piecewise" else None,
    )
    x = torch.zeros(8, 4)
    assert communication_op.tensor_model_parallel_split_owned_rows(x) is None
    assert (
        communication_op.tensor_model_parallel_all_reduce_in_place_split(
            x, lambda o: None
        )
        is None
    )
    assert group.calls == []


def test_base_communicator_is_unsupported():
    comm = DeviceCommunicatorBase.__new__(DeviceCommunicatorBase)
    x = torch.zeros(8, 4)
    assert comm.split_owned_rows(x) is None
    assert comm.all_reduce_in_place_split(x, lambda o: None) is None


def _custom_ar(ring, *, capturing=False, oneshot_max=16, twoshot=False):
    ca = car_mod.CustomAllreduce.__new__(car_mod.CustomAllreduce)
    ca.disabled = False
    ca._IS_CAPTURING = capturing
    ca._pcie_dma = ring
    ca._pcie_allreduce_max_size = oneshot_max
    ca._pcie_twoshot_accepts = lambda inp: twoshot
    ca._pcie_runtime_stream = lambda: None
    ca.should_custom_ar = lambda inp: True
    return ca


def _ring(accept=True):
    ring = MagicMock()
    ring.can_all_reduce_split.side_effect = lambda inp: accept
    ring.split_owned_rows.side_effect = lambda inp: [(4, 4)] if accept else None
    ring.all_reduce_in_place_split.side_effect = lambda inp, between, **kw: (
        between(inp),
        inp,
    )[1]
    return ring


def test_custom_all_reduce_dispatches_the_split_to_the_ring():
    ring = _ring()
    ca = _custom_ar(ring)
    x = torch.zeros(8, 4, dtype=torch.bfloat16)
    seen = []
    out = ca.pcie_dma_all_reduce_split(x, lambda o: seen.append(o), borrow_output=True)
    assert out is x and seen == [x]
    ring.all_reduce_in_place_split.assert_called_once()
    assert ring.all_reduce_in_place_split.call_args.kwargs == {"borrow_output": True}
    assert ca.pcie_dma_split_owned_rows(x) == [(4, 4)]


@pytest.mark.parametrize("case", ["capturing", "oneshot", "twoshot", "ring_refuses"])
def test_custom_all_reduce_declines_when_the_ring_path_is_unavailable(case):
    ring = _ring(accept=case != "ring_refuses")
    ca = _custom_ar(
        ring,
        capturing=case == "capturing",
        oneshot_max=(1 << 30) if case == "oneshot" else 16,
        twoshot=case == "twoshot",
    )
    x = torch.zeros(8, 4, dtype=torch.bfloat16)
    assert ca.pcie_dma_all_reduce_split(x, lambda o: None) is None
    ring.all_reduce_in_place_split.assert_not_called()
    if case != "capturing":
        assert ca.pcie_dma_split_owned_rows(x) is None
