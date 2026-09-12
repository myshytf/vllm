# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Expert-range maps of the split routed W4A16 prefill launch."""

from types import SimpleNamespace

import torch

from vllm.model_executor.layers.quantization import kquant_hybrid as kq


def test_split_launch_maps_partition_the_secondary_tier():
    # global experts 0..9: local ids of the secondary tier for 6 of them, the
    # other four belong to the kept tier (-1 in the route map)
    emap = torch.tensor([0, -1, 1, 2, -1, 3, -1, 4, 5, -1], dtype=torch.int32)
    state = SimpleNamespace(emap_secondary=emap, num_secondary=6)
    first, second = kq._split_launch_expert_maps(state)
    assert first.tolist() == [0, -1, 1, 2, -1, -1, -1, -1, -1, -1]
    assert second.tolist() == [-1, -1, -1, -1, -1, 3, -1, 4, 5, -1]
    # every secondary route is in exactly one launch; kept routes in neither
    both = (first >= 0) & (second >= 0)
    assert not both.any()
    assert ((first >= 0) | (second >= 0)).tolist() == (emap >= 0).tolist()
    assert first.dtype == second.dtype == torch.int32
    # cached on the state
    cached = kq._split_launch_expert_maps(state)
    assert cached[0] is first and cached[1] is second


def test_split_launch_maps_need_two_experts():
    state = SimpleNamespace(
        emap_secondary=torch.tensor([0, -1], dtype=torch.int32), num_secondary=1
    )
    assert kq._split_launch_expert_maps(state) is None


def test_split_launch_inactive_outside_a_split_half(monkeypatch):
    monkeypatch.setenv("VLLM_K3_MOE_SPLIT_LAUNCH", "1")
    assert not kq._moe_split_launch_active()
