# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Fused Kimi-K3 KDA speculative-decode step (`_C_k3kda.fused_kda_spec_decode`).

The op is a side extension loaded from ``VLLM_K3_KDA_SPEC_FUSED_LIB``; the GPU
test compares it with the served three-kernel chain (conv update, recurrent
KDA, gated output norm) on the production shapes and checks that the conv
state update is bit-identical. The CPU test covers the enablement gate.
"""

import os

import pytest
import torch

from vllm.model_executor.layers.mamba.ops.causal_conv1d import causal_conv1d_update
from vllm.models.kimi_k3.nvidia import kda as kda_module
from vllm.models.kimi_k3.nvidia.kda import (
    is_fused_kda_spec_decode_supported,
    load_kda_spec_fused_op,
)
from vllm.models.kimi_k3.nvidia.ops.third_party.kda.fused_recurrent import (
    fused_recurrent_kda,
)
from vllm.third_party.flash_linear_attention.ops.kda import rms_norm_gated

DEVICE = "cuda"
D, W = 128, 4


def test_fused_spec_decode_gate_requires_library(monkeypatch):
    load_kda_spec_fused_op.cache_clear()
    monkeypatch.delenv("VLLM_K3_KDA_SPEC_FUSED_LIB", raising=False)
    assert not is_fused_kda_spec_decode_supported(
        D, W, 4, torch.bfloat16, torch.bfloat16
    )
    load_kda_spec_fused_op.cache_clear()


def test_fused_spec_decode_gate_shape_conditions(monkeypatch):
    monkeypatch.setattr(kda_module, "load_kda_spec_fused_op", lambda: object())
    monkeypatch.setattr(kda_module, "is_conv_state_dim_first", lambda: False)
    monkeypatch.setattr(
        kda_module.current_platform,
        "is_device_capability_family",
        lambda cap: cap == 120,
    )
    monkeypatch.setattr(
        kda_module.current_platform, "is_device_capability", lambda cap: False
    )
    ok = kda_module.is_fused_kda_spec_decode_supported
    assert ok(D, W, 4, torch.bfloat16, torch.bfloat16)
    assert not ok(
        D, W, 0, torch.bfloat16, torch.bfloat16
    )  # plain decode uses fused_kda_decode
    assert not ok(D, W, 8, torch.bfloat16, torch.bfloat16)  # > 8 tokens per sequence
    assert not ok(64, W, 4, torch.bfloat16, torch.bfloat16)
    assert not ok(D, 3, 4, torch.bfloat16, torch.bfloat16)
    assert not ok(D, W, 4, torch.float16, torch.bfloat16)
    assert not ok(D, W, 4, torch.bfloat16, torch.float32)


@pytest.mark.parametrize(
    ("num_heads", "lengths", "accepted", "lower_bound"),
    [
        (12, [5], [1], -5.0),
        (12, [5, 5, 5, 5], [1, 3, 5, 2], -5.0),
        (12, [5, 3, 0, 0], [4, 2, 1, 1], -5.0),
        (12, [5, 5], [5, 1], None),
        (24, [5, 5], [2, 4], -5.0),
    ],
)
@torch.inference_mode()
def test_fused_kda_spec_decode_matches_chain(num_heads, lengths, accepted, lower_bound):
    if not os.getenv("VLLM_K3_KDA_SPEC_FUSED_LIB"):
        pytest.skip("VLLM_K3_KDA_SPEC_FUSED_LIB is not set")
    load_kda_spec_fused_op.cache_clear()
    if not is_fused_kda_spec_decode_supported(D, W, 4, torch.bfloat16, torch.bfloat16):
        pytest.skip("fused KDA spec decode is not supported on this platform")
    op = load_kda_spec_fused_op()
    torch.manual_seed(4321 + num_heads + sum(lengths))
    H, dim, num_spec = num_heads, num_heads * D, 4
    T_max, C = num_spec + 1, W - 1 + num_spec
    N, T_total = len(lengths), sum(lengths)
    slots = N * T_max + 1

    x = torch.randn(T_total, 3 * dim, dtype=torch.bfloat16, device=DEVICE)
    conv_w = 0.3 * torch.randn(3 * dim, W, dtype=torch.float32, device=DEVICE)
    conv_phys = torch.randn(slots, C, 3 * dim, dtype=torch.bfloat16, device=DEVICE)
    raw_g = torch.randn(1, T_total, H, D, dtype=torch.bfloat16, device=DEVICE)
    beta_store = torch.randn(1, T_total, H + 3, dtype=torch.bfloat16, device=DEVICE)
    raw_beta = beta_store[..., :H]
    A_log = 0.5 * torch.randn(H, dtype=torch.float32, device=DEVICE)
    dt_bias = 0.1 * torch.randn(dim, dtype=torch.float32, device=DEVICE)
    state = 0.01 * torch.randn(slots, H, D, D, dtype=torch.float32, device=DEVICE)
    g2_store = torch.randn(T_total, dim + 16, dtype=torch.bfloat16, device=DEVICE)
    g2 = g2_store[:, :dim].view(T_total, H, D)
    norm_w = torch.randn(D, dtype=torch.float32, device=DEVICE)
    eps = 1e-5
    qsl = torch.tensor(
        [0] + list(torch.cumsum(torch.tensor(lengths), 0)),
        dtype=torch.int32,
        device=DEVICE,
    )
    acc = torch.tensor(accepted, dtype=torch.int32, device=DEVICE)
    sidx = torch.zeros(N, T_max, dtype=torch.int32, device=DEVICE)
    for n, ln in enumerate(lengths):
        if ln > 0:
            sidx[n] = torch.arange(
                1 + n * T_max, 1 + (n + 1) * T_max, dtype=torch.int32
            )

    # served chain
    conv_ref = conv_phys.clone().transpose(-1, -2)
    state_ref = state.clone()
    mixed = causal_conv1d_update(
        x.clone(),
        conv_ref,
        conv_w,
        None,
        activation="silu",
        conv_state_indices=sidx[:, 0],
        num_accepted_tokens=acc,
        query_start_loc=qsl,
        max_query_len=T_max,
        validate_data=False,
        out=torch.empty_like(x),
    )
    q, k, v = (t.reshape(1, T_total, H, D) for t in mixed.split(dim, dim=-1))
    core = torch.empty(1, T_total, H, D, dtype=torch.bfloat16, device=DEVICE)
    out_ref, _ = fused_recurrent_kda(
        q=q,
        k=k,
        v=v,
        raw_g=raw_g,
        raw_beta=raw_beta,
        A_log=A_log,
        dt_bias=dt_bias,
        lower_bound=lower_bound,
        initial_state=state_ref,
        cu_seqlens=qsl,
        ssm_state_indices=sidx,
        num_accepted_tokens=acc,
        out=core,
    )
    y_ref = rms_norm_gated(out_ref, g2, norm_w, None, "sigmoid", eps=eps)

    # fused
    conv_act = conv_phys.clone().transpose(-1, -2)
    state_act = state.clone()
    y_act = torch.empty(1, T_total, H, D, dtype=torch.bfloat16, device=DEVICE)
    w_t = conv_w.reshape(3, dim, W).transpose(1, 2).contiguous()
    op(
        x,
        w_t,
        None,
        conv_act,
        raw_g,
        raw_beta,
        A_log,
        dt_bias,
        sidx,
        qsl,
        acc,
        state_act,
        y_act,
        lower_bound,
        g2,
        norm_w,
        eps,
    )

    rows = torch.tensor(
        [i for n in range(N) for i in range(int(qsl[n]), int(qsl[n + 1]))],
        device=DEVICE,
    )
    torch.testing.assert_close(y_act[0, rows], y_ref[0, rows], atol=3e-2, rtol=3e-2)
    used = sidx.flatten()[sidx.flatten() > 0].long()
    torch.testing.assert_close(state_act[used], state_ref[used], atol=3e-2, rtol=3e-2)
    torch.testing.assert_close(conv_act, conv_ref, atol=0, rtol=0)
