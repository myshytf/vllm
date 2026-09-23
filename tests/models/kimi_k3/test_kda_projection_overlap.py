# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Split KDA input projections overlapped on the auxiliary stream.

The MXFP8 Q/K/V projection runs on the model's auxiliary stream while the BF16
gate/factor/beta projection (and f_b for a replicated f_a) runs on the main
stream, only inside CUDA graph capture of at most
``VLLM_KIMI_KDA_PROJECTION_STREAM_TOKEN_THRESHOLD`` rows. The overlap must not
change a single output bit: every branch runs the same kernel on the same input
as sequential dispatch.
"""

from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F

from vllm.models.kimi_k3.nvidia import kda as kimi_kda

WIDTH, HEADS, HIDDEN, HEAD_DIM, PADDING = 256, 2, 128, 128, 14


def _layer(threshold: int, *, shard_f_a: bool, events, stream, weights=None):
    if weights is None:
        qkv = gfab = fb = None
    else:
        qkv, gfab, fb = weights
    return SimpleNamespace(
        local_projection_size=WIDTH,
        local_fa_size=HEAD_DIM,
        local_num_heads=HEADS,
        in_proj_padding=PADDING,
        shard_f_a=shard_f_a,
        _split_projection_overlap_max_tokens=threshold,
        _projection_aux_stream=stream,
        _projection_events=events,
        in_proj_qkv=lambda value: (F.linear(value, qkv), None),
        in_proj_gfab=lambda value: (F.linear(value, gfab), None),
        f_b_proj=lambda value: (F.linear(value, fb), None),
    )


def _active(layer, rows: int) -> bool:
    return kimi_kda.KimiK3DeltaAttention._split_projection_overlap_active(layer, rows)


@pytest.mark.parametrize(
    "rows,threshold,events,capturing,batch_invariant,expected",
    [
        (4, 16, True, True, False, True),
        (16, 16, True, True, False, True),
        (17, 16, True, True, False, False),
        (0, 16, True, True, False, False),
        (4, 0, True, True, False, False),
        (4, 16, False, True, False, False),
        (4, 16, True, False, False, False),
        (4, 16, True, True, True, False),
    ],
)
def test_overlap_gate(
    rows, threshold, events, capturing, batch_invariant, expected, monkeypatch
):
    """The overlap is limited to captured graphs within the row threshold."""
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: capturing)
    monkeypatch.setenv("VLLM_BATCH_INVARIANT", str(int(batch_invariant)))
    layer = _layer(
        threshold,
        shard_f_a=False,
        events=(object(), object()) if events else None,
        stream=object() if events else None,
    )
    assert _active(layer, rows) is expected


def _serial_reference(layer, x):
    """The served sequential order: Q/K/V, gate/factor/beta, then f_b."""
    mixed_qkv = layer.in_proj_qkv(x)[0]
    sizes = [WIDTH, HEAD_DIM, HEADS, PADDING]
    g_proj_states, f_a, beta = layer.in_proj_gfab(x)[0].split(sizes, dim=-1)[:3]
    g1 = layer.f_b_proj(f_a)[0]
    return mixed_qkv, g_proj_states, f_a, g1, beta


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA streams required")
@pytest.mark.parametrize("rows", [1, 4, 8, 16])
@pytest.mark.parametrize("shard_f_a", [False, True])
@torch.inference_mode()
def test_overlapped_capture_is_bit_identical(rows, shard_f_a):
    """Graph replay of the overlapped branches equals sequential dispatch."""
    torch.manual_seed(1234 + rows)
    weights = [
        torch.randn(n, k, device="cuda", dtype=torch.bfloat16)
        for n, k in (
            (3 * WIDTH, HIDDEN),
            (WIDTH + HEAD_DIM + HEADS + PADDING, HIDDEN),
            (WIDTH, HEAD_DIM),
        )
    ]
    layer = _layer(
        16,
        shard_f_a=shard_f_a,
        events=(torch.cuda.Event(), torch.cuda.Event()),
        stream=torch.cuda.Stream(),
        weights=weights,
    )
    x = torch.randn(rows, HIDDEN, device="cuda", dtype=torch.bfloat16)
    project = kimi_kda.KimiK3DeltaAttention._project_split_input_overlapped
    for _ in range(3):  # warm the kernels outside capture
        _serial_reference(layer, x)
    assert not _active(layer, rows)  # eager calls stay sequential
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        assert _active(layer, rows)
        actual = project(layer, x)
    mixed_qkv, g_proj_states, f_a, g1, beta = actual
    assert (g1 is None) == shard_f_a
    for _ in range(16):
        x.normal_()
        expected = _serial_reference(layer, x)
        for value in actual:
            if value is not None:
                value.fill_(float("nan"))
        for _ in range(8):
            graph.replay()
        torch.accelerator.synchronize()
        for value, reference in zip(actual, expected):
            if value is None:
                continue
            assert torch.isfinite(value).all()
            assert torch.equal(value, reference)
    graph.reset()
