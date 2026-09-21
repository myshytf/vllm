# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Contract of the decode latent norm + shard window (`_C_k3norm`): the gate,
the wrapper's op call, and the transform's decision to use it."""

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch

from vllm.models.kimi_k3.nvidia.ops import norm_window


@pytest.fixture(autouse=True)
def _clear_loader_cache():
    norm_window.load_rms_norm_window_op.cache_clear()
    yield
    norm_window.load_rms_norm_window_op.cache_clear()


def test_loader_is_none_without_the_library(monkeypatch):
    monkeypatch.delenv("VLLM_K3_LATENT_NORM_WINDOW_LIB", raising=False)
    assert norm_window.load_rms_norm_window_op() is None
    x = torch.zeros(4, 3584, dtype=torch.bfloat16)
    assert not norm_window.can_rms_norm_window(
        x, torch.ones(3584, dtype=torch.bfloat16), 0, 400
    )
    with pytest.raises(RuntimeError):
        norm_window.rms_norm_window(
            x, torch.ones(3584, dtype=torch.bfloat16), 1e-6, 0, 400
        )


def _fake_cuda_tensor(shape, dtype=torch.bfloat16, stride0=None, ptr=1 << 20):
    t = MagicMock(spec=torch.Tensor)
    t.ndim = len(shape)
    t.shape = torch.Size(shape)
    t.is_cuda = True
    t.dtype = dtype
    t.element_size.return_value = torch.empty(0, dtype=dtype).element_size()
    t.stride.side_effect = lambda d=None: (
        (stride0 if stride0 is not None else shape[1], 1)[d]
        if d is not None
        else (shape[1], 1)
    )
    t.data_ptr.return_value = ptr
    t.is_contiguous.return_value = True
    t.device = torch.device("cuda", 0)
    return t


def test_gate_accepts_the_served_decode_operands(monkeypatch):
    monkeypatch.setattr(norm_window, "load_rms_norm_window_op", lambda: object())
    x = _fake_cuda_tensor((4, 3584))
    w = _fake_cuda_tensor((3584,))
    with torch.no_grad():
        assert norm_window.can_rms_norm_window(x, w, 8 * 400, 400)
    # Autograd-enabled callers keep the served path.
    assert not norm_window.can_rms_norm_window(x, w, 8 * 400, 400)


@pytest.mark.parametrize(
    "case",
    [
        "cpu",
        "fp32",
        "col0_misaligned",
        "width_misaligned",
        "col0_past_hidden",
        "weight_dtype",
        "weight_shape",
        "unaligned_ptr",
        "no_rows",
    ],
)
def test_gate_rejects_operands_outside_the_contract(monkeypatch, case):
    monkeypatch.setattr(norm_window, "load_rms_norm_window_op", lambda: object())
    x = _fake_cuda_tensor((4, 3584))
    w = _fake_cuda_tensor((3584,))
    col0, width = 0, 400
    if case == "cpu":
        x.is_cuda = False
    elif case == "fp32":
        x.dtype = torch.float32
    elif case == "col0_misaligned":
        col0 = 4
    elif case == "width_misaligned":
        width = 396
    elif case == "col0_past_hidden":
        col0 = 3584
    elif case == "weight_dtype":
        w.dtype = torch.float32
    elif case == "weight_shape":
        w.shape = torch.Size((3583,))
    elif case == "unaligned_ptr":
        x.data_ptr.return_value = (1 << 20) + 2
    elif case == "no_rows":
        x.shape = torch.Size((0, 3584))
    assert not norm_window.can_rms_norm_window(x, w, col0, width)


def test_wrapper_calls_the_op_with_the_window(monkeypatch):
    calls = []

    def fake_op(out, x, weight, eps, col0, batch_invariant):
        calls.append(
            (
                tuple(out.shape),
                out.dtype,
                x is x_in,
                weight is w_in,
                eps,
                col0,
                batch_invariant,
            )
        )

    monkeypatch.setattr(norm_window, "load_rms_norm_window_op", lambda: fake_op)
    x_in = torch.zeros(3, 3584, dtype=torch.bfloat16)
    w_in = torch.ones(3584, dtype=torch.bfloat16)
    out = norm_window.rms_norm_window(
        x_in, w_in, 1e-6, 3200, 400, batch_invariant=False
    )
    assert out.shape == (3, 400) and out.dtype == torch.bfloat16
    assert calls == [((3, 400), torch.bfloat16, True, True, 1e-6, 3200, False)]


def test_transform_gate_requires_the_shard_pack_path(monkeypatch):
    """The transform uses the window only when the up-projection would pack
    a shard (decode rows) and the norm is a weighted full-width RMSNorm."""
    from vllm.models.kimi_k3.nvidia import model as kimi_model

    monkeypatch.setattr(kimi_model, "can_rms_norm_window", lambda *a: True)
    monkeypatch.delenv("VLLM_KQUANT_CAPTURE_DIR", raising=False)
    up_proj = MagicMock(spec=kimi_model.KimiPaddedRowParallelLinear)
    up_proj.tp_rank = 2
    up_proj.shard_width = 400
    up_proj.can_shard_pack.return_value = True
    norm = SimpleNamespace(
        pass_weight=True,
        variance_size_override=None,
        hidden_size=3584,
        weight=SimpleNamespace(data=torch.ones(3584, dtype=torch.bfloat16)),
    )
    transform = kimi_model.KimiRoutedOutputTransform.__new__(
        kimi_model.KimiRoutedOutputTransform
    )
    torch.nn.Module.__init__(transform)
    transform.norm = norm
    transform.up_proj = up_proj
    x = torch.zeros(4, 3584, dtype=torch.bfloat16)
    assert transform.can_norm_window_shard(x)
    up_proj.can_shard_pack.return_value = False
    assert not transform.can_norm_window_shard(x)
    up_proj.can_shard_pack.return_value = True
    norm.pass_weight = False
    assert not transform.can_norm_window_shard(x)
    norm.pass_weight = True
    transform.norm = None
    assert not transform.can_norm_window_shard(x)
    transform.norm = norm
    monkeypatch.setenv("VLLM_KQUANT_CAPTURE_DIR", "/tmp/capture")
    assert not transform.can_norm_window_shard(x)
