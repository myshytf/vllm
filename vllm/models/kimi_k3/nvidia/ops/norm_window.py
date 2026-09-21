# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""RMSNorm of full rows stored as one column window (`_C_k3norm`).

Kimi-K3's decode MoE tail normalizes the reduced routed latent with
``_C.rms_norm`` and then copies this rank's padded up-projection input shard
out of the normalized rows. The side extension loaded from
``VLLM_K3_LATENT_NORM_WINDOW_LIB`` performs the same normalization arithmetic
(same per-thread variance partials, block reduction, ``rsqrtf`` scale and
per-element products) and stores only ``[col0, col0 + width)`` of every row,
zero-filled past the hidden size, so its output equals
``rms_norm(x)[:, col0:col0 + width]`` bit for bit without the shard copy.
"""

import functools
import os
from collections.abc import Callable

import torch

_NORM_WINDOW_LIB_ENV = "VLLM_K3_LATENT_NORM_WINDOW_LIB"


@functools.cache
def load_rms_norm_window_op() -> Callable[..., None] | None:
    """``_C_k3norm.rms_norm_window`` from ``VLLM_K3_LATENT_NORM_WINDOW_LIB``,
    or None (the served norm + shard copy)."""
    path = os.getenv(_NORM_WINDOW_LIB_ENV, "")
    if not path:
        return None
    torch.ops.load_library(path)
    return torch.ops._C_k3norm.rms_norm_window


def can_rms_norm_window(
    x: torch.Tensor, weight: torch.Tensor, col0: int, width: int
) -> bool:
    """Whether ``rms_norm_window`` reproduces ``_C.rms_norm`` + shard copy for
    these operands (the extension's vectorized path: 16-byte vectors)."""
    if load_rms_norm_window_op() is None:
        return False
    if x.ndim != 2 or not x.is_cuda or x.stride(-1) != 1:
        return False
    if x.dtype not in (torch.bfloat16, torch.float16):
        return False
    hidden = x.shape[1]
    vec = 8 if x.element_size() == 2 else 4
    return (
        x.shape[0] > 0
        and hidden % vec == 0
        and x.data_ptr() % 16 == 0
        and x.stride(0) % vec == 0
        and weight.is_cuda
        and weight.dtype == x.dtype
        and weight.ndim == 1
        and weight.shape[0] == hidden
        and weight.is_contiguous()
        and 0 <= col0 < hidden
        and col0 % vec == 0
        and width > 0
        and width % vec == 0
        and not torch.is_grad_enabled()
    )


def rms_norm_window(
    x: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
    col0: int,
    width: int,
    *,
    batch_invariant: bool = False,
) -> torch.Tensor:
    """``rms_norm(x, weight, eps)[:, col0:col0 + width]`` (zeros past the
    hidden size) in one launch; ``can_rms_norm_window`` must hold."""
    op = load_rms_norm_window_op()
    if op is None:
        raise RuntimeError(f"{_NORM_WINDOW_LIB_ENV} is not set")
    out = torch.empty((x.shape[0], width), dtype=x.dtype, device=x.device)
    op(out, x, weight, float(eps), int(col0), bool(batch_invariant))
    return out
