# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Decode-size BF16 projection routing through the b12x GEMV kernels.

At decode row counts (m<=8) the small per-rank projections of a Kimi-K3 MoE
layer — the routed latent down-projection (K 7168, N 400) and the latent
up-projection (K 3584, N 7168 replicated) — run cuBLAS at several times
their weight-read floor. The b12x ``b12x::bf16_gemv_small_n`` op routes those
shapes to its small-N/tensor-core GEMV kernels (fp32 accumulation,
fixed-order reduction, one bf16 rounding; deterministic but not bit-identical
to cuBLAS — same precision class, so this is a qualification-track switch).

Opt-in: ``VLLM_K3_BF16_TC_GEMV=1`` (default off keeps every cuBLAS call
site). The op itself also requires ``B12X_BF16_GEMV_TC=1`` for the tc
backend and otherwise uses its SIMT kernel; shapes the op does not cover
(m>8, K not kernel-eligible, misalignment) fall back to ``F.linear`` inside
the op, so routing is safe at any row count. The MoE router gate is NOT
routed here: its logits are fp32 and the GEMV emits bf16.
"""

from __future__ import annotations

import functools
import os

import torch


@functools.lru_cache(maxsize=1)
def enabled() -> bool:
    return os.getenv("VLLM_K3_BF16_TC_GEMV", "0") == "1"


def bf16_gemv_mm(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor | None:
    """Return the b12x GEMV ``x @ weight.T``, or ``None`` to keep cuBLAS.

    ``None`` is returned only when the opt-in is off, the operands are not
    CUDA bf16, or b12x is unavailable; every kernel-level shape fallback
    lives inside the op.
    """
    if not enabled():
        return None
    if not (x.is_cuda and x.dtype == weight.dtype == torch.bfloat16):
        return None
    try:
        from b12x.gemm import bf16_gemv

        bf16_gemv.bf16_gemv_small_n  # noqa: B018  (registers the op)
    except Exception:
        return None
    return torch.ops.b12x.bf16_gemv_small_n(x, weight)


def bf16_gemv_mm_or_cublas(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    """One-liner for call sites that currently do ``torch.mm(x, weight.t())``."""
    y = bf16_gemv_mm(x, weight)
    if y is not None:
        return y
    return torch.mm(x, weight.t())


def precompile_bf16_gemv(weight: torch.Tensor) -> None:
    """Compile and warm-run every decode-m variant for ``weight``'s shape at
    load time (no JIT or lazy module load under CUDA-graph capture later).
    Compiles dedup by ``(n, k)`` across layers. No-op when the switch is off.
    """
    if not enabled():
        return
    try:
        from b12x.gemm import bf16_gemv

        bf16_gemv.precompile(weight)
    except Exception:
        import logging

        logging.getLogger("vllm.k3_bf16_gemv").warning(
            "bf16 GEMV precompile failed; decode keeps cuBLAS", exc_info=True
        )
