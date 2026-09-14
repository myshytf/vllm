# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import torch

from benchmarks.kernels.kimi_k3_decode_probe import KimiDecodeProbeWorker


def test_prefetch_control_restores_inference_descriptors_without_reallocation(
    monkeypatch,
):
    """Warmup creates inference tensors, but an idle worker RPC has no such mode."""
    monkeypatch.setattr(torch.accelerator, "synchronize", lambda: None)
    with torch.inference_mode():
        segments = torch.tensor([4096, 8192, 32768, 16384], dtype=torch.int64)
    original = segments.clone()
    pointer = segments.data_ptr()
    worker = KimiDecodeProbeWorker()
    worker.rank = 0
    worker._k3_probe_plans = [
        ("model.layers.0.self_attn", "A", SimpleNamespace(segs=segments), original)
    ]
    assert not torch.is_inference_mode_enabled()
    assert worker.k3_set_prefetch_policy("off")["active_bytes"] == 0
    assert torch.equal(segments[::2], original[::2])
    assert segments.data_ptr() == pointer
    assert worker.k3_set_prefetch_policy("all")["active_bytes"] == 24576
    assert torch.equal(segments, original)
    assert segments.data_ptr() == pointer
