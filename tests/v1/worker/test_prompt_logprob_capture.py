# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch
from safetensors import safe_open

from vllm.v1.worker.gpu.sample import prompt_logprob


def test_kld_capture_skips_synthetic_warmup(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("VLLM_KLD_CAPTURE_DIR", str(tmp_path))
    assert not prompt_logprob._should_capture_kld_batch(["_warmup_0_"])
    assert not prompt_logprob._should_capture_kld_batch(["_dummy_req_0"])
    assert prompt_logprob._should_capture_kld_batch(["cmpl-real"])


def test_kld_capture_requires_exclusive_batch(caplog) -> None:
    assert prompt_logprob._kld_capture_batch_is_exclusive(["cmpl-a"], 1)
    # A concurrent request in the batch skips the capture instead of raising.
    assert not prompt_logprob._kld_capture_batch_is_exclusive(["cmpl-a", "cmpl-b"], 1)
    assert not prompt_logprob._kld_capture_batch_is_exclusive(["cmpl-a", "cmpl-b"], 2)
    assert not prompt_logprob._kld_capture_batch_is_exclusive(["cmpl-a"], 0)


def test_capture_kld_prompt_logits(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("VLLM_KLD_CAPTURE_DIR", str(tmp_path))
    monkeypatch.setattr(prompt_logprob, "is_global_first_rank", lambda: True)
    logits = torch.arange(15, dtype=torch.bfloat16).reshape(3, 5)

    prompt_logprob._maybe_capture_kld_prompt_logits(
        logits,
        req_id="req/unsafe",
        start_idx=7,
        vocab_size=4,
    )

    output = tmp_path / "req_unsafe" / "logits.rows-000007-000010.safetensors"
    with safe_open(output, framework="pt", device="cpu") as handle:
        assert handle.metadata() == {
            "request_id": "req/unsafe",
            "row_start": "7",
            "row_end": "10",
            "vocab_size": "4",
        }
        captured = handle.get_tensor("logits")
    torch.testing.assert_close(captured, logits[:, :4].float())
