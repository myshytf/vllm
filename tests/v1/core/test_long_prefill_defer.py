# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Streak-bounded deferral of a long in-progress prefill while requests wait.

A long cold prefill takes the whole token budget every step, so a turn that
arrives behind it waits for the prefill's whole remainder. With the deferral,
the running prefill yields one step (streak bound 1) whenever the waiting
queue is non-empty, the waiting request is admitted, and the prefill is
guaranteed the next step. With the bound at 0 the scheduler is strict FCFS.
"""

import os

import pytest

if os.environ.get("VLLM_TEST_FORCE_CPU_PLATFORM") == "1":
    # A CUDA build running on a host without a visible GPU resolves no
    # platform; the scheduler needs only the CPU platform's config paths.
    import vllm.platforms as _platforms

    _platforms.builtin_platform_plugins["cpu"] = (
        lambda: "vllm.platforms.cpu.CpuPlatform"
    )

from vllm.v1.outputs import ModelRunnerOutput
from vllm.v1.request import RequestStatus

from .utils import create_requests, create_scheduler

pytestmark = pytest.mark.cpu_test

BUDGET = 64
# A local config directory stands in for the hub model on offline hosts.
_MODEL = os.environ.get("VLLM_TEST_SCHEDULER_MODEL", "facebook/opt-125m")


def _step(scheduler, output):
    """Feed an output back without sampling any token (prefills continue)."""
    req_ids = list(output.num_scheduled_tokens)
    scheduler.update_from_output(
        output,
        ModelRunnerOutput(
            req_ids=req_ids,
            req_id_to_index={rid: i for i, rid in enumerate(req_ids)},
            sampled_token_ids=[[] for _ in req_ids],
            logprobs=None,
            prompt_logprobs_dict={},
            pooler_output=[],
        ),
    )


def _make(max_streak: int):
    scheduler = create_scheduler(
        model=_MODEL,
        skip_tokenizer_init=True,
        max_num_seqs=4,
        max_num_batched_tokens=BUDGET,
        enable_chunked_prefill=True,
    )
    scheduler.long_prefill_defer_max_streak = max_streak
    (long_req,) = create_requests(
        num_requests=1, num_tokens=BUDGET * 8, req_ids=["long"]
    )
    # Short turns: a few tokens each (a warm turn's cache-miss remainder).
    short_a, short_b = create_requests(num_requests=2, num_tokens=8, req_ids=["a", "b"])
    return scheduler, long_req, short_a, short_b


def test_long_prefill_yields_one_step_then_resumes():
    scheduler, long_req, short_a, short_b = _make(max_streak=1)
    scheduler.add_request(long_req)
    out = scheduler.schedule()
    assert out.num_scheduled_tokens == {"long": BUDGET}
    _step(scheduler, out)

    # A short turn arrives while the long prefill is in progress: the long
    # prefill is deferred for this step and the turn is admitted at once.
    scheduler.add_request(short_a)
    out = scheduler.schedule()
    assert "long" not in out.num_scheduled_tokens
    assert out.num_scheduled_tokens["a"] == 8
    assert scheduler._long_prefill_defer_streak == {"long": 1}
    _step(scheduler, out)

    # Another turn waits, but the streak bound guarantees the long prefill
    # this step: it takes the whole budget and the turn keeps waiting.
    scheduler.add_request(short_b)
    out = scheduler.schedule()
    assert out.num_scheduled_tokens["long"] == BUDGET
    assert "b" not in out.num_scheduled_tokens
    assert "long" not in scheduler._long_prefill_defer_streak
    _step(scheduler, out)

    # The streak restarts: the long prefill yields once more and the second
    # turn is admitted, so a waiting turn never waits more than one step.
    out = scheduler.schedule()
    assert "long" not in out.num_scheduled_tokens
    assert out.num_scheduled_tokens["b"] == 8
    _step(scheduler, out)

    # With nothing waiting the long prefill is never deferred.
    out = scheduler.schedule()
    assert out.num_scheduled_tokens["long"] == BUDGET
    assert not scheduler.waiting


def test_deferral_disabled_is_strict_fcfs():
    scheduler, long_req, short_a, _ = _make(max_streak=0)
    scheduler.add_request(long_req)
    out = scheduler.schedule()
    _step(scheduler, out)
    scheduler.add_request(short_a)
    out = scheduler.schedule()
    # The running long prefill keeps the whole budget; the turn waits.
    assert out.num_scheduled_tokens == {"long": BUDGET}
    assert "a" not in out.num_scheduled_tokens
    assert scheduler._long_prefill_defer_streak == {}


def test_streak_state_is_dropped_when_the_request_finishes():
    scheduler, long_req, short_a, _ = _make(max_streak=1)
    scheduler.add_request(long_req)
    _step(scheduler, scheduler.schedule())
    scheduler.add_request(short_a)
    out = scheduler.schedule()
    assert scheduler._long_prefill_defer_streak == {"long": 1}
    _step(scheduler, out)
    scheduler.finish_requests("long", RequestStatus.FINISHED_ABORTED)
    assert "long" not in scheduler._long_prefill_defer_streak
