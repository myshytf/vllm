# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Measure cache geometry with synthetic token prompts and completion metrics.

No cache reset, restart, configuration change, or user prompt capture is used.
TTFT requires a nonempty completion delta; role-only events do not count.
Global metric deltas are accepted only if exactly one request completed.
"""

import argparse
import hashlib
import json
import math
import time
import urllib.request
from pathlib import Path

import regex as re

METRICS = re.compile(r"^([a-zA-Z_:][a-zA-Z0-9_:]*)(?:\{[^}]*\})? ([-+0-9.eE]+)$")


def snapshot(base):
    with urllib.request.urlopen(base + "/metrics", timeout=5) as response:
        raw = response.read().decode()
    values = {}
    for line in raw.splitlines():
        match = METRICS.fullmatch(line)
        if match:
            name, value = match.groups()
            values[name] = values.get(name, 0) + float(value)
    return values, raw


def measure(base, tokens, salt, max_tokens):
    before, _ = snapshot(base)
    if before["vllm:num_requests_running"] or before["vllm:num_requests_waiting"]:
        raise RuntimeError("Engine has active requests; measurement was not submitted")
    payload = dict(
        model="Kimi-K3",
        prompt=tokens,
        cache_salt=salt,
        max_tokens=max_tokens,
        temperature=0,
        top_p=1,
        ignore_eos=True,
        return_token_ids=True,
        stream=True,
        stream_options={"include_usage": True},
        logprobs=1,
    )
    req = urllib.request.Request(
        base + "/v1/completions",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    started_unix_ns = time.time_ns()
    start = time.monotonic()
    ttft = None
    usage = None
    text = ""
    token_ids = []
    finite = True
    probability_count = 0
    with urllib.request.urlopen(req, timeout=120) as response:
        for raw in response:
            if not raw.startswith(b"data:"):
                continue
            data = raw[5:].strip()
            if data == b"[DONE]":
                continue
            event = json.loads(data)
            if "error" in event:
                raise RuntimeError(event["error"])
            if event.get("usage"):
                usage = event["usage"]
            for choice in event.get("choices", []):
                delta = choice.get("text", "")
                ids = choice.get("token_ids") or []
                if ttft is None and (delta or ids):
                    ttft = time.monotonic() - start
                text += delta
                token_ids.extend(ids)
                for value in (choice.get("logprobs") or {}).get("token_logprobs") or []:
                    finite &= value is not None and math.isfinite(value)
                    probability_count += 1
    wall = time.monotonic() - start
    finite &= probability_count == len(token_ids) and probability_count > 0
    after = before
    deadline = time.monotonic() + 8
    while time.monotonic() < deadline:
        after, _ = snapshot(base)
        if (
            after["vllm:request_prefill_kv_computed_tokens_count"]
            > before["vllm:request_prefill_kv_computed_tokens_count"]
        ):
            break
        time.sleep(0.2)
    names = {
        "requests": "request_prefill_kv_computed_tokens_count",
        "computed_tokens": "request_prefill_kv_computed_tokens_sum",
        "local_hits": "prefix_cache_hits_total",
        "external_hits": "external_prefix_cache_hits_total",
        "prefill_s": "request_prefill_time_seconds_sum",
        "queue_s": "request_queue_time_seconds_sum",
        "engine_ttft_s": "time_to_first_token_seconds_sum",
        "drafts": "spec_decode_num_drafts_total",
        "accepted_tokens": "spec_decode_num_accepted_tokens_total",
        "preemptions": "num_preemptions_total",
    }
    delta = {
        key: after["vllm:" + name] - before["vllm:" + name]
        for key, name in names.items()
    }
    valid = (
        delta["requests"] == 1
        and not after["vllm:num_requests_running"]
        and not after["vllm:num_requests_waiting"]
        and all(value >= 0 for value in delta.values())
        and delta["computed_tokens"] + delta["local_hits"] + delta["external_hits"]
        == len(tokens)
    )
    return dict(
        prompt_tokens=len(tokens),
        ttft_s=ttft,
        wall_s=wall,
        usage=usage,
        cache_salt=salt,
        started_unix_ns=started_unix_ns,
        token_ids=token_ids,
        output_sha256=hashlib.sha256(text.encode()).hexdigest(),
        finite_logprobs=finite,
        metric_attribution_valid=valid,
        metric_delta=delta,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", default="http://127.0.0.1:8090")
    parser.add_argument("--lengths", nargs="+", type=int, default=[17125])
    parser.add_argument("--max-tokens", type=int, default=16)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    _, raw = snapshot(args.base)
    (args.out / "metrics-before.prom").write_text(raw)
    rows = []
    run_id = str(time.time_ns())
    for length in args.lengths:
        tokens = [1000 + (i * 37) % 10000 for i in range(length)]
        salt = f"ttft-audit-{run_id}-{length}"
        for tag, prompt in [
            ("cold", tokens),
            ("warm", tokens),
            ("warm_repeat", tokens),
            ("appended", tokens + [42] * 128),
        ]:
            row = measure(args.base, prompt, salt, args.max_tokens)
            row.update(tag=tag, case_length=length)
            rows.append(row)
            (args.out / "results.json").write_text(json.dumps(rows, indent=2) + "\n")
            print(json.dumps(row), flush=True)
            if not row["metric_attribution_valid"] or not row["finite_logprobs"]:
                raise RuntimeError(
                    "Measurement cannot be attributed or output is nonfinite"
                )
    _, raw = snapshot(args.base)
    (args.out / "metrics-after.prom").write_text(raw)


if __name__ == "__main__":
    main()
