# Kimi-K3 recurrent prefix replay with a separate draft cache

Status: **implemented; CPU-qualified; GPU qualification pending**.

Kimi-K3 TP9/DCP9 uses a 1,536-token hash unit, 4,608-token recurrent blocks,
13,824-token logical target attention pages and a separately annotated
1,536-token DFlash2 cache group. Attention KV and recurrent state must resume
at the same token boundary. Attention history can authenticate an earlier
prefix; recurrent state must exist at exactly the requested boundary.

For a 17,125-token prompt, the final hash boundary is 16,896. The draft rewind
requires state at 15,360. Stopping at 16,896 does not create that earlier state.
The scheduler therefore stops at the draft-reusable boundary, and the recurrent
manager publishes and preserves the corresponding state through copy-on-write.
The target manager shares the draft rewind predicate even when only the draft
cache group has the EAGLE annotation.

A matching published attention-tail hash can prove the smaller prefix inside
the same append-only page. Lookup is bounded to that page and returns no more
than the aligned resume cap. A divergent suffix does not authenticate that
alias. This lookup ports the mechanism of
[local-inference-lab/vLLM #676](https://github.com/local-inference-lab/vllm/pull/676)
to the Kimi serving source.

The implementation leaves prefill priority, speculative token count, model
weights, cache formats and kernel implementations unchanged. Prefill tail
chunk boundaries do change. Neither identical model output nor a throughput
gain follows from metadata tests alone.

Validation on the source based on `f9ac0209a2edb6d54cd40a27913fde5dd833f944`:

- Nine TP9 metadata cases pass; the base fails eight and passes one control.
- 178 prefix-cache, allocation, copy-on-write and chunk-boundary cases pass.
- Pre-commit checks, including mypy, pass.
- The unmodified serving image reproduces 3,301 and 4,262 computed tokens on
  identical 17,125- and 18,086-token replays, with 1.96 and 2.40-second TTFT.
  These are synthetic-token cache probes, not model-quality evaluations.

GPU release checks must cover exact and adjacent hash/block boundaries,
recurrent and convolution state bytes, greedy token IDs, stochastic coherence,
shared-prefix concurrency, cancellation and external-cache restoration.
Cold prefill throughput and decode step rate, acceptance and inter-token
latency require matched comparisons before deployment. Use an isolated cache
namespace until cross-configuration restore compatibility is established.
