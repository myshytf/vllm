# Kimi-K3 recurrent prefix replay with a separate draft cache

Status: **implemented and qualified for the replay conditions below**.

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

The draft's cache-restored-prefix alignment check considers only cache groups
used by its attention layers. A 15,360-token prefix is aligned to a 1,536-token
draft page even though it is not aligned to a 4,608-token target recurrent
block. Target-only page widths must not disable drafting. A genuinely partial
draft page still selects the existing fail-closed path.

The implementation leaves prefill priority, speculative token count, model
weights, cache formats and kernel implementations unchanged. Prefill tail
chunk boundaries do change. Neither identical model output nor a throughput
gain follows from metadata tests alone.

Validation on the source based on `f9ac0209a2edb6d54cd40a27913fde5dd833f944`:

- Nine TP9 metadata cases pass; the base fails eight and passes one control.
- 178 prefix-cache, allocation, copy-on-write and chunk-boundary cases pass.
- The draft prefix-masking CUDA suite passes ten cases, including rejection of
  genuinely unaligned draft prefixes. The target-width regression fails with
  the all-group predicate. Related context-graph, lifetime, sliding-window and
  lookahead checks pass in the CUDA test environment (32 cases total).
- Pre-commit checks, including mypy, pass.
- The unmodified serving image reproduces 3,301 and 4,262 computed tokens on
  identical 17,125- and 18,086-token replays, with 1.96 and 2.40-second TTFT.
  These are synthetic-token cache probes, not model-quality evaluations.

With implementation revision `6054d5e56b1e`, repeated warm requests reuse
15,360 tokens. TTFT is 1.043 and 1.497 seconds at the two lengths; the
16-token decode tails are approximately 0.281 seconds, matching the base samples.
Draft/accepted-token counts remain 8/7 and 8/8. Output token IDs match the
base. Curated evidence and the deterministic input rule are in
`benchmarks/results/kimi_k3_recurrent_prefix_replay.json`; the executable probe
is `benchmarks/benchmark_kimi_prefix_replay.py`.

Eleven hash/block/verify-width boundary lengths pass cold/warm token-ID
checks. Document recall, greedy replay, sampled output, assistant/tool
continuations and four concurrent branches pass. A local-cache reset followed
by a document request restores 9,216 tokens through LMCache and answers the
document correctly. Separate sustained decode screens at concurrency one
through 128 Ki requested context and concurrency four at 16 Ki finish without
errors or loops. These screens are absolute measurements, not a matched
historical-hardware throughput comparison.

The replay probes cover short deterministic output and document canaries.
They do not establish universal stochastic acceptance parity or a complete
million-token workload qualification. Cache formats and native kernel bytes
are unchanged. Reuse of an existing external namespace requires a retained
entry restoration check against an independently cold request. The serving
namespace passes that check: 9,216 externally restored tokens produce the same
96-token response and log probabilities as the cold control (maximum difference zero).
