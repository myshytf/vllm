# Kimi packed MLA context transport

Status: implemented; qualified by native decoder, address-range and transport
checks. Full-model serving and throughput qualification is a separate gate.

Kimi-K3 stores each packed MLA token as 512 E4M3 latent bytes, four FP32 scales
(16 bytes), and 64 BF16 RoPE values (128 bytes): 656 bytes in total. Expanding
that record into 576 BF16 values before transport sends 1,152 bytes per token.
`VLLM_K3_DCP_GATHER_PACKED=1` preserves the native record on the prefill wire
and expands it on the receiving GPU, reducing payload bytes by 43.1%.

The option is scoped to the Kimi attention layer with packed `fp8_ds_mla`
cache, latent width 512, RoPE width 64 and the direct DCP publisher. A byte-sized
transport view carries the native record's raw bytes; it does not quantize the
FP32 scales or BF16 RoPE fields. Large windows use the existing copy-engine
publisher, with 528-byte and 128-byte planes. Small windows retain the direct
SM publisher. The gather's request-major row order, topology relay and epoch
protocol are unchanged.

The receiver converts E4M3 values to FP32, multiplies by the original FP32
scales, rounds once to BF16, and copies RoPE bits unchanged. This reproduces
`cp_gather_and_upconvert_fp8_kv_cache`. Decoded planes retain the existing
`[tokens,1,width]` interface and contiguous inner layout. One decoded workspace
per ubatch is consumed on the compute stream. The publication slot is released
after decoding, before projection; later projection/attention reads use the
independent BF16 workspace. CUDA-event ordering protects the packed input until
its last read. Buffers and Triton launches are prepared before request use.

Validation:

- CUDA decoder equivalence covers physical record padding, two request ranges,
  page crossings, ordinary and extreme FP32 scales, arbitrary RoPE bits and
  poisoned-output CUDA graph replays.
- A mapped pinned pool places a live page beyond a 2-GiB byte offset. The GPU
  gather returns that page exactly, checking 64-bit page/row address arithmetic
  without requiring spare device VRAM for a mostly unused pool.
- The native publisher preserves arbitrary record bytes across nine logical
  ranks, mixed relay/direct routes, uneven request lengths and three reused
  slots. BF16 and packed variants pass on the same CUDA implementation.
- A two-GPU comparison across the two physical PEX88096 switches starts at
  the same paged cache and ends with exactly equal BF16 planes. At 3,072 local
  rows, interleaved graph medians are 154.432 microseconds for BF16 transport
  and 104.398 for packed transport including receiver expansion
  (packed/reference 0.676). Raw samples are in
  `benchmarks/results/kimi_packed_kv_transport.json`.

These are component measurements, not a model-wide speed claim. Precision,
quality, decode speed, cache compatibility and end-to-end prefill must be
validated on the frozen deployment before promotion.
