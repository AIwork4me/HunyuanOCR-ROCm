# Upstream PR draft (not opened)

## Title

`[ROCm][RDNA3] Fix W4A16 split-K nondeterminism`

## Problem

`RDNA3W4A16LinearKernel` can produce different results for identical inputs on gfx11. Both the scalar path (`gptq_gemm_rdna3`, decode M<16) and the WMMA path (M≥16) split K across concurrent blocks. gfx11 has no native packed fp16/bf16 atomic add, so each split block's FP32 partial is narrowed to the output dtype and accumulated with a CAS loop directly into the output. Because float addition is non-associative and every intermediate add rounds at bf16/fp16 precision, the final value depends on the order in which split blocks complete. Measured on gfx1100 with real W4A16 weights (K=6656, N=4096): repeated fixed-input calls give a different output on nearly every call at M=1 (200/200 distinct across 3 processes) and at M=16–64 via WMMA; the empirical onset observed in the probe is between 2 and 4 concurrent writers per output element. End to end this breaks greedy-decoding reproducibility for compressed-tensors W4A16 models (vllm#50603 class).

## Root cause

Split-K + multiple concurrent writers per output element + partials narrowed to low precision before accumulation + CAS ordering that varies per execution + repeated low-precision rounding at every intermediate add.

## Fix

Deterministic split-K epilogue for both paths, compute kernels unchanged:
- split blocks store FP32 partials to a scratch buffer `[z, tile_m, N]` (plain stores; exactly one writer per element per z-slice, so ordering is irrelevant);
- a reduce pass sums the z-slices in fixed ascending order in FP32 and rounds to the output dtype exactly once.
- Routing: scalar path serves M<65, WMMA serves M≥65 with N≥64 (measured crossover; below it the scalar path is both faster and deterministic); other large-M shapes fall back to the scalar deterministic path.
- Scratch is row-tiled (64 rows scalar, 512 rows WMMA): `scratch_bytes = z_count * tile_m * N * 4`, bounded independently of M (≤ ~144 MiB at the largest common shapes), cached per thread, resized on demand, and bypassed during CUDA graph capture (captured kernels keep baked-in pointers, so the cache must not be reallocated later).
- `VLLM_RDNA3_W4A16_LEGACY_ATOMIC=1` opts back into the legacy atomic epilogue for performance comparisons only. Default behavior is deterministic.

## Determinism

- Isolated, real weights, fixed input: 100 repeated calls → 1 distinct output at every measured M ∈ {1,2,3,4,5,7,8,9,15,16,32,64,128,512} (legacy: up to 100 distinct).
- End-to-end: Muse greedy ×8 (TP=1, ctx 512 & 8192) → 1 unique sequence in 3/3 eager engines and 1/1 graph-enabled engine; gemma-3-27b-it-w4a16 → 1/8; legacy-arm control engine in the same build reproduces 8/8.

## Accuracy

Better than legacy at every M (max abs error vs fp32 dequant reference): M=1 0.0061 vs 0.028, M=8 0.0070 vs 0.035, M=16 0.0118 vs 0.037, M=64 0.0140 vs 0.055, M=128 0.0173 vs 0.025, M=512 0.0173 vs 0.038; cosine 0.9999976–0.9999986 vs 0.9999788–0.9999942. Expected: split partials accumulate in FP32 with a single final rounding instead of 26 (K=6656) low-precision roundings.

## Performance

Real q_proj (K=6656, N=4096, bf16), median of 200: decode M=1 43.1→46.4 µs (**+7.8%**); M=8 +5.4%; M=16 +1.9%; M=64 +2.3%; M=128 **−1.2%**; M=512 +1.1%. Full-engine eager wall on a both-context Muse harness: +14.8% vs legacy, but 30% faster than the `VLLM_DISABLED_KERNELS` Triton fallback — the only previously deterministic option. Decode cost is the headline number; the larger E2E delta includes prefill-shaped calls where the deterministic WMMA path trades a little throughput for reproducibility.

## End-to-end

Muse-Glimmer-30B-INT4 (primary reproducer) and gemma-3-27b-it-w4a16 (secondary), TP=1, eager and graph-enabled, plus a port-and-run on current main (`dafbef15a`) with the same results. Patch applies cleanly to both 0.25.1-era and current main (the touched files are byte-identical there).

## Tests

`tests/kernels/quantization/test_rdna3_w4a16_determinism.py`: four gfx1100-gated bit-repeatability tests through the public op — scalar split-K (K=1024 ⇒ 4 writers) at M=1 and M=8, WMMA split (K=6656 ⇒ K_SPLIT=4) at M=16 and M=128. All four fail on unpatched vLLM and pass with this patch (verified on both builds).

## Scope

gfx11 / RDNA3 W4A16 (`csrc/rocm/q_gemm_rdna3.cu`, `q_gemm_rdna3_wmma.cu`) only. fp16 activations share the same fixed epilogue. Multi-GPU (TP>1) was not exercised in validation hardware, but the epilogue is per-rank local and topology-agnostic.
