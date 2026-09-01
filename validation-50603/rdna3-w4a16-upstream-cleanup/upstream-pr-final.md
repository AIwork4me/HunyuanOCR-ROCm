# PR draft — final

Title: `[ROCm][RDNA3] Fix W4A16 split-K nondeterminism`
Target: vllm-project/vllm, branch `fix/rdna3-w4a16-determinism` from main (base 07ea9350)
Status: draft (not opened)

## Problem

Identical inputs to `RDNA3W4A16LinearKernel` can produce different results on gfx11 because the scalar and WMMA split-K paths atomically accumulate narrowed bf16/fp16 partials in execution-dependent order. On W7900D (gfx1100) this shows up as flaky greedy decoding: identical prompts and seeds yield two output families across repeated generations.

## Root cause

Multiple K-split blocks write the same output element. Each FP32 partial is narrowed to the output dtype before accumulation, and the CAS completion order varies run to run; the low-precision intermediate rounding makes the final result order-dependent.

## Fix

Keep the existing compute kernels and dispatch. Replace only the split-K epilogue on both paths:

```text
FP32 per-split partial
→ fixed ascending-z FP32 reduction
→ one final output cast
```

`k_split == 1` keeps the original direct store. Scratch comes from the PyTorch caching allocator per call (row-tiled so the bound is independent of M: ≤ z·64·N·4 bytes scalar, ≤ k_split·512·N·4 bytes WMMA), so CUDA-graph capture and multi-stream use are safe with no persistent state.

## Determinism

- Scalar and WMMA paths: 1 distinct output / 100 repeated calls at every tested shape (bf16 and fp16), including real Muse q_proj weights (K=6656, N=4096) at M ∈ {1, 8, 16, 64, 128, 512}.
- Per-call interception over a full generation (16,640 GEMM calls): zero same-input-different-output events.
- E2E greedy decoding: bit-identical token streams across 8 repeats for Muse (graphs, ctx 512/8192) and gemma-3 (ctx 512/8192), and Muse eager at ctx 512.
- For the captured workload, ascending, descending, and fixed-shuffled FP32 reductions were bit-identical.
- The empirical onset in the fixed-input probe occurred between 2 and 4 concurrent writers.

## Accuracy

Compared against an fp32 dequant reference, max abs error improves from 0.028 (atomic path) to 0.0061 at M=1; cosine similarity ≥ 0.9999976 across tested shapes. The single final cast removes the intermediate low-precision roundings entirely.

## Performance

Same-dispatch same-path baselines on gfx1100: M=1 +4.6% (45.1 vs 43.1 µs, decode-relevant), M=8 +4.7%, while the WMMA range gets faster because the CAS epilogue it sheds was the dominant cost there: M=16 −17.2% (81.0 vs 97.8 µs), M=64 −5.8% (172.4 vs 183.0 µs), M=128 −3.2%, M=512 +3.7% (zero-init + reduce of the row-tiled partials).

## Tests

`tests/kernels/quantization/test_rdna3_w4a16_determinism.py` — bit-repeatability regression tests, parameterized bf16/fp16 across the scalar and WMMA dispatch ranges (8 cases). Fails on unpatched main (4/4 on the scalar subset against 0.25.1); passes 8/8 with this patch.

## Scope

gfx11 / RDNA3 W4A16 only. Compute kernels, tile shapes, and dispatch thresholds are untouched; no routing policy or unrelated changes. Two kernel files plus the new test.
