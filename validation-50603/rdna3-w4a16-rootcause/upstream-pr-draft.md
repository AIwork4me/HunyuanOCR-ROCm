# Upstream PR draft (not opened)

## Title

`[ROCm][RDNA3] Make W4A16 split-K reduction deterministic (FP32 scratch + fixed-order reduce)`

## Problem

`gptq_gemm_rdna3` (RDNA3W4A16LinearKernel, gfx1100) splits K across `grid.z = K/256` blocks. Each block computes an FP32 partial, **casts it to bf16/fp16**, and accumulates it directly into the pre-zeroed T-typed output via a CAS-loop packed atomic (`atomic_add_pk4_{bf16,f16}`, added because gfx11 lacks a native packed fp16/bf16 atomic add). With Z concurrent writers per output element, the low-precision reduction order — and therefore the rounded result — varies per call.

Measured on a Radeon PRO W7900D (gfx1100), real Muse-Glimmer-30B-INT4 q_proj (K=6656, N=4096, group 128, bf16):

- fixed input + fixed weights, M=1: **200/200 calls produce 200 distinct outputs** (three processes; even the first call's output differs process to process);
- Z-sweep on the unmodified op: Z=1,2 repeatable; **Z≥4 nondeterministic** (100/100 distinct);
- end-to-end this is the root cause of greedy-decoding nondeterminism (vllm#50603 class): 8 identical greedy generations in one engine give 6–8 distinct outputs with this kernel, 1/8 with the kernel disabled, and 1/8 with the fix.

## Root cause (proven, not inferred)

- per-split FP32 partials are bit-repeatable (diagnostic op);
- ascending/descending/shuffled FP32 reductions of those partials are bitwise identical — FP32 order is not the issue;
- the variance enters exclusively at the cast-then-CAS-add chain: 26 (for K=6656) unordered bf16 roundings per output element;
- replacing the epilogue removes the effect at every level (isolated kernel, logits, greedy generation).

## Fix

`gptq_gemm_rdna3_deterministic`:
1. the same split-K compute writes FP32 partials to a scratch buffer `[Z, tile_m, N]` (no atomics, no casts), row-tiled at 64 rows so scratch stays ≤ ~118 MiB for typical RDNA3 W4A16 shapes at any M;
2. a second kernel reduces in fixed ascending-z order in FP32 and rounds to the output dtype exactly once.

Dispatch is env-gated in `RDNA3W4A16LinearKernel.apply_weights` (`VLLM_RDNA3_W4A16_DETERMINISTIC=1`) so the rollout is opt-in while CI validates both paths; default-off keeps today's behavior and perf unchanged.

## Validation

- **Repeatability**: 200 calls ×3 processes → 1 distinct output each, identical sha across processes (vs 200/200 distinct on the legacy path).
- **End-to-end**: Muse/TP=1/eager greedy ×8, 3 engines: 6/8, 8/8, 8/8 (legacy) → **1/8, 1/8, 1/8** (fixed); ctx=8192 likewise. Routing logged (`RDNA3W4A16LinearKernel` + deterministic line).
- **Correctness vs fp32 dequant reference**: max abs err 0.0061 (fixed) vs 0.0261 (legacy); cosine 0.999999 vs 0.999981 — the fix is ~4× more accurate because split partials accumulate in FP32 with a single final rounding.
- **Performance** (real q_proj, CUDA events, median of 200): M=1 43.0→45.9 µs (**+6.6%**); M=512 0.53→2.11 ms (large M keeps the scalar path; a follow-up can wire a validated WMMA epilogue instead). E2E eager harness: +49% vs legacy atomic, −9% vs the Triton fallback (`VLLM_DISABLED_KERNELS=...`) which is the only currently deterministic alternative.
- **Regression test** `tests/kernels/quantization/test_rdna3_w4a16_determinism.py`: K=1024 (Z=4, above the measured determinism onset), 20 repeats — deterministic path asserts `torch.equal` (passes); legacy epilogue marked xfail documenting the known behavior.

## Tradeoffs

- Decode-shape cost is small (+6.6% per GEMM) but nonzero; the extra pass costs more at prefill shapes (~4× on the scalar path) — acceptable for an opt-in determinism mode, and still faster end-to-end than falling back to Triton.
- Scratch memory ≤ ~118 MiB transient per layer call (tiled), vs zero for the atomic epilogue.
- `VLLM_DISABLED_KERNELS=RDNA3W4A16LinearKernel` already provides a deterministic escape hatch today, at ~63% end-to-end cost on this workload.
- Alternative worth benchmarking before making the fix default: a single-split (Z=1) variant for M=1 decode (no scratch, no second pass; CU-occupancy question at K/256 blocks) — the Z=1 stability measurement above suggests it is deterministic by construction.
