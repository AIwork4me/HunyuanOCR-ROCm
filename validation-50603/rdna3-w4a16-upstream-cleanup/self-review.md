# Phase 12 — final patch review (skeptical maintainer pass)

| # | Item | Verdict | Evidence |
|---|---|---|---|
| 1 | Root cause independently proven | PASS | prior stages: `../rdna3-w4a16-rootcause/` (split-K CAS atomic order-dependence, onset between 2 and 4 concurrent writers) |
| 2 | Compute kernels unchanged | PASS | diff touches only epilogues/launchers in the two `.cu` files; mainloops byte-identical to upstream |
| 3 | Existing upstream dispatch preserved | PASS | `q_gemm_rdna3` dispatch condition byte-identical to upstream (`bf16 M>=16 / fp16 M>=64 → WMMA`); verified by regenerating the diff and comparing to the stored patch |
| 4 | No model-specific threshold | PASS | no M/N/K constants beyond upstream's own tile constants |
| 5 | No legacy incorrect-path env switch | PASS | `VLLM_RDNA3_W4A16_LEGACY_ATOMIC` deleted; 0 references in patch |
| 6 | No unsafe thread-local scratch cache | PASS | 0 `thread_local` in patch; scratch via PyTorch caching allocator per call, capture-safe (graphs E2E 1/8) |
| 7 | Deterministic scalar | PASS | 1/100 at all tested M, bf16+fp16; 0/16640 kernel-level divergences under interception |
| 8 | Deterministic WMMA | PASS | all 7 kernels + 4 cascading launchers + head have the partials epilogue; 1/100 at M=16..512 |
| 9 | bf16 tested | PASS | micro + E2E |
| 10 | fp16 tested | PASS | micro (scalar z4/z26, WMMA 64/128/512) |
| 11 | Graph mode tested | PASS | Muse graphs ctx512/8192 = 1/8 |
| 12 | Numerical accuracy preserved/improved | PASS | max_abs_err 0.0061 vs 0.028 (M=1); cosine ≥ 0.9999976 |
| 13 | M=1 overhead acceptable | PASS | +4.6% (45.1 vs 43.1 µs) |
| 14 | Large-M overhead acceptable | PASS | +3.7% @ M=512; −5.8% @ M=64; −17.2% @ M=16 (net win on WMMA range) |
| 15 | Bounded scratch | PASS | scalar: z·64·N·4 (single tile, M-independent); WMMA: k_split·512·N·4 ≤ ~144 MiB row-tiled, M-independent |
| 16 | Regression tests fail before / pass after | PASS | 4/4 FAIL on 0.25.1 → 8/8 PASS (parameterized) |
| 17 | Current main clean build | PASS | fresh worktree at 07ea9350 + patch builds (Phase 7); smoke below |
| 18 | No forensic/debug code in patch | PASS | M-logger lived only in the python worktree file, never in the patch; patch contains 3 files, reviewed line-by-line |

Residual (out of scope, documented): Muse eager ctx8192 = 2/8 traced to upstream ROCm custom paged attention KV last-block stale-slot sensitivity — same build with TRITON_ATTN gives 1/8; W4A16 GEMMs proven bit-deterministic (see `muse-eager-8192-attention-rootcause.md`).

All critical items PASS. Ready for upstream submission branch.
