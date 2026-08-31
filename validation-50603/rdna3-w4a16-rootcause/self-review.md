# Self-review — rdna3-w4a16-rootcause — PASS

Numbers re-verified against the machine-readable evidence immediately before writing (e2e JSONs recounted; microprobe/partials/zsweep/correctness/perf JSONLs re-parsed; routing lines re-grepped from logs).

| check | verdict | evidence |
|---|---|---|
| kernel ON reproduces? | PASS | 6/8, 8/8, 8/8 @512 (kernel-on-off/on-eng*.json) + post-work re-verify 8/8 (environment/on-verify-restored-env-eng9.json) |
| kernel OFF removes the effect? | PASS | 1/8 ×3 both depths (kernel-on-off/eng*.json) |
| fallback routing verified? | PASS | runtime log line `Using TritonW4A16LinearKernel` in all 3 OFF engines (logs/e2e-off-eng*.log) |
| fixed input and weights truly identical? | PASS | same seeded x (torch.manual_seed 20260831) and same in-memory layer across the atomic/deterministic calls in one process; cross-process identical construction (deterministic sha equal across 3 processes proves weight/input bit-equality) |
| split partials themselves deterministic? | PASS | 50 calls → 1 distinct (phase4-partials.jsonl) |
| only reduction order changing? | PASS | partials capture is the unmodified compute path (same kernel body, epilogue branch on a null/non-null pointer); R1/R2/R3 differ ONLY in z-order |
| deterministic prototype bit-repeatable? | PASS | 1/200 ×3 processes, same sha cross-process |
| E2E greedy fixed? | PASS | 1/8 ×3 engines, both depths, no divergence position anywhere |
| numerical correctness preserved? | PASS | 4× closer to fp32 dequant reference than the atomic path (max 0.0061 vs 0.0261) |
| performance cost disclosed? | PASS | +6.6% @M=1, 3.96× @M=512, E2E +49% vs atomic / −9% vs fallback (perf-microbench.jsonl + driver timestamps) |
| regression test fails before / passes after? | PASS | atomic test XFAILED (fails repeatability on gfx1100), deterministic test PASSED (pytest run in log) |
| root-cause claim no stronger than evidence? | PASS | "caused by the epilogue" is backed by: same-kernel ON/OFF A/B, deterministic-partial + order-invariance isolation, Z-threshold on the unmodified op, and the prototype flipping the outcome; mechanism INSIDE the CAS/bf16 path is characterized, not claimed at instruction granularity |
| all figures regenerate from raw data? | PASS | JSON/JSONL evidence + scripts/ shipped |
| unrelated code changes? | PASS | diff touches only q_gemm_rdna3.cu, ops.h, torch_bindings.cpp, rdna3_w4a16.py + the new test |
| debug patches separated? | PASS | diagnostic.patch (partials op + M-logger) vs deterministic-prototype.patch (everything, as built) vs upstream-candidate.patch (clean: deterministic op + dispatch + test, 0 refs to the diagnostic op/logger) |

Known blemishes, disclosed rather than hidden:
- The first prototype E2E attempt was M≤8-gated and stayed nondeterministic (6/8, 5/8) — kept as evidence of the serving-path discovery (prefill-tail recomputation at M>8); fixed by 64-row tiling + routing all M.
- The un-tiled first build OOM'd engine profiling at 31.69 GiB (log kept) — fixed by tiling.
- The hardlink-venv pip-shebang issue briefly repointed the pristine env's editable metadata to the dev tree; restored and re-verified (8/8); all arm data shown was collected before the first modified build (timestamps in logs).
- diagnostic.patch is the instrumentation delta for the record; it was not built standalone in that exact form (the built state is deterministic-prototype.patch — everything ran from it).
- M≥16 WMMA epilogue determinism is unassessed; the prototype bypasses WMMA (correct but slower at large M).

Verdict: all critical items PASS — evidence fit for upstream discussion.
