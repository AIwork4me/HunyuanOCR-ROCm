# Self-review — production fix — PASS (upstream-ready)

Reviewed as a skeptical vLLM ROCm maintainer; every number below re-derived from the JSONL/logs in this tree before writing.

| check | verdict | evidence |
|---|---|---|
| root cause still reproduced before fix | PASS | legacy-arm control engine in the production build: 8/8 @512, 3/8 @8192 (logs/e2e-legacy-control); fail-before tests 4/4 FAIL on pristine 0.25.1 |
| scalar deterministic after fix | PASS | 1/100 distinct at M ∈ {1,2,3,4,5,7,8,9,15} + M=1/8 in matrix; E2E 1/8 ×3 engines |
| WMMA audited | PASS | Phase 1 audit: K_SPLIT=1 control stable, K_SPLIT=4 nondeterministic M=16–128 (100/100 at M=16/64), M=512 coincidence-stable (2 procs, same sha) — treated as non-guaranteed |
| WMMA deterministic after fix | PASS | 1/100 at M=16,64,128,512 through the public op; E2E graphs arm green |
| no low-precision unordered split accumulation remains in affected paths | PASS | deterministic dispatch covers scalar (all M<65, any N) and WMMA (M≥65, N≥64; other shapes route to scalar); legacy CAS path only reachable via explicit opt-out env |
| M=1 performance acceptable | PASS | +7.8% (target ≤10%; stretch 5% not met — disclosed) |
| large-M performance acceptable | PASS | M=64 +2.3%, M=128 −1.2%, M=512 +1.1% (target ≤10–15%); E2E +14.8% incl. prefill shapes |
| memory bounded | PASS | scratch = z·tile·N·4 (scalar ≤26·64·N·4, WMMA ≤4·512·N·4 ≈ 144 MiB worst), thread-cached, capture-safe; no multi-GB allocations at any M (profiling M=8192 runs tiled) |
| accuracy preserved/improved | PASS | max abs err and cosine better than legacy at all six M; all outputs finite |
| Muse fixed | PASS | 1/8 ×3 eager engines, both depths; 1/1 graphs engine |
| secondary model fixed | PASS | gemma-3-27b-it-w4a16 1/8 both depths |
| graph mode works | PASS | after fixing a real capture-time fault in the first scratch-cache design (baked pointer freed on resize); capture now bypasses the cache; graphs arm 1/8 |
| tests fail before / pass after | PASS | 4/4 FAIL on pristine vLLM 0.25.1; 4/4 PASS on production build; 4/4 PASS on patched current main |
| current main validated | PASS | patch applies cleanly to `dafbef15a` (touched files byte-identical); build OK (0.28.1rc1.dev159+gdafbef15a, torch 2.12.0+rocm7.14.0); tests 4/4; Muse E2E 1/8 both depths; M=1 and M=512 benched (det within noise/legacy band) |
| patch contains no forensic clutter | PASS | upstream-production-fix.patch = 2 kernel files + 1 test file, 808 lines; zero refs to zexp / M-logger / FORCE_KSPLIT1; only env var is the sanctioned LEGACY_ATOMIC opt-out (3 refs) |
| PR claim matches evidence | PASS | draft numbers spot-checked against perf-correct-det3/prodLeg JSONLs and E2E logs |

Blemishes disclosed:
- M=1 stretch target (+5%) not reached (+7.8%); primary target (≤10%) met.
- First graphs arm faulted (thread-local scratch under capture) — diagnosed, fixed with a capture-status guard, rerun green; the fault and fix are part of the engineering record and motivated a code comment.
- Single-writer and Z-reduction experiments live in the development worktree only (`scalar/`, `microbench/`); they are excluded from the upstream patch.
- E2E wall comparison uses the earlier session's legacy/prototype/Triton numbers (same box, same harness, same day) — noted in performance/production-performance.md.
- TP=2 not runnable (single GPU installed); documented in final-report/README.md with the topology-agnostic argument.

Verdict: **READY** (with the M=1 stretch miss and TP=2 follow-up disclosed).
