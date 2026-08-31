# Self-review — first-divergence-logit-forensics — PASS

All numbers in README/summary.json re-verified against the machine-readable evidence immediately before writing (28 pairs, D 5–24, stage-B equal 0/28, margins n=56 min 0/median 0.5/max 2.0, aligned n=35 max 8.8125 median 3.25, offline replay all cpu/gpu stable and matching, microprobe 6/6 records, M=1 first-call shas distinct across 3 processes). Two transcription errors found and fixed during this pass (D-range 5–50→5–24; max-diff 3.3→4.125).

| # | check | verdict | evidence |
|---|---|---|---|
| 1 | baseline nondeterminism reproduced | PASS | 3 fresh uninstrumented engines, 8/8 unique each, earliest D=5 (results/baseline/, baseline-forensics.json) |
| 2 | instrumentation nondeterminism reproduced | PASS | probe1 (logits processor) 8/8; probe2 (+LM-head hook) 8/8; same competing tokens 112740/200007 as baseline |
| 3 | exact first-divergence positions identified | PASS | 28/28 pairs, D range 5–24, in first-divergence-forensics.json |
| 4 | histories before divergence identical | PASS | history_identical_through_Dm1=true for all pairs (recomputed from token_ids); plus cache-hit call counting (520 = 8+8×64) proving identical inputs at aligned steps |
| 5 | raw logits captured | PASS | full fp32 rows archived per generation (probe dirs on box); divergence+aligned rows in logit-snapshots/ as lossless bf16; per-step sha256 in steps.jsonl |
| 6 | processed logits captured | PASS | the probe captures the exact argmax input (post-cast, pre-argmax position in the chain); cast-only chain documented in token-selection-path.md; bf16→fp32 injectivity makes Stage-A equality ⟺ Stage-B equality |
| 7 | selected token captured | PASS | run.json token_ids + steps.jsonl own_cpu_argmax |
| 8 | top-10 preserved | PASS | top10 in forensics JSON; top-64 in steps.jsonl |
| 9 | full divergence-position logits preserved | PASS | snapshots (66/70 rows); full archives on box (documented) |
| 10 | offline CPU argmax performed | PASS | fresh process, 20× CPU + 20× GPU per saved row, all stable, all match engine (offline-argmax-replay.json) |
| 11 | raw vs processed boundary established | PASS | chain is float32 cast only for this config (code-mapped in Phase 2, corroborated at runtime by argmax-consistency at all 512 steps) |
| 12 | margins computed from raw values | PASS | full-precision floats from saved tensors; no rounding in machine-readable output |
| 13 | no conclusions exceed evidence | PASS | classification B; W4A16 stated as "localized to" via isolated fixed-input microprobe with disclosed packing caveat; mechanism inside the kernel explicitly open |
| 14 | W4A16 not blamed without isolation | PASS | isolation performed (Phase 10) BEFORE any attribution; real weights, real shapes, 3 processes |
| 15 | all numbers regenerate from raw files | PASS | analyze_forensics.py / offline_replay.py / lmhead_analysis.py / w4a16_probe.py shipped; SHA256SUMS over the whole directory |

Blemishes kept visible:
- First instrumented run used max_model_len=1024 (vs validated 8704); detected in self-check, rerun with exact kwargs; the 1024 run kept as supplementary evidence (logs/probe-eng1-maxlen1024-supplementary.log references).
- Two failed hook iterations (eager-import stdout pollution breaking cpuinfo's JSON subprocess; bf16-vs-numpy hashing TypeError) — diagnosed, fixed, failure logs kept.
- The initial microprobe JSONL contained one non-JSON log line from a tee; filtered; checksums regenerated.

Verdict: all critical items PASS; fit to commit.
