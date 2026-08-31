# Phase 1 Status — PASS

| Gate item | Result | Evidence |
|---|---|---|
| upstream repo cloned | PASS | `upstream/dual-radeon-vllm`, remote = https://github.com/cadamcat/dual-radeon-vllm |
| exact commit recorded | PASS | `782c7d431ab8e821242f2a717bf5d03b0be3301d` (artifacts/upstream-clone-info.txt, clone time 2026-08-30T22:53:59+00:00) |
| original harness preserved | PASS | `git status --short` empty at freeze; tree untouched since |
| all measurement constants identified | PASS | artifacts/cadamcat-harness-analysis.md (constants table + env table + model provenance resolution) |
| local changes have minimal auditable diff | PASS | artifacts/harness.diff (165 lines, generation semantics untouched; TP 2→1, model paths, evidence JSON, teardown) |

Frozen upstream facts: vLLM `v0.23.1.dev1+g9ddef7117.d20260715` on their side; harness = `nondet_eager.py` one engine per invocation, both depths per engine, warm-up max_tokens=8 then 8×(temperature=0.0, max_tokens=64, ignore_eos=True) per depth, prompt `[1000 + (i % 20000) for i in range(d)]`, engine kwargs max_model_len=8704 / max_num_seqs=128 / gpu_memory_utilization=0.92 / disable_log_stats / enforce_eager=<arg> / TP=2 (ours: 1).
