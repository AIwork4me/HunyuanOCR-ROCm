# latest-main — revalidation on upstream `1dc464d426` (2026-08-30)

Upstream moved 5 commits past the original base (`b5707bf994`, see the historical directories) to `1dc464d42681d22f38caf1fdc1eb632dc4421c45` ("[Bugfix] Bound cache_salt length to prevent DoS via scheduler CPU exhaustion (#54353)"). None of the 5 commits touch the candidate's production files, so the candidate reapplied cleanly (`git apply --check` clean; uncommitted on the local branch `investigate/hunyuanocr-transformers-regression`, never pushed). Updated 9-file patch: `candidate-1dc464d426.patch` (also mirrored at `../diff/hunyuanocr-vllm-candidate.patch`).

## Contents

| File | What it is |
|---|---|
| `run-baseline-latest-main.log` | STATE B on pristine `1dc464d426`: `ValueError: Expected 4 multimodal RoPE channels, got position_ids with shape (3, 1, 8192)` (exit 1) — root cause 1 reconfirmed |
| `config-probe-latest-main.log` | Live config probe on latest main: `mrope_section=[16,16,16,16]` (4 axes, sum 64 = head_dim//2), `uses_mrope=True`, `uses_xdrope_dim=0`, vestigial `kv_lora_rank=512 / qk_nope_head_dim=128 / qk_rope_head_dim=64` — root causes 1+2 inputs reconfirmed (RC2/RC3 code sites verified pristine via `git show HEAD:…`) |
| `tests-new-before-fix.log` | First pristine run of the two NEW focused tests (before the grid-fixture shape fix) — superseded by `tests-all-before-fix.log` |
| `tests-all-before-fix.log` | Pristine-main run of all 6 focused tests: 3 regression tests FAIL (`assert 3 == 4`, `assert 192 == 128`, `TypeError: … 'video_grid_thw'`), 3 guards PASS |
| `tests-focused-and-mrope-after-fix.log` | Candidate run: 6/6 focused + 11 mrope executor tests + new upstream `test_qwen3_asr_mrope.py` = 17/17 PASS |
| `tests-surrounding-after-fix.log` | Candidate run of kernels `test_mrope.py` + transformers backend tests: 64 failed / 14 passed / 1 skipped / 1 error |
| `tests-surrounding-pristine-control.log` | Pristine control: identical counts; `diff` of FAILED lines = empty → every surrounding failure is pre-existing/environmental (hub downloads on an offline host) |
| `e2e-eager.json` / `run-e2e-eager.log` | W7900 E2E, `enforce_eager`: 4/4 PASS, ×3 deterministic, doc hash `6cd5fba4cdb2f135` |
| `e2e-default-piecewise.json` / `run-e2e-default-piecewise.log` | default mode + `cudagraph_mode="PIECEWISE"`: 4/4 PASS, same doc hash |
| `e2e-v1-runner.json` / `run-e2e-v1-runner.log` | legacy V1 runner (`VLLM_USE_V2_MODEL_RUNNER=0`): 4/4 PASS, same doc hash |
| `candidate-1dc464d426.patch` | the 9-file candidate on this base (510+/9−; ruff clean, `git diff --check` clean) |

The document token hash `6cd5fba4cdb2f135` matches the original-base runs exactly — the candidate's greedy output is byte-identical across both upstream bases and all three execution modes. Transformers 5.15.1, TP=1, model revision `de8f10ad…`, `HF_HUB_OFFLINE=1`. FULL cudagraph capture on ROCm remains unsupported (pre-existing HF-rotary cache-growth vs HIP capture limitation; not re-tested here, not claimed).

Exact commands: see `../REPRODUCE.md`. Duplicate-work search (no overlapping upstream PR found) and the two-PR split recommendation: `../SUMMARY.md` Part 2.
