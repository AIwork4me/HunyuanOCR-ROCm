# Changelog

All notable changes to this project are documented here. The format is based on
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [Unreleased]

## [0.1.2] — 2026-07-18

### Added — GPU-CI bridge (real gfx1100 smoke from GitHub)
- A box-side poller (`src/hunyuan_ocr/ci/`) that runs the real 1-page gfx1100
  smoke for `gpu-smoke (gfx1100)` commit statuses created by the new
  `gpu-smoke.yml` (`workflow_dispatch`), and reports success/failure as a commit
  status. Uses **commit statuses** (not check-runs) because a user OAuth token
  can read AND write statuses, whereas check-run writes require a GitHub App.
  Includes a flock single-run lock, SHA-idempotency, a 30-min stale-sweep (no
  silent hangs), a resilient loop, and a trusted-harness + explicit-ref security
  model. CPU-tested (no network/GPU); live-proven on gfx1100 (~61s end-to-end,
  21s smoke).
- `scripts/make_smoke_input.py` for a deterministic 1-page smoke input.
- `docs/ci/gpu-ci-bridge.md` (method, measured data, anruicloud requirements).
- README hero image.

### Notes
- The GPU-CI bridge is an **MVP**: the poller is `nohup` (no systemd on the
  Radeon Cloud Docker box), poll-only (no inbound from GitHub), and status-based
  (≤140-char description, not rich check-run output). Production hardening
  (persistence / inbound webhook / GitHub App for check-runs) is documented as
  the anruicloud-backed next phase. No benchmark changes.

## [0.1.1] — 2026-07-18

A reliability, first-use, and credibility pass. **No benchmark numbers changed**
— the four formal results (vLLM canary 94.81, transformers canary 94.11,
llama.cpp canary 93.33, llama.cpp full 1651 = 92.09) are untouched.

### Fixed — first-use path & doc drift (P0)
- **README Quick Start rewritten** to not depend on the user's current directory:
  explicit `HUNYUAN_ROCM_DIR` / `LLAMA_DIR` / `GGUF_DIR` / `DATA_DIR` variables,
  a Terminal A (server) / Terminal B (predict → validate → score) split, and a
  fixed invalid line-continuation (`\` followed by a comment).
- **Model download** uses the current `hf download` command, backed by a new
  `download` extra (`huggingface_hub`); install is `pip install -e ".[client,download]"`.
- **Single source of truth:** README no longer claims HF revision / GGUF LFS oid
  are `not_recorded` — `reproducibility.lock.yaml` now records them
  (cross-checked byte-for-byte against the official repos) and is the canonical
  source, not duplicated in prose.
- **Canonical canary naming:** `OmniDocBench_canary_148.json` (materialized from
  the full GT via `hunyuan-ocr canary materialize`); the legacy
  `OmniDocBench_150.json` is retired from user docs / Makefile (same 148 pages,
  same SHA256).
- **Makefile** redesigned: repo-relative defaults (no `/root` or `/workspace`),
  required-`DATA_DIR`/`MODEL_DIR` gates with clear errors, canonical canary
  naming, and targets `install-dev | check | test | lint | build | doctor |
  canary-materialize | eval-canary-transformers | score-canary`.
- **Strengthened `scripts/check_repo.py`** (structurally, not via brittle string
  match): stale `not_recorded` claims, forbidden `canary-150`/oracle-150 tokens,
  canonical `canary_148`, the four formal results matching the lock, no positive
  `precision-aligned AMD ROCm port` claim, README-referenced scripts exist, no
  invalid bash line-continuations, and metric-formula consistency across README /
  methodology / lock.

### Fixed — reliability (P1)
- **EndpointPool true single-probe half-open:** a half-open endpoint can only be
  acquired by one thread at a time (`half_open_in_flight`); concurrent acquires
  raise `AllProbesInFlight` instead of double-dispatching. Proven by a 10-thread
  concurrency test.
- **Crash-safe run manifest:** an unexpected worker exception,
  `KeyboardInterrupt`, endpoint-pool fatal, or executor error is captured and the
  manifest is **always** written with a terminal status (`ok` / `failed` /
  `crashed` / `interrupted`) and a redacted crash record. The predict orchestration
  moved to `src/hunyuan_ocr/driver.py` (the script is now a thin wrapper).
- **Manifest schema v2** (`validate_manifest`): never tracebacks on corrupt/missing
  input; validates required fields, non-negative-integer counts (booleans
  rejected), non-empty backend/model, parseable timestamp, `status=="ok"` ⇒
  failed=0 & pending=0, and conservation including a new `run_counts.interrupted`.
  `extra` is namespaced under `extensions` and may not collide with a reserved
  core field. v1 manifests still validate on read.
- **RunLock stale-tail:** acquiring over a longer stale lock JSON now
  `ftruncate`s, so no old tail survives; the "held by another writer" message no
  longer invites deleting a still-held lock.
- **doctor `--strict --backend {llamacpp,transformers,vllm} --json`:** a real,
  CI-friendly environment gate (non-zero on missing critical prereqs); plain mode
  stays advisory (exit 0). Scorer venv/repo now overridable via
  `OMNIDOCBENCH_VENV` / `OMNIDOCBENCH_REPO`.

### Changed — CLI & CI
- **Self-contained CLI:** `hunyuan-ocr predict` (llamacpp/vllm/openai) and `score`
  run from a wheel install via `hunyuan_ocr.driver` / `scoring.score_directory`
  (no `scripts/` checkout required); `predict --backend transformers` still needs
  the checkout + ROCm torch (documented).
- **ROCm workflow renamed `rocm-runner-preflight`** with a real
  `scripts/rocm_smoke.sh` (start server, `/v1/models` wait, one-page predict,
  validate, manifest verify, artifact upload, fail-fast) gated by
  `doctor --strict --backend llamacpp --json` — no more fake echo "smoke".
- **CI:** Python 3.11 / 3.12 / 3.13 matrix (CPU-only, no torch), `reuse lint`,
  repo-integrity check, coverage report (`--cov-fail-under=65`), and a
  built-wheel CLI smoke; Dependabot configured for actions + pip.

### Verified — reproducibility (carried from the unreleased backlog)
- **Model weights cross-checked against official HuggingFace.** All four
  benchmark artifacts (`HunyuanOCR-bf16.gguf`, `mmproj-HunyuanOCR-bf16.gguf`,
  `model.safetensors`, `config.json`) are byte-identical to `tencent/HunyuanOCR`
  and `ggml-org/HunyuanOCR-GGUF` on HF (queried via the `hf-mirror.com` mirror,
  since `huggingface.co` is unreachable from the bench env). `reproducibility.lock.yaml`
  `current_remote_artifact` now records the verified HF revisions + LFS oids.

### Changed — scientific positioning & correctness (carried from the unreleased backlog)
- **Evaluation-backed, not precision-aligned.** Removed the unverified
  "precision-aligned" claim repo-wide; the README separates the 148-page canary
  (Table A) from the 1651-page full set (Table B), and does not place the
  official (TensorRT, full-set) 94.10 in the canary column.
- Fixed the double-score bug: the scorer config is written to a private temp dir.
- `llama.cpp` runs record `backend: llamacpp` in the manifest (was hardcoded
  `vllm`); added `--backend-name`.
- Run manifest split into `run_counts` + `final_state` with enforced conservation
  laws; `schema_version`, ISO-8601 timestamp, repo-root git resolution, platform info.
- Exclusive prediction-directory writer lock (`.run.lock`) prevents two writers.

### Added — tooling & community (carried from the unreleased backlog)
- Circuit-breaking OpenAI-compatible endpoint pool; pre-flight input validation
  + sharding fix; unified `hunyuan-ocr` CLI; canary materializer; atomic writes
  fsync the parent dir; split GPU-free dependencies with `client` / `download` /
  `transformers` / `dev` extras.
- CI, CONTRIBUTING, SECURITY, SUPPORT, CODE_OF_CONDUCT, issue/PR templates,
  CODEOWNERS, CITATION.cff; REUSE-compliant licensing; `reproducibility.lock.yaml`
  recording verified model/GGUF SHA256, OmniDocBench commit, eval-config + manifest
  hashes, and the Overall metric formula.

## [0.1.0] — 2026-07-16

- Initial three-backend evaluation on AMD gfx1100 (RDNA3): vLLM canary 94.81,
  transformers canary 94.11, llama.cpp canary 93.33, llama.cpp full 1651 = 92.09.
- Filed ROCm/ROCm#6416 (>14k ViT NaN) and Tencent-Hunyuan/HunyuanOCR#114.
