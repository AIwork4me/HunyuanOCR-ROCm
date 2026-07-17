# Changelog

All notable changes to this project are documented here. The format is based on
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [Unreleased]

### Verified — reproducibility
- **Model weights cross-checked against official HuggingFace.** All four
  benchmark artifacts (`HunyuanOCR-bf16.gguf`, `mmproj-HunyuanOCR-bf16.gguf`,
  `model.safetensors`, `config.json`) are byte-identical to `tencent/HunyuanOCR`
  and `ggml-org/HunyuanOCR-GGUF` on HF (queried via the `hf-mirror.com` mirror,
  since `huggingface.co` is unreachable from the bench env). `reproducibility.lock.yaml`
  `current_remote_artifact` now records the verified HF revisions + LFS oids.

### Changed — scientific positioning & correctness
- **Evaluation-backed, not precision-aligned.** Removed the unverified
  "precision-aligned" claim repo-wide; the README now separates the 148-page
  canary (Table A) from the 1651-page full set (Table B), and no longer places
  the official (TensorRT, full-set) 94.10 in the canary column. See
  `docs/benchmark-methodology.md`.
- Fixed the double-score bug: the scorer config is now written to a private temp
  dir, so a prediction directory can be scored twice under strict validation.
- `llama.cpp` runs now record `backend: llamacpp` in the manifest (was hardcoded
  `vllm`); added `--backend-name`.
- Run manifest split into `run_counts` + `final_state` with enforced conservation
  laws (`attempted == succeeded+failed`, `expected == attempted+skipped`,
  `expected == complete+failed+pending`); `schema_version`, ISO-8601 timestamp,
  repo-root git resolution, platform info.
- Exclusive prediction-directory writer lock (`.run.lock`) prevents two writers
  on the same directory.

### Added — reliability, tooling, reproducibility
- Circuit-breaking OpenAI-compatible endpoint pool (`endpoint_pool.py`): probes
  `/v1/models`, opens circuits after consecutive failures, half-open probes,
  fast-fails when no endpoint is healthy.
- Pre-flight input validation + sharding fix (`preflight.py`): fails before model
  load on bad GT/images/ports/GPUs; `shard()` now returns exactly `n` buckets so
  GPUs > pages no longer raises `IndexError`.
- Client-side image cap no longer pollutes the dataset dir (content-hash cache,
  read-only datasets supported).
- Unified `hunyuan-ocr` CLI: `doctor | validate | manifest verify | canary
  materialize | predict | score`.
- Canary materializer (`canary.py`): rebuilds the 148-page canary **byte-identically**
  from the full GT + manifest; the manifest now stores pages in file order.
- Atomic writes now fsync the parent directory.
- Split dependencies: core is GPU-free (pillow/pyyaml/tqdm/requests); `client`,
  `transformers`, `eval`, `dev` extras; plain-PyPI torch is never a dependency.

### Added — project / community
- CI (`.github/workflows/ci.yml`) runs lint/test/build with **no torch** on
  Python 3.12; repo-integrity check (`scripts/check_repo.py`); manual-only ROCm
  smoke workflow.
- CONTRIBUTING, SECURITY, SUPPORT, CODE_OF_CONDUCT, issue templates, PR template,
  CODEOWNERS, CITATION.cff.
- REUSE-compliant licensing (`reuse lint` passes); upstream-derived files now use
  `LicenseRef-Tencent-Hunyuan-Community-License` (was `NOASSERTION`).
- `reproducibility.lock.yaml` records verified model/GGUF SHA256, OmniDocBench
  repo URL + commit, eval-config + manifest hashes, the Overall metric formula.

## [0.1.0] — 2026-07-16

- Initial three-backend evaluation on AMD gfx1100 (RDNA3): vLLM canary 94.81,
  transformers canary 94.11, llama.cpp canary 93.33, llama.cpp full 1651 = 92.09.
- Filed ROCm/ROCm#6416 (>14k ViT NaN) and Tencent-Hunyuan/HunyuanOCR#114.
