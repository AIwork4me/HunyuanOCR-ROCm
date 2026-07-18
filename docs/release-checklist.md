# Release checklist (v0.1.x)

Run every item before tagging. The goal: a release whose numbers are unchanged,
whose install path works from a clean checkout, and whose license/CI posture is
verifiable — not "theoretically passing".

## 1. Version consistency (single source)

- [ ] `pyproject.toml` `version`
- [ ] `src/hunyuan_ocr/__init__.py` `__version__`
- [ ] `CHANGELOG.md` section heading matches (and `[Unreleased]` is empty)
- [ ] `CITATION.cff` `version` + `date-released`
- [ ] the git tag (created last, only on approval) matches

> For this round the target is **v0.1.1**, **not** v0.1.0 — v0.1.0 is already
> published on `origin` (tag at `16391c4`). Force-moving a published tag is
> destructive and dishonest; a no-benchmark-change reliability pass is a patch
> bump.

## 2. Numbers did not drift

- [ ] `python scripts/check_repo.py` passes (this cross-checks the four formal
      results in README against `reproducibility.lock.yaml`, plus the metric
      formula across README / methodology / lock).
- [ ] No new prediction directory / manifest / scorer evidence was invented.
      If you actually re-ran a benchmark, attach the new `run_manifest.json` and
      scored `run_summary.json` and update the lock + README together.

## 3. Clean-build + install verification (run, don't assert)

- [ ] `python -m compileall -q src scripts`
- [ ] `ruff check .` and `ruff format --check .`
- [ ] `bash -n scripts/*.sh`
- [ ] `pytest -q -m "not gpu" --cov=hunyuan_ocr --cov-report=term-missing`
- [ ] `python scripts/check_repo.py`
- [ ] `reuse lint`
- [ ] `python -m build`
- [ ] Clean-venv wheel install:
      ```bash
      python -m venv /tmp/hunyuan-ocr-wheel-test
      /tmp/hunyuan-ocr-wheel-test/bin/pip install dist/*.whl
      /tmp/hunyuan-ocr-wheel-test/bin/hunyuan-ocr --help
      /tmp/hunyuan-ocr-wheel-test/bin/hunyuan-ocr doctor --json
      ```
- [ ] Extras install: `pip install -e ".[client,download,dev]"` and confirm
      `hf`, `openai`, `pytest`, `ruff`, `reuse` are available.

## 4. CI is green

- [ ] `ci` workflow passes on the release commit for **all** of Python 3.11 /
      3.12 / 3.13 (CPU-only, no torch).
- [ ] `reuse lint` step is green.
- [ ] Coverage is at/above the gate (`--cov-fail-under=65`).

## 5. ROCm preflight honesty

- [ ] If dispatching `rocm-runner-preflight`: it runs `doctor --strict --backend
      llamacpp --json` and `scripts/rocm_smoke.sh` against a runner that actually
      holds the weights + a smoke page. Do **not** mark a release "GPU-verified"
      unless a real smoke passed and its artifacts are attached.
- [ ] No gfx1100 runner available ⇒ state the limitation in the release notes;
      do not imply a smoke passed.

## 6. Release assets

- [ ] `dist/hunyuan_ocr-<ver>-py3-none-any.whl`
- [ ] `dist/hunyuan_ocr-<ver>.tar.gz`
- [ ] `SHA256SUMS` (sha256 of both dist artifacts) attached to the release.
- [ ] (Optional) `rocm-preflight` artifact archive if a real smoke ran.

## 7. License & restrictions (must be stated, not assumed)

- [ ] `reuse lint` passes (SPDX headers + `REUSE.toml`).
- [ ] Upstream-derived files remain `LicenseRef-Tencent-Hunyuan-Community-License`
      (never re-marked Apache-2.0).
- [ ] Release notes state: weights are under the Tencent Hunyuan Community
      License (not OSI Open Source; excludes EU/UK/KR); original packaging/tooling
      is Apache-2.0.
- [ ] No weights / datasets / predictions / secrets / private paths in the tree
      or the artifacts.

## 8. Known limitations (state explicitly)

- vLLM full-set is **invalid** (46.31 excluded); only the vLLM canary (94.81) is
  a reliable vLLM number.
- Not precision-aligned (no same-page-set CUDA control; official 94.10 uses
  TensorRT on an unlabeled dataset version).
- `predict --backend transformers` requires the repo checkout + a ROCm torch.
- GitHub Actions are not yet pinned to full commit SHAs (Dependabot configured;
  SHA-pinning is a tracked follow-up).

## 9. Publish (only on explicit authorization)

- [ ] Tag `v<ver>` on the release commit and push the tag.
- [ ] Create the GitHub Release from the tag, attaching the dist assets +
      `SHA256SUMS`.
- [ ] Suggested repo "About" update (do not run without authorization):
      ```bash
      gh repo edit AIwork4me/HunyuanOCR-ROCm \
        --description "Evaluation-backed AMD ROCm port of HunyuanOCR-1.5 with reproducible OmniDocBench v1.6 benchmarks across llama.cpp, vLLM, and Transformers on gfx1100."
      ```
