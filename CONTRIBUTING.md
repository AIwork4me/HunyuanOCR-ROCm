# Contributing to HunyuanOCR-ROCm

Thanks for considering a contribution. This project is an **evaluation-backed**
AMD ROCm port of HunyuanOCR-1.5 — keep claims evidence-scoped (see
[docs/benchmark-methodology.md](docs/benchmark-methodology.md)).

## Set up

```bash
git clone https://github.com/AIwork4me/HunyuanOCR-ROCm.git
cd HunyuanOCR-ROCm
pip install -e ".[client,dev]"     # CPU core + openai client + dev tools; NO torch
```

ROCm PyTorch is **not** installed by the above; it is only needed for the
transformers/vLLM backends. Install it separately from a verified ROCm source.

## Tests (CPU, no GPU required)

```bash
pytest -q                 # the acceptance command; must pass with NO torch installed
ruff check .
ruff format --check .
python -m compileall -q src scripts
bash -n scripts/*.sh
python scripts/check_repo.py   # lock, canary manifest, doc links, SPDX
```

CI ([.github/workflows/ci.yml](.github/workflows/ci.yml)) runs exactly these on
`ubuntu-latest` Python 3.12 with **no torch**. A change that only passes with
torch installed is not acceptable.

## GPU tests

End-to-end GPU tests are marked `@pytest.mark.gpu` and are **deselected in CI**.
They require a gfx1100 box with weights + dataset. Run them locally with
`pytest -m gpu`. Never describe a CPU/mock test as a GPU test.

## Code style

- `ruff check .` and `ruff format --check .` must return 0.
- **Vendored** files (`src/hunyuan_ocr/postprocess.py`, `tasks.py`, `contract.py`)
  are lint-only and **excluded from formatting** so diffs against upstream stay
  trackable (see `[tool.ruff]` in `pyproject.toml`). Do not reformat them.
- Every `src/**/*.py` and `scripts/**/*.py` carries an `SPDX-License-Identifier`
  header. `reuse lint` must pass.

## License

- **Original** code: Apache-2.0 (use the SPDX header shown in `runner.py`).
- **Code ported from HunyuanOCR**: license id `LicenseRef-Tencent-Hunyuan-Community-License` with the Tencent + AIwork4me copyright block (see `contract.py` for the template). Never mark upstream-derived code as Apache.
- Do **not** commit model weights, datasets, secrets, or private documents (see [SECURITY.md](SECURITY.md)).

## Benchmark changes

If you change anything that affects scores (prompt, sampling, post-processing,
resolution policy, scorer commit, metric config), you have **re-baselined**. In
your PR you must:

1. State which page set changed (canary 148 / full 1651) and never mix them.
2. Provide the new numbers **and** the manifest (`run_manifest.json`) proving the
   run was complete.
3. Not claim "precision-aligned" without a same-page-set CUDA control.
4. **Never** write an invalid/diagnostic number into the README as a formal
   result (e.g. the vLLM full-set 46.31 is excluded).

## PR checklist

- [ ] `pytest -q`, `ruff check .`, `ruff format --check .`, `python -m build` all pass.
- [ ] `python scripts/check_repo.py` passes.
- [ ] New behavior has a CPU unit test (no torch / no GPU).
- [ ] No unverified numbers, commits, checksums, or "precision-aligned" claims.
- [ ] License headers + `reuse lint` clean.
