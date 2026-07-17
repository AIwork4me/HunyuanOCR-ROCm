## Summary

<!-- What does this change do, and why? -->

## Type

- [ ] bug fix
- [ ] new feature / tooling
- [ ] reproducibility / lock change
- [ ] docs
- [ ] benchmark-result change (re-baseline)

## Verification (run locally; paste summaries)

- [ ] `pytest -q` —
- [ ] `ruff check .` / `ruff format --check .` —
- [ ] `python -m build` —
- [ ] `python scripts/check_repo.py` —
- [ ] new behavior has a CPU unit test (no torch / no GPU)

## If this touches results

- [ ] I named the page set (canary 148 / full 1651) and did **not** mix them.
- [ ] I attached the `run_manifest.json` proving completeness + conservation laws.
- [ ] I did **not** add an invalid/diagnostic number as a formal result.
- [ ] I did **not** claim "precision-aligned" without a same-page-set CUDA control.

## License

- [ ] Original code is Apache-2.0; upstream-derived code uses
      `LicenseRef-Tencent-Hunyuan-Community-License` (see `contract.py`).
- [ ] No secrets, weights, datasets, or private documents committed.
- [ ] `reuse lint` passes.
