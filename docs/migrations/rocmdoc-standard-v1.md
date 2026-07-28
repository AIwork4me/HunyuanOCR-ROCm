# Migration Record — HunyuanOCR-ROCm → ROCmDoc Model Repository Standard v1

This records the migration of HunyuanOCR-ROCm to the central OmniDocBench-ROCm
platform contract. It is a decision log, not a benchmark report. No GPU test that
was not actually run is claimed here.

## Central lock

| field | value |
|---|---|
| central repository | AIwork4me/OmniDocBench-ROCm |
| central commit | `ccd466ef317fd6a710131db3a19ec9d55a65ce2e` |
| central branch at lock | `main` (clean working tree) |
| central package version | 0.3.2 (declared) / 0.3.3 (committed by upstream maintainer — `rfc3339-validator` so `date-time` formats are actually enforced) |
| lock file | `.rocmdoc/spec-lock.json` |

## Migration scope + date

- migrated on: 2026-07-27
- execution mode: `all-safe` (implementation + evidence migration; **no GPU inference, no new scoring**)
- baseline repository commit: `d3ab6aa1ad990d7107b006f334a410a6321d98f0` (branch `main`)

## Old model card → new results[] mapping

The legacy v1 `model_card.json` is **retained** (the structural checker
`check_repo` validates it against the v1 `$def`, `schema_version:const 1`). The v2
single source of truth is the new `model_card_v2.json`.

| legacy v1 field | v2 mapping |
|---|---|
| `overall: 93.64` (backend vllm) | primary `result_record` `hunyuan-ocr__linux-rocm__vllm__bf16__v1-6__38a99096c23d`, `assurance=evidence-complete`, `metrics.overall=93.64` |
| `badge.linux-rocm: community` | (no model-wide badge in v2 — ADR-0008; capability `status=supported` in `rocmdoc.yaml`) |
| `badge.windows-hip: community-wanted` | `rocmdoc.yaml` implementation `windows-hip/llama-cpp status=planned` (no result) |
| `license`, `commercial_use` | `license_record` (category `restricted`; code+weights Tencent Hunyuan Community, excludes EU/UK/KR) |
| `hardware` | per-`result_record` `hardware` |

Additional v2 `result_record` added (not in v1 card): llama.cpp full 1651 = 92.09,
`assurance=submitted` (documented; no committed platform artifacts).

## Legacy badge → assurance mapping

v1 had a single model-wide `badge`. v2 assurance is **per-result** and never
propagated (ADR-0008):

| legacy badge concept | v2 |
|---|---|
| "verified" (was not held) | — |
| `community` (linux-rocm result exists) | primary vLLM result → `evidence-complete` (files complete, `prediction_manifest_sha256` verified, page conservation 1651==1651, backend clear) |
| documented but not artifact-committed | llama.cpp full → `submitted` |

`evidence-complete` was awarded only after verifying: artifact files present;
`prediction_manifest_sha256` `995f8fff…` matches the committed file; page
conservation 1651 ok / 0 failed / 0 fallback; actual backend `vllm`; weights +
recipe locatable in `REPRO.yaml`. **`score-reproduced` was NOT awarded** — it
requires actually re-scoring the fixed predictions against the scorer, which was
not performed in `all-safe` mode.

## Conflict notes (no auto-resolution)

This repo carried multiple Overall figures for overlapping combinations. They
were classified, not silently reconciled (raw evidence is authoritative; §1.3/§8.3):

- **93.64 vs 91.31 (vLLM, full 1651, same prediction set `predictions_vllm_full_seq`):**
  `93.64` is computed from the platform `run_summary.json` `readme_metrics` +
  `metric_result.json` **page** aggregates (text 0.0452, CDM 92.46, TEDS 92.97).
  `91.31` (hand-written in README, labelled "validated") uses a **mixed**
  aggregation (page CDM 92.46 + `all_page_avg` TEDS 86.00). The canonical value is
  `93.64` (evidence-derived); the README's `91.31` is superseded by the generated
  results block. See `audits/hunyuan.json` conflict_group `different-result-aggregation-mismatch`.
- **Canary 148-page numbers (vLLM 94.81, transformers 94.11, llama.cpp 93.33):**
  experimental/historical subset measurements (per `reports/canary-baseline.md`).
  The central `result_id` tuple `(model, platform, backend, precision, benchmark)`
  does **not** encode page-set, so canary and full collide; they are therefore NOT
  separate `result_records`. They are recorded in `docs/benchmark-context.md` as
  experimental references. No committed platform artifacts exist for them.
- **Official TensorRT 94.10 / upstream-reported 94.74:** external references only,
  in `docs/benchmark-context.md`. Never in `results[]`.

## Unmigrated / unverifiable artifacts

- `predictions_vllm_full_capped/` — empty (superseded by `predictions_vllm_full_seq/`). Not migrated.
- canary + llama.cpp-full prediction artifacts — machine-local, not committed to
  this repo; not migrated as platform artifacts. The llama.cpp-full result is
  recorded as `submitted` only.

## Retained compatibility interfaces

- All expert CLI commands retained: `doctor` (now also emits central `status`),
  `validate`, `manifest verify`, `canary materialize`, `predict`, `score`,
  `benchmark`, `report`.
- v1 `model_card.json` retained (structural checker requires it).
- `adapter/run_adapter.py` unchanged (the OmniDocBench engine still calls it).
- The new `version`/`capabilities`/`parse` commands share ONE inference core
  (`backends.vllm_client.infer_one`) with `predict` and the adapter.

## Known behavior changes

- `doctor --json` now includes a `status` field (`ready`/`not-ready`) per the
  central doctor contract; the legacy `ok` field is retained.
- README results are now generated from `model_card_v2.json`
  (`scripts/generate_results_block.py --check` enforces no drift). The canonical
  vLLM Overall shown is `93.64`; the historical hand-written `91.31` is superseded.
- `benchmark-omnidocbench-v16` conformance is demonstrated via the fake-CLI
  fixture `tests/fixtures/fake_cli.py` (contract machinery, no GPU) — the same
  pattern the central repo uses in its own CI. The real-model benchmark
  (`hunyuan-ocr parse` against a live server on the full 1651-page set) is a GPU
  workflow and was **not** run in `all-safe` mode.

## Conformance (run 2026-07-27, central `ccd466e`)

| profile | result | against |
|---|---|---|
| structural (`check_repo`) | CONFORMANT | repo |
| `base` | PASS | real `hunyuan-ocr` CLI |
| `runtime-core` | PASS | real `hunyuan-ocr` CLI |
| `benchmark-omnidocbench-v16` | PASS | fake-CLI fixture (contract machinery) |
| `reproducible-score` | PASS | vLLM `result_record` (hash verified) |
| existing pytest | 163 passed, 1 skipped | — |

## Rollback

All new files are additive (`rocmdoc.yaml`, `model_card_v2.json`,
`.rocmdoc/spec-lock.json`, `docs/migrations/`, `docs/benchmark-context.md`,
`PATCHES.md`, `src/hunyuan_ocr/standard_cli.py`, `tests/fixtures/`,
`scripts/generate_results_block.py`). The only edits to existing files are: the
three new standard subcommands + the `status` field in `doctor --json`
(`src/hunyuan_ocr/cli.py`), and the generated results block in `README.md`
(between ROCmDoc markers). Rollback = revert `cli.py` and remove the generated
block; delete the additive files. No historical result was deleted or altered.
