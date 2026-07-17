# HunyuanOCR-ROCm — P0 Hardening Design

**Date:** 2026-07-17
**Status:** Approved (brainstorming output) — pending implementation plan
**Branch:** `p0-integrity-license-repro` (off `main` @ `e17fab1`)
**Scope:** First-round P0 only. Four tasks. No push / no PR.

---

## 1. Problem statement

The repo is technically rich but not yet a *trustworthy, third-party-reproducible* project. Concretely (all verified during exploration):

1. **False-completion risk.** Both prediction drivers write `ERROR: <ExcType>: <msg>` into the *final* `.md` on inference failure, resume by `.exists()` only, and can exit 0 after printing `[done]` while pages are actually broken. `run_phase1_transformers.py` ignores worker exit codes; `run_phase2_vllm.py` swallows exceptions inside `work()` and never calls `Future.result()`. The HANDOFF already documents the real incident this caused: a vLLM full-set run produced **~780 ERROR pages → a false 46.31 score**.
2. **No pre-score validation.** `score_predictions.py` invokes the OmniDocBench scorer on whatever is in the pred-dir — missing pages, empty files, `.partial` files, `ERROR:` markers, duplicate stems are all silently scored.
3. **License is wrong/misleading.** Root `LICENSE` is an *abridged* Apache-2.0 (no full §§2–9, no `APPENDIX`) — GitHub cannot detect it. NOTICE overstates the "Powered by Tencent Hunyuan" notice as mandatory (it is **encouraged** under §3c), omits the **mandatory** §3d notice string and the §3e non-affiliation statement, and upstream-derived code is blanket-declared pure Apache-2.0.
4. **Over-attribution in public conclusions.** README/reports/src assert the formula-CDM gap is a *confirmed* root cause, that the >14k ViT instability is *only* an SDPA-kernel problem, and that llama.cpp is *the* fastest/most-stable backend — claims stronger than the evidence supports.
5. **Reproducibility not frozen.** No machine-readable lock file; llama.cpp is "clone master"; no canary manifest; reproduce scripts hardcode `/root`.

## 2. Goals / non-goals

**Goals (P0):**
- (T1) Prediction integrity: atomic output, separated error records, correct resumability + retries, real failure propagation (non-zero exit), output-name conflict detection, pre-score validation gate, per-run manifest, CPU-only tests.
- (T2) License/attribution: full Apache-2.0 `LICENSE`, vendored upstream Tencent license, honest per-file attribution headers, corrected NOTICE, corrected README license section.
- (T3) Downgrade over-attribution; unify doc status (historical headers, remove stale "in progress", invalidate the 46.31).
- (T4) Freeze reproducibility: `reproducibility.lock.yaml`, locked llama.cpp commit, generated canary manifest, parameterized reproduce scripts, README "Reproducibility" section.

**Non-goals (explicitly deferred):**
- Running any GPU inference / re-deriving scores.
- Refactoring vendored upstream algorithm files (postprocess patterns, etc.).
- Rewriting the concurrency model of the drivers.
- Anything under "Phase 4" (OmniDocBench-AMD integration).

## 3. Verified facts (do not re-derive during implementation)

| Fact | Value | Source |
|---|---|---|
| Repo HEAD | `e17fab1d3c2586599b9ee0c845784e4b000e2101` | `git rev-parse HEAD` |
| llama.cpp commit `a320cbf` (full) | `a320cbfcb7056b7b81fb854d97fe01d0ea77c4b5` (2026-07-16; contains `tools/mtmd/models/hunyuanvl.cpp`, `src/models/hunyuan-vl.cpp`) | GitHub API |
| Upstream HunyuanOCR default branch | `main`; repo also has `License.txt` == `LICENSE` (identical Tencent Hunyuan Community License) | GitHub API |
| Apache-2.0 canonical text | 202 lines / 11358 chars; has `TERMS AND CONDITIONS` + `APPENDIX` | apache.org (cached `/tmp/apache2.txt`) |
| Tencent license — §3c "Powered by Tencent Hunyuan" | **encouraged** (not mandatory) | upstream §3c |
| Tencent license — §3d mandatory Notice string | `"Tencent Hunyuan is licensed under the Tencent Hunyuan Community License Agreement, Copyright © 2025 Tencent. All Rights Reserved. The trademark rights of "Tencent Hunyuan" are owned by Tencent or its affiliate."` | upstream §3d |
| GT SHA256 — canary (`OmniDocBench_150.json`) | `3e3fbea07702084d9466e231260ad92141848a32631c9895d8e55b24e2c2f7b5` (148 pages) | `sha256sum` |
| GT SHA256 — full (`OmniDocBench.json`) | `a45cd84b04ad8b793e775089640e6b681209abea33ead54c1828ddca35fae496` | `sha256sum` |
| OmniDocBench scorer commit (local) | `2b161d010d2e3aff77a0edef359ea3a6411d23cd` (`/root/ocr-eval/OmniDocBench`) | `git rev-parse HEAD` |
| OmniDocBench repo | `opendatalab/OmniDocBench`, default `main`, CVPR 2025; v1.6 | GitHub API |
| Python | 3.12.3 | `sys.version` |
| ROCm/HIP | `7.2.53211-e1a6bc5663` | `torch.version.hip` |
| torch | `2.9.1+gitff65f5b` | `torch.__version__` |
| transformers — benchmark venv | `5.13.0` (REPORTED in HANDOFF, isolated `/root/hunyuanocr-venvs/transformers`) | HANDOFF |
| transformers — this `/opt/venv` | `4.57.6` (verified this session) | `transformers.__version__` |
| vLLM | `0.16.1.dev0+g89a77b108.d20260317` | `vllm.__version__` |
| GPU | gfx1100 / RDNA3; rocm-smi device id `0x744b` | `rocm-smi` |

**Not reachable from this env (mark `not_recorded` + give fill command):** HF model revision (`tencent/HunyuanOCR` `main` sha); GGUF LFS oid (`ggml-org/HunyuanOCR-GGUF`) — the `huggingface.co` API returns empty here (MITM proxy).

**Honesty invariant:** the published benchmark was **not** produced in this `/opt/venv` (transformers 4.57.6 here vs 5.13.0 benchmark venv). The lock file records *both* with provenance; never asserts the current env produced the benchmark.

## 4. Design decisions (approved)

1. **Shared primitives, not a shared loop.** New `runner.py` exposes atomic-write / error-record / resume / conflict-detect / manifest primitives. Each driver keeps its own concurrency shape (phase1 = `spawn` procs; phase2 = threads) but calls the primitives. (Chosen over "minimal localized edits" — removes duplicated *policy* without a god-loop; respects scope discipline.)
2. **Git:** new branch `p0-integrity-license-repro`; chunked commits per task; no push/PR.
3. **Historical paths:** machine-local evidence paths stay inside historical reports (with `Historical` header); parameterized/env-var paths only in repro scripts, lock file, README.

## 5. Task 1 — Prediction integrity

### 5.1 Files

| File | Change |
|---|---|
| `src/hunyuan_ocr/runner.py` (NEW) | `write_atomic`, `commit_success`, `record_error`, `aggregate_errors`, `is_complete`, `detect_stem_conflicts`, `write_run_manifest`, `safe_argv`, `iter_outcomes` |
| `src/hunyuan_ocr/validation.py` (NEW) | `validate_predictions(gt_json, pred_dir, *, strict=True) -> Report`; dataclasses `Report`/`Problem`; severity levels |
| `scripts/validate_predictions.py` (NEW) | CLI wrapper; exit 0 iff no hard errors |
| `scripts/run_phase1_transformers.py` (EDIT) | use runner primitives; `mp.Queue` results; exit-code gate; `--max-retries/--retry-backoff/--overwrite/--retry-failed`; manifest |
| `scripts/run_phase2_vllm.py` (EDIT) | use runner primitives; real `future.result()` drain; endpoint-health rotation; retries; manifest |
| `scripts/score_predictions.py` (EDIT) | run `validate_predictions` first; fail non-zero without invoking scorer; add `--skip-validation` (loud warning) |
| `conftest.py` (NEW, repo root) | prepend `src/` to `sys.path` so `pytest -q` works without editable install |

### 5.2 Page outcome state model

After a run cycle each page is exactly one of:
- **COMPLETE** — valid `<stem>.md` present (non-empty, not starting with `ERROR:`) **and** no `_errors/<stem>.json`
- **FAILED** — `_errors/<stem>.json` present (attempts exhausted)
- **PENDING** — neither, or only a stale `.partial`

### 5.3 CLI flags (each distinct, all bounded)

| Flag | Default | Semantics |
|---|---|---|
| resume (no flag) | on | skip COMPLETE only; **FAILED and PENDING are re-run** → "retry failed pages next run" is the default |
| `--retry-failed` | off | scope this run to **FAILED pages only** (leave COMPLETE and PENDING) |
| `--overwrite` | off | also re-run COMPLETE (delete first) |
| `--max-retries` | 2 | in-run bounded retry (initial + 1 retry) |
| `--retry-backoff` | 2.0 | exponential backoff seconds between in-run retries |

### 5.4 Atomic output & error records

- `write_atomic(path, md)`: write `path.with_suffix(path.suffix + ".partial")` → `flush()` + `os.fsync()` → `os.replace(→path)`. On exception, unlink `.partial` and re-raise. **Never write `ERROR:` md.**
- `record_error(pred_dir, stem, *, image_path, backend, endpoint, exc, attempt, ts)`: writes `_errors/<stem>.json` (one file per page → no write race; thread/process safe). Fields: image_path, backend, endpoint, exc_type, exc_message, attempt, timestamp. Presence of `_errors/<stem>.json` ⇒ page FAILED.
- **Success clears the error record.** A helper `commit_success(pred_dir, stem, md)` wraps `write_atomic` and then `unlink(_errors/<stem>.json, missing_ok=True)`. This preserves the invariant **COMPLETE ⟺ valid `.md` present AND no `_errors/<stem>.json`** across retries (a page that fails attempt 1 then succeeds attempt 2 must not retain a stale error file). All success paths go through `commit_success`, never raw `write_atomic`.
- `aggregate_errors(pred_dir)`: called **once** by the main thread/process after all workers join → `_errors.jsonl`. Never concurrent.

### 5.5 Failure propagation

**Phase 1 (multiprocessing):** workers push `(gpu, stem, status, err)` tuples to a shared `multiprocessing.Queue`; main drains + joins. Main exits **non-zero** if any `proc.exitcode != 0` (worker crash / model-load failure) **or** ≥1 FAILED page (post-retry) **or** any PENDING page (incomplete). `[done]` printed only when all COMPLETE.

**Phase 2 (threads):** `work()` does bounded retry internally; on final failure calls `record_error(...)` and returns `{stem,status}`. **Never swallow exceptions internally.** In the `as_completed` drain: call `future.result()` so any unhandled exception (write error, client error) propagates to main and aborts. Endpoint health: keep a live set; on connection error, probe once and drop the endpoint from rotation for a cooldown (logged). No fancy circuit-breaker.

### 5.6 Pre-score validation (`validation.py`)

Checks: expected count (from GT), valid count, **missing** pages, **unexpected** files (excluding our own `_errors/`, `_errors.jsonl`, `run_manifest.json`, `*.partial`), **empty** files, **`.partial`** files, **`ERROR:`** markers, **duplicate stems** (within GT), **unresolved `_errors/*.json`**.

Two tiers: **hard errors** (missing/empty/partial/ERROR/unresolved) always make the report non-zero and block scoring; **warnings** (unexpected files) are non-zero under `--strict` (the default) and non-fatal under `--lenient`. CLI exit code: **0 iff no hard errors and (under `--strict`) no warnings.** Prints a summary either way.

`score_predictions.py`: runs validation first (strict); on non-zero report, prints problems and exits **without** invoking the scorer. `--skip-validation` bypasses but prints `WARNING: validation bypassed — score may be invalid`.

### 5.7 Run manifest

`run_manifest.json` (in pred_dir): repo commit, backend, model id+path, model revision (read from config or null), sanitized command line (scrub `*_TOKEN`/`*KEY*`/`OPENAI_API_KEY`/`HF_TOKEN`), timestamp, expected/succeeded/failed/skipped counts, ports/GPU, pixel cap, max tokens, env versions (torch/transformers/vllm/hip, lazy→null), final status. No secrets.

### 5.8 Tests (CPU-only) — `tests/test_runner.py`, `tests/test_validation.py`

Cover all 10 acceptance bullets (§9.1). Fake `infer` callable (raises on chosen pages) drives the runner; no GPU/server.

## 6. Task 2 — License & third-party attribution

### 6.1 `LICENSE` (root) ← full unabridged Apache-2.0 (202 lines). Project copyright line in the header block above the standard text.

### 6.2 `LICENSES/` (NEW)
- `Apache-2.0.txt` — same full text.
- `Tencent-Hunyuan-Community-License.txt` — **verbatim** upstream `LICENSE` (fetched via API).

### 6.3 Per-file attribution audit (honest; not blanket Apache)

| File | Attribution | Header |
|---|---|---|
| `omnidocbench.py`, `scoring.py`, `runner.py`, `validation.py`, `validate_predictions.py`, `score_predictions.py` | Original | `SPDX-License-Identifier: Apache-2.0 ; Copyright 2026 AIwork4me` |
| `contract.py` | **verbatim values** from upstream HunyuanOCR inference recipe | Derivative — Tencent license |
| `tasks.py` | 12 task prompts are **verbatim upstream text** | Derivative — Tencent license |
| `postprocess.py` Group 1 (`has_tail_repetition`/`clean_repeated_substrings`/`infer_stream`/`encode_image_as_data_url`) | mirrors upstream `hunyuan_utils`/`infer_vllm` | Derivative — Tencent license |
| `postprocess.py` Group 2 (`process_one`, 10 patterns) | README says "verbatim port", file docstring says "original normalization" → **genuine uncertainty** | Flagged **mixed/uncertain → maintainer-confirm** (conservative) |
| `backends/transformers.py` | ported from upstream `infer_hf_8gpu_hyocr15.py` + original gfx1100 adaptations | Mixed — Tencent (ported logic) + Apache (adaptations) |
| `backends/vllm_client.py` | ported from upstream `batch_infer.run_one` + original client | Mixed — Tencent + Apache |

Each derivative/mixed file gets a header block preserving upstream copyright + source; NOTICE and README license tables list them honestly. Uncertain items land in the "open items" list.

### 6.4 `NOTICE` (rewrite per §3d/§3e)
- Mandatory §3d notice string **verbatim**.
- §3e non-affiliation statement ("Tencent is not affiliated, associated with, sponsoring, or endorsing…").
- "Powered by Tencent Hunyuan" described as **encouraged (§3c), not required**.
- State weights are **not** OSI Open Source.
- Point to `LICENSES/` for full texts.

### 6.5 README license section
Four categories (original tooling = Apache-2.0; ported code = mixed/Tencent; weights = Tencent, not OSI; llama.cpp = MIT). Badge `code-Apache-2.0` → `license-mixed (see NOTICE)`. `License.txt` link → our vendored `LICENSES/` copy + upstream reference.

## 7. Task 3 — Downgrade over-attribution + doc status

Before → after (README / reports / src):

- `confirmed by systematic ablation` / `confirmed to be inference-engine-level` → **"After ruling out resolution, streaming, post-processing, and systematic formula omission, inference-engine-level numerical divergence is the **leading explanation**."** (keeps what was ruled out; states what is not singly proven)
- `only affects the SDPA path` / `the >14k NaN only affects transformers SDPA` → **"Observed in the transformers/ROCm full-ViT forward path using SDPA on the tested gfx1100 stack. A standalone SDPA op does not reproduce it; we have not isolated it to a specific kernel, and have no NVIDIA control to rule out non-ROCm environments."**
- `llama.cpp is the fastest and most stable backend … the only one that runs at full resolution` → **"Among the three tested backends, on the tested gfx1100/ROCm 7.2 stack, llama.cpp is fastest and most stable with no pixel cap."**
- Add explicit **vLLM-full invalidity block**: "The vLLM canary 148-page result (94.81) is the reliable vLLM number. The 1651-page full-set run **never produced a valid score** — servers crashed under sustained load, yielding ~780 ERROR pages; the resulting 46.31 is **not a valid benchmark** and is excluded from all comparisons."
- Add `> **Historical** — 2026-07-16. Retained as evidence; README is the single source of current status. Some conclusions here read stronger than the evidence now supports.` to `HANDOFF.md`, `canary-baseline.md`, `project-stage-summary.md`. Keep machine-local evidence paths; fix stale "28 commits / feature branch / in progress" state.
- `git rm --cached docs/.ipynb_checkpoints/rocm-issue-draft-checkpoint.md` (gitignored but tracked).
- Scan src/scripts comments for `confirmed`/`root-caused`/`only` and downgrade.

## 8. Task 4 — Reproducibility freeze

### 8.1 `reproducibility.lock.yaml` (NEW) — values per §3 table. `not_recorded` for HF-model-revision + GGUF-oid with a `# how-to-fill` command block per field.

### 8.2 `scripts/create_canary_manifest.py` (NEW)
Reads a GT json → emits `eval/canary_148.manifest.json` (subset name, source dataset/version, expected_count=148, sorted page stems + image_paths, source SHA256, manifest SHA256). **Run on the real `OmniDocBench_150.json` and commit the generated manifest** (148 stems, deterministic, no secrets). Document regeneration.

### 8.3 `scripts/reproduce_llamacpp_canary.sh` + `scripts/reproduce_llamacpp_full.sh` (NEW)
`set -euo pipefail`; env-var paths (`LLAMA_DIR`, `GGUF_DIR`, `DATA_DIR`, `GT_JSON`, `OUT_DIR`, `PORTS`; `HOST` default `127.0.0.1`); check inputs exist; document locked commit (auto-clone only with explicit `--download`); predict (`run_phase2_vllm.py`) → `validate_predictions.py` → `score_predictions.py`; any step failure ⇒ non-zero; no auto-download; overwrite-guarded.

### 8.4 README "Reproducibility" section
Published results ↔ lock file; how to reproduce canary vs full-set; which numbers are reliable-formal vs diagnostic-only; why newer deps may differ.

## 9. Acceptance & verification

### 9.1 CPU-only functional acceptance (must run + paste output)
1. Build 3-page fake dataset.
2. Fake infer raises on 1 page.
3. Exactly 2 valid `.md` produced.
4. Failed page → `_errors/<stem>.json` record.
5. Run exits non-zero.
6. Re-run: 2 complete skipped, failed retried.
7. Fabricate empty / `ERROR:` / `.partial` files.
8. `validate_predictions` catches all.
9. `score_predictions` rejects the invalid dir.
10. All-valid dir → validation passes.

### 9.2 Static checks
`git diff --check`; `python -m compileall src scripts`; `pytest -q` (new ~20 CPU tests); `ruff check .` (minimal `[tool.ruff]` with explained per-file ignores for vendored/upstream-derived files — **P0 ruff scope: opt-in**).

### 9.3 License acceptance
Root LICENSE = full standard Apache-2.0; Tencent license vendored verbatim; README no broken `License.txt` link; "Powered by" not described as mandatory; upstream-derived code not blanket Apache; all external links resolve.

### 9.4 Doc acceptance
No "vLLM full run in progress"; 46.31 marked invalid; formula CDM = leading explanation (not confirmed root cause); ViT issue not attributed to a single SDPA kernel; "only/fastest/most stable" scoped to tested env + 3 backends; README/HANDOFF/stage-summary mutually consistent.

## 10. Open items / needs-maintainer-confirm

- **`postprocess.py` Group 2** (`process_one`): README vs docstring conflict — flagged mixed/uncertain; needs maintainer ruling (upstream-derivative vs original).
- **HF model revision + GGUF LFS oid**: not reachable from this env; left `not_recorded` with fill commands.
- **transformers benchmark venv (5.13.0)**: reported in HANDOFF, not re-verified this session; recorded as reported alongside the verified current-env 4.57.6.
- **GPU re-derivation of scores**: out of P0 scope; no GPU run claimed as done.

## 11. Risks

- **Existing pred-dirs with legacy `ERROR:` md** (e.g. the vLLM full-set 780-error dir): new resume treats them as FAILED/PENDING → will be re-run, not silently skipped. (Behavioral improvement, but changes old dirs' resume semantics.)
- **`pytest -q` currently broken** without editable install; root `conftest.py` fixes it — CI/Makefile that assumed install are unaffected (still work).
- **License scope**: declaring code "mixed/Tencent-derivative" rather than pure Apache is a stricter, more honest posture; downstream re-users who assumed Apache-only should re-check NOTICE.
- **README links**: any rewritten badge/link must be re-checked to resolve.
- **No score changes**: nothing here touches model weights or decoding contract → published canary/llama.cpp-full numbers unchanged.
