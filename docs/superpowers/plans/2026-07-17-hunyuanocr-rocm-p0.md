# HunyuanOCR-ROCm P0 Hardening Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Harden HunyuanOCR-ROCm so its predictions can't silently fail-complete, its licensing is honest and complete, its public conclusions are evidence-scoped, and its results are third-party-reproducible — the four P0 tasks.

**Architecture:** Add shared filesystem-level integrity primitives (`runner.py`) and a pure validation module (`validation.py`) consumed by both prediction drivers and the scorer. Rewrite license/attribution as honest mixed-licensing. Downgrade over-attributed conclusions in prose. Freeze reproducibility in a machine-readable lock file + parameterized scripts.

**Tech Stack:** Python 3.12, stdlib only for the new integrity/validation code (no new deps); existing deps (torch/transformers/vllm/openai/PIL/yaml) remain lazy-imported. Bash for reproduce scripts. Markdown for docs/license.

**Spec:** `docs/superpowers/specs/2026-07-17-hunyuanocr-rocm-p0-design.md` (approved, committed `b40b46a`). This plan implements it verbatim.

## Global Constraints

- **Branch:** `p0-integrity-license-repro` (created off `main` @ `e17fab1`). **No push, no PR.**
- **Chunked commits:** logical commits per task boundary (Task 1 = several; Tasks 2/3/4 = one each). Every commit message ends with `Co-Authored-By: Claude <noreply@anthropic.com>`.
- **No GPU runs.** Every verification step is CPU-only. Never claim a GPU test passed.
- **No reformatting vendored upstream-derived files** (`postprocess.py` patterns, `tasks.py` prompts). Edits there are additive headers / minimal targeted comment changes only.
- **Honest provenance.** The benchmark was **not** produced in this `/opt/venv` (transformers 4.57.6 here vs 5.13.0 benchmark venv). Record both; never assert the current env produced the benchmark.
- **No fabrication.** Unknown commits/checksums/versions are recorded as `not_recorded` with a fill command, never invented.
- **Verified facts to reuse (from spec §3):** repo HEAD `e17fab1d3c2586599b9ee0c845784e4b000e2101`; llama.cpp commit `a320cbfcb7056b7b81fb854d97fe01d0ea77c4b5`; canary GT SHA256 `3e3fbea07702084d9466e231260ad92141848a32631c9895d8e55b24e2c2f7b5` (148 pages); full GT SHA256 `a45cd84b04ad8b793e775089640e6b681209abea33ead54c1828ddca35fae496`; scorer commit `2b161d010d2e3aff77a0edef359ea3a6411d23cd`; torch `2.9.1+gitff65f5b`; hip `7.2.53211-e1a6bc5663`; vLLM `0.16.1.dev0+g89a77b108.d20260317`; Python 3.12.3; GPU gfx1100 / rocm-smi device `0x744b`; Apache-2.0 canonical = 202 lines from `https://www.apache.org/licenses/LICENSE-2.0.txt`.

## File Structure

| File | Responsibility | New/Edit |
|---|---|---|
| `conftest.py` | prepend `src/` to `sys.path` so `pytest -q` works without editable install | NEW |
| `src/hunyuan_ocr/runner.py` | atomic write, error records, resumability, conflict detection, manifest | NEW |
| `src/hunyuan_ocr/validation.py` | pure pre-score validation → Report | NEW |
| `tests/test_runner.py` | CPU tests for runner primitives | NEW |
| `tests/test_validation.py` | CPU tests for validation | NEW |
| `tests/test_score_gate.py` | scorer refuses invalid dirs | NEW |
| `tests/test_phase2_no_gpu.py` | phase2 end-to-end with fake infer (CPU) | NEW |
| `scripts/validate_predictions.py` | validation CLI | NEW |
| `scripts/run_phase2_vllm.py` | rewrite to use runner primitives + real `future.result()` + retries + manifest | EDIT |
| `scripts/run_phase1_transformers.py` | rewrite to use runner primitives + worker exit codes + retries + manifest | EDIT |
| `scripts/score_predictions.py` | validation gate before scorer | EDIT |
| `LICENSE` | full unabridged Apache-2.0 | REPLACE |
| `LICENSES/Apache-2.0.txt` | vendored full Apache-2.0 | NEW |
| `LICENSES/Tencent-Hunyuan-Community-License.txt` | verbatim upstream Tencent license | NEW |
| `NOTICE` | corrected per §3d/§3e | REWRITE |
| `src/hunyuan_ocr/*.py`, `src/hunyuan_ocr/backends/*.py` | SPDX/attribution headers | EDIT |
| `README.md` | results scope, license, reproducibility | EDIT |
| `reports/HANDOFF.md`, `reports/canary-baseline.md`, `reports/project-stage-summary.md` | Historical headers + stale-state fixes | EDIT |
| `reproducibility.lock.yaml` | machine-readable repro freeze | NEW |
| `scripts/create_canary_manifest.py` | manifest generator | NEW |
| `eval/canary_148.manifest.json` | generated canary manifest | NEW |
| `scripts/reproduce_llamacpp_canary.sh`, `scripts/reproduce_llamacpp_full.sh` | parameterized reproduce scripts | NEW |
| `pyproject.toml` | optional minimal `[tool.ruff]` with per-file ignores (Task 1 end) | EDIT |

---

## Task 1: Prediction integrity

### Task 1.1: Unbreak `pytest -q` with a root conftest

**Files:**
- Create: `conftest.py`
- Test: `tests/test_smoke.py` (existing; should pass with plain `pytest -q`)

**Produces:** `pytest -q` works from repo root without `PYTHONPATH=src` or editable install.

- [ ] **Step 1: Create the conftest**

```python
# conftest.py
"""Make the in-tree `src/` layout importable for pytest without an editable install.

The package is NOT installed editable in CI/dev; this adds `src` to sys.path so
`pytest -q` (the documented acceptance command) collects tests that import
`hunyuan_ocr`.
"""
import sys
from pathlib import Path

_SRC = Path(__file__).resolve().parent / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))
```

- [ ] **Step 2: Verify plain pytest works**

Run: `cd /workspace/HunyuanOCR-ROCm && python -m pytest -q`
Expected: `23 passed, 1 skipped` (the pre-existing suite), rc 0. Previously this failed with `ModuleNotFoundError: No module named 'hunyuan_ocr'`.

- [ ] **Step 3: Commit**

```bash
git add conftest.py
git commit -m "test: root conftest so pytest -q works without editable install

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

### Task 1.2: `runner.write_atomic`

**Files:**
- Create: `src/hunyuan_ocr/runner.py`
- Test: `tests/test_runner.py`

**Interfaces:**
- Produces: `runner.write_atomic(path: Path, content: str) -> None`; module constant `runner.ERROR_PREFIX = "ERROR:"`, `runner._partial_of(path) -> Path`.

- [ ] **Step 1: Write the failing tests** (`tests/test_runner.py`)

```python
# tests/test_runner.py
from pathlib import Path
import pytest

from hunyuan_ocr import runner


def test_write_atomic_creates_final_and_no_partial(tmp_path):
    out = tmp_path / "page.md"
    runner.write_atomic(out, "# hello")
    assert out.read_text(encoding="utf-8") == "# hello"
    assert not (tmp_path / "page.md.partial").exists()


def test_write_atomic_is_atomic_on_error(tmp_path, monkeypatch):
    out = tmp_path / "page.md"
    import os as _os

    real_replace = _os.replace

    def boom(src, dst):
        # fail the rename step
        raise OSError("simulated rename failure")

    monkeypatch.setattr(_os, "replace", boom)
    with pytest.raises(OSError):
        runner.write_atomic(out, "data")
    # no final file, and the .partial was cleaned up
    assert not out.exists()
    assert not (tmp_path / "page.md.partial").exists()


def test_write_atomic_creates_parent_dir(tmp_path):
    out = tmp_path / "nested" / "deep" / "page.md"
    runner.write_atomic(out, "x")
    assert out.exists()
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_runner.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'hunyuan_ocr.runner'`.

- [ ] **Step 3: Create `src/hunyuan_ocr/runner.py` with `write_atomic`**

```python
"""Prediction-integrity primitives shared by the phase drivers.

Centralizes the rules that prevent "false completion":
  * atomic .md writes (partial -> fsync -> rename), never an ERROR: file
  * structured per-page error records (_errors/<stem>.json)
  * resumability that skips only genuinely-complete pages
  * output-name conflict detection
  * per-run manifest

No GPU, no model deps. Pure filesystem + stdlib.
"""
from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable

ERROR_PREFIX = "ERROR:"
_OWN_ARTIFACTS = {"_errors", "_errors.jsonl", "run_manifest.json"}


def _partial_of(path: Path) -> Path:
    return Path(path).with_suffix(Path(path).suffix + ".partial")


def write_atomic(path: Path, content: str) -> None:
    """Write ``content`` to ``path`` atomically.

    Writes ``<path>.partial`` first, flushes + fsyncs, then ``os.replace`` onto
    the final path. On any error the ``.partial`` is removed and the exception
    re-raised. Callers that see the final path can trust it is complete.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = _partial_of(path)
    try:
        with open(partial, "w", encoding="utf-8") as f:
            f.write(content)
            f.flush()
            os.fsync(f.fileno())
        os.replace(partial, path)
    except BaseException:
        try:
            partial.unlink()
        except FileNotFoundError:
            pass
        raise
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_runner.py -v`
Expected: 3 passed.

- [ ] **Step 5: Commit**

```bash
git add src/hunyuan_ocr/runner.py tests/test_runner.py
git commit -m "feat(runner): atomic write primitive

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

### Task 1.3: error records, `commit_success`, `is_complete`, `page_status`

**Files:**
- Modify: `src/hunyuan_ocr/runner.py` (append functions)
- Test: `tests/test_runner.py` (append tests)

**Interfaces:**
- Produces: `runner._error_path(pred_dir, stem, ext='.md') -> Path`; `runner.record_error(pred_dir, stem, *, image_path, backend, endpoint, exc, attempt, ts=None) -> None`; `runner.commit_success(pred_dir, stem, md, *, ext='.md') -> Path`; `runner.is_complete(pred_dir, stem, ext='.md') -> bool`; `runner.page_status(pred_dir, stem, ext='.md') -> str` (`'complete'|'failed'|'pending'`).

- [ ] **Step 1: Append failing tests** to `tests/test_runner.py`

```python
def test_record_error_writes_structured_record(tmp_path):
    try:
        raise ValueError("boom")
    except ValueError as e:
        runner.record_error(tmp_path, "stem1",
                            image_path="/x/y.png", backend="vllm",
                            endpoint="127.0.0.1:8080", exc=e, attempt=2, ts=1.5)
    rec = json.loads((tmp_path / "_errors" / "stem1.json").read_text("utf-8"))
    assert rec["exception_type"] == "ValueError"
    assert rec["exception_message"] == "boom"
    assert rec["attempt"] == 2
    assert rec["backend"] == "vllm"
    assert rec["image_path"] == "/x/y.png"


def test_commit_success_writes_md_and_clears_stale_error(tmp_path):
    try:
        raise RuntimeError("first try failed")
    except RuntimeError as e:
        runner.record_error(tmp_path, "s", image_path="i", backend="b",
                            endpoint="e", exc=e, attempt=1)
    assert not runner.is_complete(tmp_path, "s")  # has error record
    runner.commit_success(tmp_path, "s", "# real output")
    assert runner.is_complete(tmp_path, "s")
    assert not (tmp_path / "_errors" / "s.json").exists()


def test_is_complete_false_for_missing_empty_error_partial(tmp_path):
    assert not runner.is_complete(tmp_path, "missing")
    (tmp_path / "empty.md").write_text("")
    assert not runner.is_complete(tmp_path, "empty")
    (tmp_path / "err.md").write_text("ERROR: ValueError: x")
    assert not runner.is_complete(tmp_path, "err")
    (tmp_path / "good.md").write_text("# fine")
    assert runner.is_complete(tmp_path, "good")


def test_is_complete_false_if_partial_only(tmp_path):
    (tmp_path / "p.md.partial").write_text("half")
    assert not runner.is_complete(tmp_path, "p")


def test_page_status_states(tmp_path):
    assert runner.page_status(tmp_path, "n") == "pending"
    runner.commit_success(tmp_path, "ok", "x")
    assert runner.page_status(tmp_path, "ok") == "complete"
    try:
        raise ValueError("z")
    except ValueError as e:
        runner.record_error(tmp_path, "bad", image_path="i", backend="b",
                            endpoint="e", exc=e, attempt=2)
    assert runner.page_status(tmp_path, "bad") == "failed"
```

Add `import json` to the test file imports.

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_runner.py -v`
Expected: FAIL — functions not defined.

- [ ] **Step 3: Append to `src/hunyuan_ocr/runner.py`**

```python
def _error_path(pred_dir, stem: str, ext: str = ".md") -> Path:
    return Path(pred_dir) / "_errors" / f"{stem}.json"


def record_error(pred_dir, stem: str, *, image_path, backend, endpoint,
                 exc, attempt: int, ts: float | None = None) -> None:
    """Write ``_errors/<stem>.json`` (one file per page -> no concurrent-write race).

    The presence of this file means the page is FAILED. ``write_atomic`` is used
    so the record is never half-written.
    """
    ts = time.time() if ts is None else ts
    rec = {
        "image_path": str(image_path),
        "stem": stem,
        "backend": backend,
        "endpoint": str(endpoint),
        "exception_type": type(exc).__name__,
        "exception_message": str(exc),
        "attempt": attempt,
        "timestamp": ts,
    }
    write_atomic(_error_path(pred_dir, stem), json.dumps(rec, ensure_ascii=False, indent=2))


def commit_success(pred_dir, stem: str, md: str, *, ext: str = ".md") -> Path:
    """Atomically write the final prediction AND clear any stale error record.

    Preserves the invariant  COMPLETE <=> valid .md present AND no _errors/<stem>.json
    across retries: a page that failed attempt 1 then succeeded attempt 2 must not
    retain a stale error file. All success paths go through here, never raw write_atomic.
    """
    out = Path(pred_dir) / f"{stem}{ext}"
    write_atomic(out, md)
    try:
        _error_path(pred_dir, stem, ext).unlink()
    except FileNotFoundError:
        pass
    return out


def is_complete(pred_dir, stem: str, ext: str = ".md") -> bool:
    """True iff a valid prediction exists (non-empty, not ERROR:) and no unresolved error."""
    out = Path(pred_dir) / f"{stem}{ext}"
    if not out.is_file():
        return False
    try:
        if out.stat().st_size == 0:
            return False
        with open(out, "r", encoding="utf-8") as f:
            head = f.read(len(ERROR_PREFIX) + 32)
    except OSError:
        return False
    if head.lstrip().startswith(ERROR_PREFIX):
        return False
    if _error_path(pred_dir, stem, ext).exists():
        return False
    return True


def page_status(pred_dir, stem: str, ext: str = ".md") -> str:
    """'failed' | 'complete' | 'pending'."""
    if _error_path(pred_dir, stem, ext).exists():
        return "failed"
    if is_complete(pred_dir, stem, ext):
        return "complete"
    return "pending"
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_runner.py -v`
Expected: all pass (8 total).

- [ ] **Step 5: Commit**

```bash
git add src/hunyuan_ocr/runner.py tests/test_runner.py
git commit -m "feat(runner): error records, commit_success, is_complete, page_status

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

### Task 1.4: `select_todo`, `detect_stem_conflicts`, `decide_run_status`

**Files:**
- Modify: `src/hunyuan_ocr/runner.py` (append)
- Test: `tests/test_runner.py` (append)

**Interfaces:**
- Produces: `runner.select_todo(items, pred_dir, *, overwrite=False, retry_failed=False, ext='.md') -> tuple[list[tuple[str,str]], int]` where `items` is `[(stem, image_path), ...]` and return is `(todo, n_skipped)`; `runner.detect_stem_conflicts(image_paths) -> list[tuple[str, list[str]]]`; `runner.decide_run_status(final_failed, final_pending, worker_errors=0, crashed=0) -> str` (`'ok'|'failed'`).

- [ ] **Step 1: Append failing tests**

```python
def test_select_todo_default_resumes_and_retries_failed(tmp_path):
    items = [("a", "a.png"), ("b", "b.png"), ("c", "c.png"), ("d", "d.png")]
    runner.commit_success(tmp_path, "a", "ok")          # complete -> skip
    try:
        raise ValueError("x")
    except ValueError as e:
        runner.record_error(tmp_path, "b", image_path="b.png", backend="b",
                            endpoint="e", exc=e, attempt=1)  # failed -> retry
    # c pending, d pending
    todo, skipped = runner.select_todo(items, tmp_path)
    assert {s for s, _ in todo} == {"b", "c", "d"}
    assert skipped == 1


def test_select_todo_retry_failed_only(tmp_path):
    items = [("a", "a.png"), ("b", "b.png"), ("c", "c.png")]
    runner.commit_success(tmp_path, "a", "ok")
    try:
        raise ValueError("x")
    except ValueError as e:
        runner.record_error(tmp_path, "b", image_path="b.png", backend="b",
                            endpoint="e", exc=e, attempt=1)
    todo, skipped = runner.select_todo(items, tmp_path, retry_failed=True)
    assert {s for s, _ in todo} == {"b"}
    assert skipped == 2


def test_select_todo_overwrite(tmp_path):
    items = [("a", "a.png")]
    runner.commit_success(tmp_path, "a", "ok")
    todo, skipped = runner.select_todo(items, tmp_path, overwrite=True)
    assert todo == [("a", "a.png")] and skipped == 0


def test_detect_stem_conflicts(tmp_path):
    conflicts = runner.detect_stem_conflicts(["dirA/page-1.png", "dirB/page-1.png", "page-2.png"])
    assert len(conflicts) == 1
    stem, srcs = conflicts[0]
    assert stem == "page-1" and len(srcs) == 2


def test_decide_run_status():
    assert runner.decide_run_status(0, 0) == "ok"
    assert runner.decide_run_status(1, 0) == "failed"
    assert runner.decide_run_status(0, 1) == "failed"
    assert runner.decide_run_status(0, 0, worker_errors=1) == "failed"
    assert runner.decide_run_status(0, 0, crashed=1) == "failed"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_runner.py -v` → FAIL (functions undefined).

- [ ] **Step 3: Append to `runner.py`**

```python
def select_todo(items, pred_dir, *, overwrite: bool = False,
                retry_failed: bool = False, ext: str = ".md"):
    """Build the run's todo list per the resume policy.

    items: iterable of (stem, image_path). Returns (todo, n_skipped).
      default      -> skip COMPLETE; run FAILED + PENDING (failed retried across runs)
      retry_failed -> run FAILED only
      overwrite    -> run everything
    """
    todo: list[tuple[str, str]] = []
    skipped = 0
    for stem, img in items:
        st = page_status(pred_dir, stem, ext)
        if overwrite:
            todo.append((stem, img))
        elif retry_failed:
            if st == "failed":
                todo.append((stem, img))
            else:
                skipped += 1
        else:
            if st == "complete":
                skipped += 1
            else:
                todo.append((stem, img))
    return todo, skipped


def detect_stem_conflicts(image_paths) -> list:
    """Return [(stem, [source_paths...])] for any stem produced by >1 distinct image."""
    seen: dict[str, list[str]] = {}
    for p in image_paths:
        stem = Path(p).stem
        seen.setdefault(stem, []).append(str(p))
    return [(stem, srcs) for stem, srcs in seen.items() if len(srcs) > 1]


def decide_run_status(final_failed: int, final_pending: int,
                      worker_errors: int = 0, crashed: int = 0) -> str:
    """Pure exit decision shared by both drivers."""
    if final_failed or final_pending or worker_errors or crashed:
        return "failed"
    return "ok"
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_runner.py -v` → all pass.

- [ ] **Step 5: Commit**

```bash
git add src/hunyuan_ocr/runner.py tests/test_runner.py
git commit -m "feat(runner): select_todo, detect_stem_conflicts, decide_run_status

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

### Task 1.5: `aggregate_errors`, `safe_argv`, `write_run_manifest`

**Files:**
- Modify: `src/hunyuan_ocr/runner.py` (append)
- Test: `tests/test_runner.py` (append)

**Interfaces:**
- Produces: `runner.aggregate_errors(pred_dir, out_name='_errors.jsonl') -> Path`; `runner.safe_argv(argv=None) -> list[str]`; `runner.write_run_manifest(pred_dir, *, backend, model, model_revision=None, command=None, counts=None, ports=None, gpu_ids=None, max_pixels=None, max_tokens=None, status='ok', extra=None) -> Path`.

- [ ] **Step 1: Append failing tests**

```python
def test_aggregate_errors_concatenates_records(tmp_path):
    for stem, msg in [("a", "e1"), ("b", "e2")]:
        try:
            raise ValueError(msg)
        except ValueError as e:
            runner.record_error(tmp_path, stem, image_path=stem + ".png",
                                backend="b", endpoint="e", exc=e, attempt=1)
    out = runner.aggregate_errors(tmp_path)
    lines = [json.loads(l) for l in out.read_text("utf-8").splitlines() if l.strip()]
    assert {l["exception_message"] for l in lines} == {"e1", "e2"}


def test_safe_argv_redacts_secrets():
    argv = ["--gt-json", "x.json", "--hf-token", "SECRET123",
            "--api-key=TOPSECRET", "--ports", "8000"]
    redacted = runner.safe_argv(argv)
    assert "SECRET123" not in redacted
    assert "TOPSECRET" not in redacted
    assert "--gt-json" in redacted and "x.json" in redacted and "8000" in redacted


def test_safe_argv_no_false_positive_on_monkey():
    # 'monkey' contains 'key' substring but is not a secret flag
    redacted = runner.safe_argv(["--monkey", "tail"])
    assert redacted == ["--monkey", "tail"]


def test_write_run_manifest_structure_and_no_secret(tmp_path):
    p = runner.write_run_manifest(
        tmp_path, backend="vllm", model="HYVL",
        counts={"expected": 3, "succeeded": 2, "failed": 1, "skipped": 0},
        ports=[8000, 8001], max_pixels=0, max_tokens=32768, status="failed")
    m = json.loads(p.read_text("utf-8"))
    assert m["backend"] == "vllm" and m["status"] == "failed"
    assert m["counts"] == {"expected": 3, "succeeded": 2, "failed": 1, "skipped": 0}
    assert m["ports"] == [8000, 8001]
    assert "torch" in m["env"]  # current env has torch
```

- [ ] **Step 2: Run tests to verify they fail** → FAIL (undefined).

- [ ] **Step 3: Append to `runner.py`**

```python
def aggregate_errors(pred_dir, out_name: str = "_errors.jsonl") -> Path:
    """Concatenate ``_errors/*.json`` into ``_errors.jsonl``. Call ONCE from main
    after all workers join (single writer). Uses write_atomic for safety."""
    edir = Path(pred_dir) / "_errors"
    out = Path(pred_dir) / out_name
    rows = []
    if edir.is_dir():
        for f in sorted(edir.glob("*.json")):
            try:
                rows.append(json.loads(f.read_text(encoding="utf-8")))
            except (OSError, json.JSONDecodeError):
                continue
    body = "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows)
    write_atomic(out, body)
    return out


_SECRET_FLAGS = {
    "--token", "--api-key", "--apikey", "--key", "--password",
    "--secret", "--hf-token", "--hugging-face-token", "--venv-python",
}


def safe_argv(argv=None) -> list[str]:
    """Return argv with secret-bearing flag values redacted (exact flag match only)."""
    argv = list(sys.argv[1:] if argv is None else argv)
    out: list[str] = []
    i = 0
    while i < len(argv):
        tok = argv[i]
        if "=" in tok and tok.split("=", 1)[0] in _SECRET_FLAGS:
            out.append(f"{tok.split('=', 1)[0]}=<redacted>")
        elif tok in _SECRET_FLAGS and i + 1 < len(argv):
            out.append(tok); out.append("<redacted>"); i += 1
        else:
            out.append(tok)
        i += 1
    return out


def _git_head(repo: str = ".") -> str | None:
    try:
        cp = subprocess.run(["git", "-C", str(repo), "rev-parse", "HEAD"],
                            capture_output=True, text=True, timeout=10)
        if cp.returncode == 0:
            return cp.stdout.strip() or None
    except (OSError, subprocess.SubprocessError):
        pass
    return None


def _env_versions() -> dict:
    v: dict[str, str] = {}
    try:
        import torch  # type: ignore
        v["torch"] = getattr(torch, "__version__", None)
        hip = getattr(getattr(torch, "version", None), "hip", None)
        if hip:
            v["hip"] = hip
    except Exception:
        pass
    try:
        import transformers  # type: ignore
        v["transformers"] = getattr(transformers, "__version__", None)
    except Exception:
        pass
    try:
        import vllm  # type: ignore
        v["vllm"] = getattr(vllm, "__version__", None)
    except Exception:
        pass
    return {k: val for k, val in v.items() if val}


def write_run_manifest(pred_dir, *, backend: str, model: str,
                       model_revision: str | None = None,
                       command: list[str] | None = None,
                       counts: dict | None = None, ports=None, gpu_ids=None,
                       max_pixels=None, max_tokens=None, status: str = "ok",
                       extra: dict | None = None) -> Path:
    """Write ``run_manifest.json`` (atomic). No secrets (command via safe_argv)."""
    counts = counts or {}
    manifest = {
        "repo_commit": _git_head(),
        "backend": backend,
        "model": model,
        "model_revision": model_revision,
        "command": safe_argv() if command is None else command,
        "timestamp": time.time(),
        "counts": {
            "expected": counts.get("expected"),
            "succeeded": counts.get("succeeded"),
            "failed": counts.get("failed"),
            "skipped": counts.get("skipped"),
        },
        "ports": ports,
        "gpu_ids": gpu_ids,
        "pixel_cap": max_pixels,
        "max_tokens": max_tokens,
        "env": _env_versions(),
        "status": status,
    }
    if extra:
        manifest.update(extra)
    out = Path(pred_dir) / "run_manifest.json"
    write_atomic(out, json.dumps(manifest, ensure_ascii=False, indent=2))
    return out
```

Note: `--venv-python` is redacted because it is a machine-local absolute path, not because it is a credential — keeping it out of the portable manifest. Document this in the comment if desired (optional).

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_runner.py -v` → all pass.

- [ ] **Step 5: Commit**

```bash
git add src/hunyuan_ocr/runner.py tests/test_runner.py
git commit -m "feat(runner): aggregate_errors, safe_argv, write_run_manifest

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

### Task 1.6: `validation.validate_predictions`

**Files:**
- Create: `src/hunyuan_ocr/validation.py`
- Test: `tests/test_validation.py`

**Interfaces:**
- Consumes: `runner.ERROR_PREFIX`, `runner._OWN_ARTIFACTS` (re-exported).
- Produces: `validation.Problem` (dataclass: severity, code, message, detail); `validation.Report` (dataclass: expected, valid, problems; props `.ok`, `.ok_strict`, `.errors()`, `.warnings()`); `validation.validate_predictions(gt_json, pred_dir, *, strict=True) -> Report`.

- [ ] **Step 1: Write failing tests** (`tests/test_validation.py`)

```python
# tests/test_validation.py
import json
from pathlib import Path
from hunyuan_ocr import validation


def _gt(tmp_path, stems):
    pages = [{"page_info": {"image_path": f"{s}.png"}} for s in stems]
    p = tmp_path / "gt.json"
    p.write_text(json.dumps(pages), encoding="utf-8")
    return p


def test_clean_dir_passes(tmp_path):
    gt = _gt(tmp_path, ["a", "b"])
    pred = tmp_path / "pred"; pred.mkdir()
    (pred / "a.md").write_text("ok a"); (pred / "b.md").write_text("ok b")
    r = validation.validate_predictions(gt, pred)
    assert r.ok and r.ok_strict and r.expected == 2 and r.valid == 2


def test_missing_pages(tmp_path):
    gt = _gt(tmp_path, ["a", "b", "c"])
    pred = tmp_path / "pred"; pred.mkdir()
    (pred / "a.md").write_text("ok")
    r = validation.validate_predictions(gt, pred)
    assert not r.ok
    codes = {p.code for p in r.errors()}
    assert "missing" in codes


def test_empty_error_partial_markers(tmp_path):
    gt = _gt(tmp_path, ["a", "b", "c", "d"])
    pred = tmp_path / "pred"; pred.mkdir()
    (pred / "a.md").write_text("")                       # empty
    (pred / "b.md").write_text("ERROR: ValueError: x")   # error marker
    (pred / "c.md").write_text("ok")
    (pred / "d.md.partial").write_text("half")           # leftover partial
    r = validation.validate_predictions(gt, pred)
    assert not r.ok
    codes = {p.code for p in r.errors()}
    assert {"empty", "error_marker", "partial", "missing"} <= codes


def test_unresolved_error_record(tmp_path):
    gt = _gt(tmp_path, ["a"])
    pred = tmp_path / "pred"; pred.mkdir()
    (pred / "_errors").mkdir()
    (pred / "_errors" / "a.json").write_text(json.dumps({"stem": "a"}))
    r = validation.validate_predictions(gt, pred)
    assert not r.ok and "unresolved_error" in {p.code for p in r.errors()}


def test_unexpected_file_warning(tmp_path):
    gt = _gt(tmp_path, ["a"])
    pred = tmp_path / "pred"; pred.mkdir()
    (pred / "a.md").write_text("ok")
    (pred / "junk.txt").write_text("??")
    r = validation.validate_predictions(gt, pred)
    assert r.ok is True            # no hard error
    assert r.ok_strict is False    # warning present under strict
    assert "unexpected_file" in {p.code for p in r.warnings()}


def test_duplicate_stem_in_gt(tmp_path):
    gt = _gt(tmp_path, ["dup", "dup"])
    pred = tmp_path / "pred"; pred.mkdir()
    r = validation.validate_predictions(gt, pred)
    assert not r.ok and "duplicate_stem" in {p.code for p in r.errors()}
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_validation.py -v` → FAIL (no module).

- [ ] **Step 3: Create `src/hunyuan_ocr/validation.py`**

```python
"""Pre-score validation of a prediction directory against OmniDocBench GT.

Pure function: read GT json + pred dir -> structured Report. No GPU, no model.
A non-clean report blocks scoring (see scripts/validate_predictions.py and the
gate in scripts/score_predictions.py).
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

from .runner import ERROR_PREFIX

# Files/dirs the runners legitimately emit alongside predictions.
_OWN_ARTIFACTS = {"_errors", "_errors.jsonl", "run_manifest.json"}


@dataclass
class Problem:
    severity: str   # "error" | "warning"
    code: str
    message: str
    detail: object = None


@dataclass
class Report:
    expected: int
    valid: int
    problems: list = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not any(p.severity == "error" for p in self.problems)

    @property
    def ok_strict(self) -> bool:
        return not self.problems

    def errors(self):
        return [p for p in self.problems if p.severity == "error"]

    def warnings(self):
        return [p for p in self.problems if p.severity == "warning"]


def _gt_stems(gt_json) -> tuple[list[str], list[Problem]]:
    with open(gt_json, encoding="utf-8") as f:
        pages = json.load(f)
    problems: list[Problem] = []
    seen: dict[str, int] = {}
    stems: list[str] = []
    for p in pages:
        rel = p["page_info"]["image_path"]
        stem = Path(rel).stem
        seen[stem] = seen.get(stem, 0) + 1
        stems.append(stem)
    for stem, n in seen.items():
        if n > 1:
            problems.append(Problem("error", "duplicate_stem",
                                    f"GT maps {n} pages to stem '{stem}'",
                                    {"stem": stem, "count": n}))
    return stems, problems


def validate_predictions(gt_json, pred_dir, *, strict: bool = True) -> Report:
    pred_dir = Path(pred_dir)
    stems, problems = _gt_stems(gt_json)
    expected = len(stems)

    valid = 0
    missing: list[str] = []
    error_markers: list[str] = []
    for stem in stems:
        out = pred_dir / f"{stem}.md"
        if not out.is_file():
            missing.append(stem)
            continue
        try:
            if out.stat().st_size == 0:
                problems.append(Problem("error", "empty", f"'{stem}.md' is empty", {"stem": stem}))
                continue
            with open(out, "r", encoding="utf-8") as f:
                head = f.read(len(ERROR_PREFIX) + 32)
        except OSError:
            problems.append(Problem("error", "empty", f"'{stem}.md' unreadable", {"stem": stem}))
            continue
        if head.lstrip().startswith(ERROR_PREFIX):
            error_markers.append(stem)
            continue
        valid += 1

    for stem in missing:
        problems.append(Problem("error", "missing", f"'{stem}.md' missing", {"stem": stem}))
    for stem in error_markers:
        problems.append(Problem("error", "error_marker",
                                f"'{stem}.md' starts with 'ERROR:'", {"stem": stem}))

    for p in sorted(pred_dir.glob("*.partial")):
        problems.append(Problem("error", "partial",
                                f"leftover partial '{p.name}'", {"file": p.name}))

    edir = pred_dir / "_errors"
    if edir.is_dir():
        for ef in sorted(edir.glob("*.json")):
            problems.append(Problem("error", "unresolved_error",
                                    f"unresolved error record '_errors/{ef.name}'",
                                    {"stem": ef.stem}))

    if pred_dir.is_dir():
        for entry in sorted(pred_dir.iterdir()):
            if entry.is_dir():
                if entry.name not in _OWN_ARTIFACTS:
                    problems.append(Problem("warning", "unexpected_dir",
                                            f"unexpected dir '{entry.name}/'", {"name": entry.name}))
                continue
            if entry.name in _OWN_ARTIFACTS:
                continue
            if entry.name.endswith(".md") or entry.name.endswith(".partial"):
                continue
            problems.append(Problem("warning", "unexpected_file",
                                    f"unexpected file '{entry.name}'", {"name": entry.name}))

    return Report(expected=expected, valid=valid, problems=problems)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_validation.py -v` → all pass.

- [ ] **Step 5: Commit**

```bash
git add src/hunyuan_ocr/validation.py tests/test_validation.py
git commit -m "feat(validation): pure pre-score validation -> Report

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

### Task 1.7: `scripts/validate_predictions.py` CLI

**Files:**
- Create: `scripts/validate_predictions.py`
- Test: `tests/test_validation.py` (append a subprocess test) — or manual run.

**Interfaces:** CLI exit 0 iff clean (under `--strict`, default); `--lenient` downgrades warnings to non-fatal.

- [ ] **Step 1: Create the CLI**

```python
#!/usr/bin/env python3
"""Validate a prediction directory against OmniDocBench GT before scoring.

Exit 0 iff no hard errors (and, under --strict [default], no warnings).

Usage:
  python scripts/validate_predictions.py --gt-json GT.json --pred-dir ./predictions
"""
from __future__ import annotations
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from hunyuan_ocr.validation import validate_predictions  # noqa: E402


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--gt-json", required=True)
    p.add_argument("--pred-dir", required=True)
    p.add_argument("--strict", action="store_true", default=True,
                   help="warnings are fatal (default)")
    p.add_argument("--lenient", action="store_true",
                   help="warnings are non-fatal")
    args = p.parse_args()
    strict = args.strict and not args.lenient

    r = validate_predictions(args.gt_json, args.pred_dir, strict=strict)
    print(f"expected={r.expected} valid={r.valid} "
          f"errors={len(r.errors())} warnings={len(r.warnings())}")
    for prob in r.problems:
        tag = "ERROR" if prob.severity == "error" else "WARN "
        print(f"  [{tag}] {prob.code}: {prob.message}")
    ok = r.ok_strict if strict else r.ok
    print("RESULT: PASS" if ok else "RESULT: FAIL")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Manual smoke test (CPU)**

```bash
cd /workspace/HunyuanOCR-ROCm
tmp=$(mktemp -d); pred=$tmp/pred; mkdir -p "$pred/_errors"
printf '[]' > "$tmp/gt.json"            # empty GT
echo 'ok' > "$pred/a.md"
echo 'ERROR: boom' > "$pred/b.md"
: > "$pred/empty.md"
python scripts/validate_predictions.py --gt-json "$tmp/gt.json" --pred-dir "$pred"; echo "rc=$?"
# expect: RESULT: FAIL, rc=1 (b=error_marker, empty=empty; a is unexpected under empty GT)
rm -rf "$tmp"
```

- [ ] **Step 3: Commit**

```bash
git add scripts/validate_predictions.py
git commit -m "feat(validate_predictions): pre-score validation CLI

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

### Task 1.8: `score_predictions.py` validation gate

**Files:**
- Modify: `scripts/score_predictions.py`
- Test: `tests/test_score_gate.py`

**Interfaces:**
- Consumes: `validation.validate_predictions`.
- Produces: `--skip-validation` flag; on invalid dir, exits before calling `scoring.run_scorer`.

- [ ] **Step 1: Write failing test** (`tests/test_score_gate.py`)

```python
# tests/test_score_gate.py
import json
import sys
from pathlib import Path


def test_scorer_refuses_invalid_dir(tmp_path, monkeypatch):
    # invalid pred dir: a page missing
    gt = tmp_path / "gt.json"
    gt.write_text(json.dumps([{"page_info": {"image_path": "a.png"}},
                              {"page_info": {"image_path": "b.png"}}]), "utf-8")
    pred = tmp_path / "pred"; pred.mkdir()
    (pred / "a.md").write_text("ok")   # b.md missing

    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
    import score_predictions as sp

    called = {"n": 0}
    def must_not_run(*a, **k):
        called["n"] += 1
        raise AssertionError("scorer must not run on invalid dir")
    monkeypatch.setattr(sp.scoring, "run_scorer", must_not_run)
    monkeypatch.setattr(sp.scoring, "parse_run_summary", lambda *a, **k: {})

    import pytest
    with pytest.raises(SystemExit):
        sp.main_with_args(["--pred-dir", str(pred), "--gt-json", str(gt)])
    assert called["n"] == 0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_score_gate.py -v` → FAIL (`main_with_args` doesn't exist; scorer runs).

- [ ] **Step 3: Refactor `scripts/score_predictions.py`**

Replace the body so `main()` parses and delegates to `main_with_args(argv)`, and validation runs first:

```python
#!/usr/bin/env python3
"""Score a predictions dir against OmniDocBench v1.6 and print the score table.

VALIDATES the prediction dir first (missing/empty/ERROR/.partial/unresolved-error
pages block scoring). Use --skip-validation ONLY for debugging.

Usage:
  python scripts/score_predictions.py \
      --pred-dir ./predictions --gt-json /path/to/OmniDocBench.json [--label x]
"""
from __future__ import annotations
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from hunyuan_ocr import scoring  # noqa: E402
from hunyuan_ocr.validation import validate_predictions  # noqa: E402


def main_with_args(argv):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--pred-dir", required=True)
    p.add_argument("--gt-json", required=True)
    p.add_argument("--label", default="backend")
    p.add_argument("--omnidocbench-repo", default=scoring.DEFAULT_OMNIDOCBENCH_REPO)
    p.add_argument("--venv-python", default=scoring.DEFAULT_VENV_PYTHON)
    p.add_argument("--skip-validation", action="store_true",
                   help="DANGEROUS: bypass pre-score validation")
    args = p.parse_args(argv)

    if args.skip_validation:
        print("WARNING: validation bypassed -- score may be invalid", file=sys.stderr)
    else:
        rep = validate_predictions(args.gt_json, args.pred_dir, strict=True)
        if not rep.ok_strict:
            print(f"[validation] {len(rep.errors())} error(s), "
                  f"{len(rep.warnings())} warning(s); refusing to score:",
                  file=sys.stderr)
            for prob in rep.problems:
                tag = "ERROR" if prob.severity == "error" else "WARN"
                print(f"  [{tag}] {prob.code}: {prob.message}", file=sys.stderr)
            sys.exit(
                "[error] predictions invalid; fix them or re-run the driver "
                "(use --skip-validation to override at your own risk)")

    cfg_path = Path(args.pred_dir) / "_eval_config.yaml"
    scoring.write_eval_config(gt_json=args.gt_json, pred_dir=args.pred_dir, out_yaml=cfg_path)
    res = scoring.run_scorer(omnidocbench_repo=args.omnidocbench_repo,
                             config_yaml=str(cfg_path), venv_python=args.venv_python)
    if res.returncode != 0:
        print(res.stdout[-4000:]); print(res.stderr[-4000:], file=sys.stderr)
        sys.exit(f"[error] scorer failed (rc={res.returncode})")

    save_name = f"{Path(args.pred_dir).name}_quick_match"
    s = scoring.parse_run_summary(Path(args.omnidocbench_repo) / "result", save_name)

    def fmt(v, pct=False):
        if v is None:
            return "n/a"
        return f"{v * 100:.2f}" if pct else f"{v:.4f}"

    print(f"\n=== {args.label} -- OmniDocBench v1.6 ===")
    ov = s["overall"]
    print(f"  Overall          : {'n/a (CDM missing on this subset)' if ov is None else f'{ov:.2f}'}")
    print(f"  text  EditDist   : {fmt(s['text_edit_dist'])}   -> {fmt(s['text_edit_dist'], pct=True)}")
    print(f"  formula CDM      : {fmt(s['formula_cdm'])}   -> {fmt(s['formula_cdm'], pct=True)}")
    print(f"  table  TEDS      : {fmt(s['table_teds'])}   -> {fmt(s['table_teds'], pct=True)}")
    print(f"  order  EditDist  : {fmt(s['reading_order_edit'])}")
    recomputed = scoring.overall_score({"text_edit_dist": s["text_edit_dist"],
                                        "formula_cdm": s["formula_cdm"],
                                        "table_teds": s["table_teds"]})
    print(f"  (overall recomputed: {'n/a' if recomputed is None else f'{recomputed:.2f}'})")


def main():
    main_with_args(sys.argv[1:])


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_score_gate.py -v` → PASS (scorer not called; SystemExit raised).

- [ ] **Step 5: Commit**

```bash
git add scripts/score_predictions.py tests/test_score_gate.py
git commit -m "feat(score): validate predictions before invoking the scorer

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

### Task 1.9: Rewrite `run_phase2_vllm.py` (real failure propagation, retries, manifest)

**Files:**
- Modify: `scripts/run_phase2_vllm.py` (full rewrite of `main` + `work`)
- Test: `tests/test_phase2_no_gpu.py` (end-to-end on CPU with a fake `infer_one`)

**Interfaces:**
- Consumes: `runner.detect_stem_conflicts`, `runner.select_todo`, `runner.commit_success`, `runner.record_error`, `runner.aggregate_errors`, `runner.page_status`, `runner.decide_run_status`, `runner.write_run_manifest`.
- Produces: a driver that exits non-zero on any FAILED/PENDING page; writes `run_manifest.json`; honors `--max-retries/--retry-backoff/--overwrite/--retry-failed`.

- [ ] **Step 1: Write the CPU end-to-end test** (`tests/test_phase2_no_gpu.py`)

This test drives the real driver with a **fake** `infer_one` (monkeypatched on the driver module) so no server/GPU is needed. It covers the 10-step acceptance: 3 pages, 1 fails → 2 valid `.md`, 1 `_errors/<stem>.json`, non-zero exit; re-run skips the 2 complete and retries the failed (which now succeeds) → exit 0.

```python
# tests/test_phase2_no_gpu.py
import json
import os
import sys
from pathlib import Path

import pytest


def _make_gt(tmp_path, stems):
    gt = tmp_path / "gt.json"
    gt.write_text(json.dumps([{"page_info": {"image_path": f"{s}.png"}} for s in stems]),
                  encoding="utf-8")
    img = tmp_path / "images"; img.mkdir()
    for s in stems:
        (img / f"{s}.png").write_bytes(b"x")
    return gt, img


def _import_driver():
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
    import run_phase2_vllm as drv
    return drv


def test_phase2_two_ok_one_failed_then_rerun(tmp_path, monkeypatch):
    drv = _import_driver()
    gt, img = _make_gt(tmp_path, ["a", "b", "c"])
    pred = tmp_path / "pred"

    # fake infer: succeed for a,b; raise for c on attempt 1, succeed on attempt 2 (rerun)
    state = {"c_tries": 0}

    def fake_infer(client, image_path, prompt, *, model, max_pixels):
        stem = Path(image_path).stem
        if stem == "c":
            state["c_tries"] += 1
            if state["c_tries"] == 1:
                raise RuntimeError("server 500")
        return f"# output for {stem}"

    monkeypatch.setattr(drv, "infer_one", fake_infer)
    # The driver imported OpenAI into its own namespace at module load, so patch
    # drv.OpenAI (not openai.OpenAI) to avoid even constructing a real client.
    monkeypatch.setattr(drv, "OpenAI", lambda *a, **k: object())

    # run 1: c fails (max-retries=1 -> one attempt) -> non-zero exit
    with pytest.raises(SystemExit) as ei:
        drv.main_with_args(["--gt-json", str(gt), "--images-dir", str(img),
                            "--pred-dir", str(pred), "--ports", "9999",
                            "--concurrency", "2", "--max-retries", "1"])
    assert ei.value.code != 0
    assert (pred / "a.md").exists() and (pred / "b.md").exists()
    assert not (pred / "c.md").exists()
    assert (pred / "_errors" / "c.json").exists()
    assert (pred / "run_manifest.json").exists()

    # run 2: default resume skips a,b; retries c (now succeeds) -> exit 0
    drv.main_with_args(["--gt-json", str(gt), "--images-dir", str(img),
                        "--pred-dir", str(pred), "--ports", "9999",
                        "--concurrency", "2", "--max-retries", "1"])
    assert (pred / "c.md").read_text("utf-8") == "# output for c"
    # stale error record must be cleared on success
    assert not (pred / "_errors" / "c.json").exists()


def test_phase2_conflict_aborts(tmp_path, monkeypatch):
    drv = _import_driver()
    # two distinct image paths with same stem
    gt = tmp_path / "gt.json"
    gt.write_text(json.dumps([
        {"page_info": {"image_path": "dir1/x.png"}},
        {"page_info": {"image_path": "dir2/x.png"}},
    ]), encoding="utf-8")
    (tmp_path / "dir1").mkdir(); (tmp_path / "dir1" / "x.png").write_bytes(b"")
    (tmp_path / "dir2").mkdir(); (tmp_path / "dir2" / "x.png").write_bytes(b"")
    with pytest.raises(SystemExit):
        drv.main_with_args(["--gt-json", str(gt), "--images-dir", str(tmp_path),
                            "--pred-dir", str(tmp_path / "pred"),
                            "--ports", "9999", "--max-retries", "1"])
```

- [ ] **Step 2: Run test to verify it fails** → FAIL (`main_with_args` absent).

- [ ] **Step 3: Rewrite `scripts/run_phase2_vllm.py`**

```python
#!/usr/bin/env python3
"""Phase-2 driver: run HunyuanOCR-1.5 via an OpenAI-compatible server over OmniDocBench.

One <stem>.md per page, written atomically; errors recorded to _errors/<stem>.json.
Resumable (skips only COMPLETE pages; FAILED/PENDING are retried). Exits non-zero
on any page that ends up FAILED or PENDING, or on any unhandled worker exception.

Usage:
  # start servers first (one/GPU), then:
  python scripts/run_phase2_vllm.py --gt-json GT.json --images-dir images \
      --pred-dir ./predictions --ports 8081,8082,8083,8084 --model HYVL --concurrency 16
"""
from __future__ import annotations
import argparse
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from openai import OpenAI  # noqa: E402

from hunyuan_ocr import runner  # noqa: E402
from hunyuan_ocr.backends.vllm_client import infer_one  # noqa: E402
from hunyuan_ocr.contract import CONTRACT  # noqa: E402


def _load_pages(gt_json, images_dir, limit=0):
    pages = json.load(open(gt_json, encoding="utf-8"))
    if limit:
        pages = pages[:limit]
    out = []
    for pg in pages:
        rel = pg["page_info"]["image_path"]
        out.append((Path(rel).stem, os.path.join(images_dir, rel)))
    return out


def main_with_args(argv):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--gt-json", required=True)
    p.add_argument("--images-dir", required=True)
    p.add_argument("--pred-dir", required=True)
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--ports", default="8000", help="comma-separated server ports")
    p.add_argument("--model", default="tencent/HunyuanOCR")
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--concurrency", type=int, default=24)
    p.add_argument("--max-pixels", type=int, default=0,
                   help="client-side ViT cap (0 = uncapped)")
    p.add_argument("--max-retries", type=int, default=2)
    p.add_argument("--retry-backoff", type=float, default=2.0)
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--retry-failed", action="store_true",
                   help="scope this run to FAILED pages only")
    args = p.parse_args(argv)

    pages = _load_pages(args.gt_json, args.images_dir, args.limit)
    os.makedirs(args.pred_dir, exist_ok=True)

    conflicts = runner.detect_stem_conflicts([img for _, img in pages])
    if conflicts:
        for stem, srcs in conflicts:
            print(f"[conflict] stem '{stem}' from {len(srcs)} images: {srcs}", file=sys.stderr)
        sys.exit("[fatal] output filename conflict(s); refusing to overwrite")

    ports = [int(x) for x in args.ports.split(",") if x.strip()]
    clients = [OpenAI(api_key="EMPTY", base_url=f"http://{args.host}:{pt}/v1", timeout=3600.0)
               for pt in ports]
    max_pixels = args.max_pixels or None

    todo, skipped = runner.select_todo(pages, args.pred_dir,
                                       overwrite=args.overwrite,
                                       retry_failed=args.retry_failed)
    print(f"[info] {len(todo)} to do ({skipped} skipped) across ports {ports}", flush=True)

    def work(item):
        idx, (stem, img) = item
        last_exc, ep = None, f"{args.host}:{ports[idx % len(ports)]}"
        attempt = 0
        for attempt in range(1, args.max_retries + 1):
            client = clients[(idx + attempt - 1) % len(clients)]
            ep = f"{args.host}:{ports[(idx + attempt - 1) % len(ports)]}"
            try:
                md = infer_one(client, img, CONTRACT.prompt,
                               model=args.model, max_pixels=max_pixels)
                runner.commit_success(args.pred_dir, stem, md)
                return {"stem": stem, "status": "complete"}
            except Exception as e:  # bounded retry; recorded if exhausted
                last_exc = e
                if attempt < args.max_retries:
                    time.sleep(args.retry_backoff * (2 ** (attempt - 1)))
        runner.record_error(args.pred_dir, stem, image_path=img, backend="vllm",
                            endpoint=ep, exc=last_exc, attempt=attempt)
        return {"stem": stem, "status": "failed", "error": str(last_exc)}

    results = []
    with ThreadPoolExecutor(max_workers=max(1, args.concurrency)) as ex:
        futs = [ex.submit(work, it) for it in enumerate(todo)]
        for f in as_completed(futs):
            res = f.result()  # propagates any UNEXPECTED exception to main -> abort
            results.append(res)
            if len(results) % 20 == 0:
                print(f"[info] {len(results)}/{len(todo)}", flush=True)

    runner.aggregate_errors(args.pred_dir)

    final_complete = sum(1 for s, _ in pages if runner.page_status(args.pred_dir, s) == "complete")
    final_failed = sum(1 for s, _ in pages if runner.page_status(args.pred_dir, s) == "failed")
    final_pending = len(pages) - final_complete - final_failed
    status = runner.decide_run_status(final_failed, final_pending)

    runner.write_run_manifest(args.pred_dir, backend="vllm", model=args.model,
                              counts={"expected": len(pages), "succeeded": final_complete,
                                      "failed": final_failed, "skipped": skipped},
                              ports=ports, max_pixels=args.max_pixels,
                              max_tokens=32768, status=status)
    print(f"[summary] expected={len(pages)} complete={final_complete} failed={final_failed} "
          f"pending={final_pending} skipped={skipped} -> {args.pred_dir}", flush=True)
    if status != "ok":
        sys.exit(f"[error] {final_failed} page(s) failed, {final_pending} pending; see _errors/")
    print("[done] all pages complete", flush=True)


def main():
    main_with_args(sys.argv[1:])


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_phase2_no_gpu.py -v` → 2 passed.

- [ ] **Step 5: Commit**

```bash
git add scripts/run_phase2_vllm.py tests/test_phase2_no_gpu.py
git commit -m "feat(phase2): atomic output, error records, retries, manifest, real failure propagation

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

### Task 1.10: Rewrite `run_phase1_transformers.py` (worker exit codes, queue, manifest)

**Files:**
- Modify: `scripts/run_phase1_transformers.py` (full rewrite)
- Test: the shared pure helpers (`select_todo`, `decide_run_status`, `page_status`) are already covered by `tests/test_runner.py`. The multiprocessing worker (imports torch, runs on GPU) is **not** CPU-testable; that gap is documented in §"Open items".

**Interfaces:** Same runner primitives as Task 1.9; plus a `multiprocessing.Queue` for `(gpu, kind, stem?)` worker→main messages; main joins + checks `exitcode`.

- [ ] **Step 1: Rewrite `scripts/run_phase1_transformers.py`**

```python
#!/usr/bin/env python3
"""Phase-1 driver: run HunyuanOCR-1.5 (transformers) over OmniDocBench pages.

One spawned worker process per GPU, sharded. One <stem>.md per page, written
atomically; errors to _errors/<stem>.json. Resumable. Exits non-zero on any
worker crash, model-load failure, or page that ends up FAILED/PENDING.

Usage:
  python scripts/run_phase1_transformers.py --gt-json GT.json --images-dir images \
      --pred-dir ./predictions --model /path/to/HunyuanOCR --gpu-ids 0,1,2 [--limit N]
"""
from __future__ import annotations
import argparse
import json
import multiprocessing as mp
import os
import queue as _queue
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from hunyuan_ocr import runner  # noqa: E402


def _load_page_list(gt_json, images_dir, limit=None):
    pages = json.load(open(gt_json, encoding="utf-8"))
    if limit:
        pages = pages[:limit]
    return [(Path(p["page_info"]["image_path"]).stem,
             os.path.join(images_dir, p["page_info"]["image_path"])) for p in pages]


def _shard(items, n):
    k = -(-len(items) // n)
    return [items[i:i + k] for i in range(0, len(items), k)]


def _worker(gpu_id, chunk, args_dict, out_q):
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
    try:
        from hunyuan_ocr.backends.transformers import load_model_and_processor, infer_one
        from hunyuan_ocr.contract import CONTRACT
    except Exception as e:  # import failure (e.g. missing torch)
        out_q.put({"gpu": gpu_id, "kind": "worker_error", "msg": f"import failed: {e}"})
        return
    a = argparse.Namespace(**args_dict)
    try:
        print(f"[GPU {gpu_id}] loading model ...", flush=True)
        t0 = time.time()
        model, processor = load_model_and_processor(a.model, device="cuda:0")
        print(f"[GPU {gpu_id}] model ready in {time.time()-t0:.1f}s", flush=True)
    except Exception as e:
        out_q.put({"gpu": gpu_id, "kind": "worker_error",
                   "msg": f"model load failed: {type(e).__name__}: {e}"})
        return

    os.makedirs(a.pred_dir, exist_ok=True)
    todo, skipped = runner.select_todo(chunk, a.pred_dir,
                                       overwrite=a.overwrite, retry_failed=a.retry_failed)
    for _ in range(skipped):
        out_q.put({"gpu": gpu_id, "kind": "skip"})

    for stem, img in todo:
        last_exc, attempt = None, 0
        for attempt in range(1, a.max_retries + 1):
            try:
                md = infer_one(model, processor, img, CONTRACT.prompt, device="cuda:0")
                runner.commit_success(a.pred_dir, stem, md)
                out_q.put({"gpu": gpu_id, "kind": "complete", "stem": stem})
                break
            except Exception as e:
                last_exc = e
                if attempt < a.max_retries:
                    time.sleep(a.retry_backoff * (2 ** (attempt - 1)))
        else:
            runner.record_error(a.pred_dir, stem, image_path=img, backend="transformers",
                                endpoint=f"gpu{gpu_id}", exc=last_exc, attempt=attempt)
            out_q.put({"gpu": gpu_id, "kind": "failed", "stem": stem})
    out_q.put({"gpu": gpu_id, "kind": "worker_done"})


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--gt-json", required=True)
    p.add_argument("--images-dir", required=True)
    p.add_argument("--pred-dir", required=True)
    p.add_argument("--model", default="/root/models/HunyuanOCR")
    p.add_argument("--gpu-ids", default="0,1,2")
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--max-retries", type=int, default=2)
    p.add_argument("--retry-backoff", type=float, default=2.0)
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--retry-failed", action="store_true")
    args = p.parse_args()

    pages = _load_page_list(args.gt_json, args.images_dir, args.limit)
    conflicts = runner.detect_stem_conflicts([img for _, img in pages])
    if conflicts:
        for stem, srcs in conflicts:
            print(f"[conflict] stem '{stem}' from {len(srcs)} images: {srcs}", file=sys.stderr)
        sys.exit("[fatal] output filename conflict(s)")

    os.makedirs(args.pred_dir, exist_ok=True)
    gpu_ids = [int(x) for x in args.gpu_ids.split(",") if x.strip()]
    chunks = _shard(pages, len(gpu_ids))
    print(f"[info] {len(pages)} pages across GPUs {gpu_ids}: {[len(c) for c in chunks]}", flush=True)

    ctx = mp.get_context("spawn")
    out_q = ctx.Queue()
    procs = [ctx.Process(target=_worker, args=(gid, chunks[i], vars(args), out_q), daemon=False)
             for i, gid in enumerate(gpu_ids)]
    for pr in procs:
        pr.start()

    n_done = 0
    worker_errors = []
    counts = {"complete": 0, "failed": 0, "skip": 0}

    def _handle(msg):
        nonlocal n_done
        k = msg.get("kind")
        if k == "worker_done":
            n_done += 1
        elif k in counts:
            counts[k] += 1
        elif k == "worker_error":
            worker_errors.append(msg)

    # drain while workers run; stop when all reported done OR all dead (crash)
    while n_done < len(procs) and any(pr.is_alive() for pr in procs):
        try:
            _handle(out_q.get(timeout=0.5))
        except _queue.Empty:
            continue
    # best-effort final drain
    try:
        while True:
            _handle(out_q.get_nowait())
    except _queue.Empty:
        pass
    for pr in procs:
        pr.join()
    crashed = [pr for pr in procs if pr.exitcode not in (0, None)]

    runner.aggregate_errors(args.pred_dir)

    final_complete = sum(1 for s, _ in pages if runner.page_status(args.pred_dir, s) == "complete")
    final_failed = sum(1 for s, _ in pages if runner.page_status(args.pred_dir, s) == "failed")
    final_pending = len(pages) - final_complete - final_failed
    status = runner.decide_run_status(final_failed, final_pending,
                                      worker_errors=len(worker_errors), crashed=len(crashed))
    runner.write_run_manifest(args.pred_dir, backend="transformers", model=args.model,
                              counts={"expected": len(pages), "succeeded": final_complete,
                                      "failed": final_failed, "skipped": counts["skip"]},
                              gpu_ids=gpu_ids, status=status)
    print(f"[summary] expected={len(pages)} complete={final_complete} failed={final_failed} "
          f"pending={final_pending} worker_errors={len(worker_errors)} crashed={len(crashed)}",
          flush=True)
    for e in worker_errors:
        print(f"[worker_error] GPU {e['gpu']}: {e['msg']}", file=sys.stderr)
    if status != "ok":
        sys.exit(f"[error] run failed: {final_failed} failed, {final_pending} pending, "
                 f"{len(worker_errors)} worker errors, {len(crashed)} crashed")
    print("[done] all pages complete", flush=True)


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Verify it compiles (CPU; the torch import is lazy inside `_worker`)**

Run: `python -m compileall -q scripts/run_phase1_transformers.py` → rc 0.

- [ ] **Step 3: Commit**

```bash
git add scripts/run_phase1_transformers.py
git commit -m "feat(phase1): worker exit codes, queue results, retries, manifest, non-zero on failure

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

### Task 1.11: Optional minimal ruff config (opt-in; do NOT reformat vendored files)

**Files:**
- Modify: `pyproject.toml`

- [ ] **Step 1: Add a minimal, explained `[tool.ruff]`**

Append to `pyproject.toml`:

```toml
[tool.ruff]
line-length = 120
target-version = "py312"
# Vendored/upstream-derived algorithm files: intentionally NOT reformatted so
# diffs against upstream stay trackable. Lint only; never auto-format these.
[tool.ruff.lint.per-file-ignores]
"src/hunyuan_ocr/postprocess.py" = ["E501", "E741"]
"src/hunyuan_ocr/tasks.py" = ["E501"]
"src/hunyuan_ocr/contract.py" = ["E501"]
```

- [ ] **Step 2: Run ruff (if installed); record result, do not mass-fix vendored files**

Run: `cd /workspace/HunyuanOCR-ROCm && (ruff check . 2>&1 | tail -20 || echo 'ruff not installed — skipped (optional in P0)')`

- [ ] **Step 3: Commit**

```bash
git add pyproject.toml
git commit -m "chore: minimal ruff config with per-file ignores for vendored upstream files

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

## Task 2: License & third-party attribution

> Doc/task steps. No unit tests; each step has a verification command.

### Task 2.1: Full Apache-2.0 `LICENSE` + `LICENSES/`

**Files:**
- Replace: `LICENSE`
- Create: `LICENSES/Apache-2.0.txt`
- Create: `LICENSES/Tencent-Hunyuan-Community-License.txt`

- [ ] **Step 1: Fetch the canonical Apache-2.0 text (verified: 202 lines)**

```bash
cd /workspace/HunyuanOCR-ROCm
curl -sL https://www.apache.org/licenses/LICENSE-2.0.txt -o /tmp/apache2.txt
test "$(wc -l < /tmp/apache2.txt)" -eq 202
grep -q "APPENDIX" /tmp/apache2.txt
cp /tmp/apache2.txt LICENSES/Apache-2.0.txt
```

- [ ] **Step 2: Write the root `LICENSE` = copyright header + full Apache-2.0**

Create `LICENSE` with exactly this content (header lines, then the full text from `/tmp/apache2.txt`):

```
Copyright 2026 AIwork4me

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

----
The full Apache License 2.0 text follows.
```

Then append the full text: `cat /tmp/apache2.txt >> LICENSE`.

- [ ] **Step 3: Fetch the upstream Tencent license verbatim (GitHub API; base64-decode)**

```bash
curl -sL "https://api.github.com/repos/Tencent-Hunyuan/HunyuanOCR/contents/LICENSE" \
  | python3 -c "import sys,json,base64; print(base64.b64decode(json.load(sys.stdin)['content']).decode('utf-8'))" \
  > LICENSES/Tencent-Hunyuan-Community-License.txt
# verify it is the Tencent Hunyuan Community License (not empty, has §3d notice string)
grep -q "TENCENT HUNYUAN COMMUNITY LICENSE AGREEMENT" LICENSES/Tencent-Hunyuan-Community-License.txt
grep -q "Powered by Tencent Hunyuan" LICENSES/Tencent-Hunyuan-Community-License.txt
```

- [ ] **Step 4: Verify**

```bash
test -f LICENSE && test -f LICENSES/Apache-2.0.txt && test -f LICENSES/Tencent-Hunyuan-Community-License.txt
grep -c "APPENDIX" LICENSE  # expect >=1
wc -l LICENSE LICENSES/Apache-2.0.txt  # both ~200+ lines
```

- [ ] **Step 5: Commit**

```bash
git add LICENSE LICENSES/
git commit -m "license: full Apache-2.0 LICENSE + vendored Apache-2.0 and Tencent Hunyuan Community licenses

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

### Task 2.2: Per-file attribution headers

**Files:** `src/hunyuan_ocr/contract.py`, `tasks.py`, `postprocess.py`, `omnidocbench.py`, `scoring.py`, `runner.py`, `validation.py`, `backends/transformers.py`, `backends/vllm_client.py`.

Header policy:
- **Original files** (`omnidocbench.py`, `scoring.py`, `runner.py`, `validation.py`) → top-of-file comment:

```python
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 AIwork4me
```

- **Upstream-derivative files** (`contract.py`, `tasks.py`, `postprocess.py` Group 1, `backends/transformers.py`, `backends/vllm_client.py`) → top-of-file comment block:

```python
# SPDX-License-Identifier: LicenseRef-Tencent-Hunyuan-Community-License
# Copyright (c) Tencent. All rights reserved.
# Copyright 2026 AIwork4me (modifications for the ROCm port)
#
# This file derives from Tencent HunyuanOCR (https://github.com/Tencent-Hunyuan/HunyuanOCR),
# licensed under the Tencent Hunyuan Community License Agreement. Upstream-derived
# portions retain that license; see LICENSES/LicenseRef-Tencent-Hunyuan-Community-License.txt.
# The "Powered by Tencent Hunyuan" mark is encouraged (license §3c), not required.
```

- **`postprocess.py` Group 2 (`process_one`)** → additionally flag the attribution uncertainty in the existing module docstring (append one line): `"# Attribution note: Group 2 (process_one) patterns are original normalization work; Group 1 mirrors upstream. See NOTICE."` Mark in NOTICE open-items.

- [ ] **Step 1: Add the appropriate header to each file (Edit tool, top of file).** Do NOT otherwise modify vendored bodies.

- [ ] **Step 2: Verify every src .py has a header**

```bash
for f in src/hunyuan_ocr/*.py src/hunyuan_ocr/backends/*.py; do
  head -1 "$f" | grep -q "SPDX-License-Identifier" || echo "MISSING HEADER: $f"
done
# expect no MISSING lines
```

- [ ] **Step 3: Commit**

```bash
git add src/hunyuan_ocr/
git commit -m "license(headers): honest per-file SPDX/attribution; upstream-derived code is NOASSERTION not Apache

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

### Task 2.3: Rewrite `NOTICE`

**Files:** `NOTICE` (rewrite).

- [ ] **Step 1: Replace `NOTICE` contents with**

```
HunyuanOCR-ROCm
Copyright 2026 AIwork4me

This NOTICE fulfills the distribution conditions of the licenses below.

------------------------------------------------------------------------------
1. Original packaging, tooling, and integration code in this repository
------------------------------------------------------------------------------
Copyright 2026 AIwork4me. Licensed under the Apache License, Version 2.0
(see LICENSE and LICENSES/Apache-2.0.txt).

------------------------------------------------------------------------------
2. Code ported from or derived from Tencent HunyuanOCR
------------------------------------------------------------------------------
Some files (contract.py, tasks.py, postprocess.py, backends/*) derive from
Tencent HunyuanOCR (https://github.com/Tencent-Hunyuan/HunyuanOCR). Those
upstream-derived portions are licensed under the Tencent Hunyuan Community
License Agreement
(see LICENSES/Tencent-Hunyuan-Community-License.txt). When distributing those
portions you must include a copy of that agreement and prominent notices on
any modified files stating that you changed them.

------------------------------------------------------------------------------
3. Tencent HunyuanOCR model weights
------------------------------------------------------------------------------
The model weights (tencent/HunyuanOCR; ggml-org/HunyuanOCR-GGUF) are licensed
under the Tencent Hunyuan Community License Agreement. They are NOT OSI Open
Source and are NOT licensed in the EU, UK, or South Korea (territory-limited).
Distributions (other than via a Hosted Service) must carry this Notice text:

    Tencent Hunyuan is licensed under the Tencent Hunyuan Community License
    Agreement, Copyright © 2025 Tencent. All Rights Reserved. The trademark
    rights of "Tencent Hunyuan" are owned by Tencent or its affiliate.

The "Powered by Tencent Hunyuan" mark is ENCOURAGED by the license (§3c),
not required.

------------------------------------------------------------------------------
4. Non-affiliation
------------------------------------------------------------------------------
This project is an independent community port. Tencent is not affiliated with,
associated with, sponsoring, or endorsing this project. "Tencent Hunyuan" is a
trademark of Tencent or its affiliate.

------------------------------------------------------------------------------
5. Third-party projects used
------------------------------------------------------------------------------
- llama.cpp (https://github.com/ggml-org/llama.cpp) -- MIT
- vLLM (https://github.com/vllm-project/vllm) -- Apache-2.0
- OmniDocBench (https://github.com/opendatalab/OmniDocBench) -- benchmark (see its license)

Powered by Tencent Hunyuan.
```

- [ ] **Step 2: Verify the mandatory §3d string is present verbatim**

```bash
grep -F "Tencent Hunyuan is licensed under the Tencent Hunyuan Community License" NOTICE
grep -qi "ENCOURAGED" NOTICE  # 'Powered by' framed as encouraged, not required
grep -qi "not OSI Open Source" NOTICE
```

- [ ] **Step 3: Commit**

```bash
git add NOTICE
git commit -m "license(notice): correct §3d notice, §3e non-affiliation, weights-not-OSI, 'Powered by' framed as encouraged

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

### Task 2.4: README license section

**Files:** `README.md` (edit the `## License` block, line ~118-121).

- [ ] **Step 1: Replace the License section** with the honest 4-category version, and fix the badge (line 8) from `code-Apache--2.0` → `license-mixed`.

New `## License` section:

```markdown
## License

This repository is **mixed-licensed** (see [NOTICE](NOTICE)):

1. **Original packaging/tooling** (drivers, `runner.py`, `validation.py`, `scoring.py`, `omnidocbench.py`): **Apache-2.0** ([LICENSE](LICENSE), [LICENSES/Apache-2.0.txt](LICENSES/Apache-2.0.txt)).
2. **Code ported from HunyuanOCR** (`contract.py`, `tasks.py`, `postprocess.py`, `backends/*`): upstream-derived portions under the **Tencent Hunyuan Community License** ([LICENSES/Tencent-Hunyuan-Community-License.txt](LICENSES/Tencent-Hunyuan-Community-License.txt)) — *not* Apache.
3. **HunyuanOCR model weights** (`tencent/HunyuanOCR`, `ggml-org/HunyuanOCR-GGUF`): **Tencent Hunyuan Community License** — **not OSI Open Source**; excludes EU/UK/KR; "Powered by Tencent Hunyuan" is *encouraged*, not required.
4. **llama.cpp**: MIT. **vLLM**: Apache-2.0.

Tencent is not affiliated with, sponsoring, or endorsing this project.
```

Badge line edit: `[![License](https://img.shields.io/badge/license-mixed%20(see%20NOTICE)-blue)](NOTICE)`.

- [ ] **Step 2: Verify no broken `License.txt` link remains pointing to a wrong target**

```bash
grep -n "License.txt" README.md || echo "no License.txt refs (ok)"
```

- [ ] **Step 3: Commit (together with the Task 3 README edits in the Task 3 commit, OR separately here)**

```bash
git add README.md
git commit -m "docs(readme): honest mixed-license section + fixed license badge

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

## Task 3: Downgrade over-attribution + unify doc status

### Task 3.1: README results-scope downgrades

**Files:** `README.md` (edit the `## Results` / Key findings / gfx1100 sections).

- [ ] **Step 1: Apply the before→after edits from spec §7.** Concretely, in `README.md`:
  - Replace the `**Key findings:**` block so that:
    - "llama.cpp is the fastest and most stable backend on gfx1100, and the only one…" → "**Among the three tested backends, on the tested gfx1100/ROCm 7.2 stack**, llama.cpp is fastest and most stable, and the only one of the three that runs with no pixel cap."
    - "formula CDM gap … confirmed by systematic ablation" → "After ruling out resolution, streaming, post-processing, and systematic formula omission, **inference-engine-level numerical divergence is the leading explanation** (not a singly-proven root cause)."
    - ">14k ViT instability … only affects the transformers SDPA path" → "Observed in the **transformers/ROCm full-ViT path using SDPA on the tested gfx1100 stack**; a standalone SDPA op does not reproduce it, so it is not pinned to a single SDPA kernel, and we have no NVIDIA control to bound it to ROCm."
  - Add an explicit **vLLM full-set** note near the Results table:

```markdown
> **vLLM full-set: NOT a valid result.** The 1651-page vLLM run never produced a
> valid score — servers crashed under sustained load, yielding ~780 ERROR pages;
> the resulting 46.31 is **not a valid benchmark** and is excluded from all
> comparisons. The vLLM **canary (148 pages, 94.81) is the only reliable vLLM
> number.** No "full run in progress."
```

- [ ] **Step 2: Verify the forbidden absolute phrases are gone/scoped**

```bash
grep -nE "confirmed (by|to)|root-caused|the fastest and most stable backend on gfx1100" README.md || echo "ok: no unqualified absolutes"
grep -n "46.31" README.md   # must appear in the invalid-result note, not as a score
grep -ni "full run in progress" README.md || echo "ok: no 'in progress'"
```

- [ ] **Step 3: Commit** (fold into the README commit from Task 2.4, or a dedicated one).

```bash
git add README.md
git commit -m "docs(readme): scope over-attributed conclusions; mark vLLM 46.31 invalid; drop 'in progress'

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

### Task 3.2: Historical headers + stale-state fixes on reports

**Files:** `reports/HANDOFF.md`, `reports/canary-baseline.md`, `reports/project-stage-summary.md`.

- [ ] **Step 1: Prepend a Historical banner to each report** (right after the H1 title line):

```markdown
> **Historical — 2026-07-16.** Retained as experimental evidence; `README.md` is
> the single source of current status. Some conclusions in this file read
> stronger than the evidence now supports (see README). Machine-local paths
> (`/root/...`, `/workspace/...`) are factual cross-session evidence, not user
> repro paths — use `scripts/reproduce_*.sh` + `reproducibility.lock.yaml`.
```

- [ ] **Step 2: Fix stale state**
  - `project-stage-summary.md`: remove/replace the "🔄 Full 1651-page vLLM run in progress" bullet and the "In progress" section; replace with "vLLM full-set attempted, never completed a valid run (server crashes). Canary 94.81 is the reliable vLLM number."
  - `project-stage-summary.md` / `HANDOFF.md`: correct stale "Branch: feat/phase1-transformers" / "28 commits" framing to "snapshot 2026-07-16; see git history for current state."

- [ ] **Step 3: Verify**

```bash
grep -L "Historical" reports/HANDOFF.md reports/canary-baseline.md reports/project-stage-summary.md || echo "all reports carry the Historical banner"
grep -ni "run in progress" reports/*.md || echo "ok: no 'in progress' in reports"
```

- [ ] **Step 4: Commit**

```bash
git add reports/
git commit -m "docs(reports): Historical banners + stale-state fixes; remove 'vLLM run in progress'

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

### Task 3.3: Remove tracked `.ipynb_checkpoints`; scan src for overclaims

- [ ] **Step 1: Untrack the checkpoints file** (already gitignored; just remove from index)

```bash
cd /workspace/HunyuanOCR-ROCm
git rm --cached docs/.ipynb_checkpoints/rocm-issue-draft-checkpoint.md
```

- [ ] **Step 2: Scan src/scripts comments for overclaim words and downgrade where they assert root-cause**

```bash
grep -rniE "confirmed|root-caused|root cause|only (affects|backend)|precision[- ]aligned" src/ scripts/ || echo "no overclaim comments"
```
Fix any hit that asserts a root cause (e.g., in `backends/transformers.py` the ViT-cap comment says "non-deterministic" — keep the observation, but if it says "the SDPA kernel" as a root cause, soften to "observed in the full ViT SDPA path on the tested gfx1100 stack").

- [ ] **Step 3: Commit**

```bash
git add -A
git commit -m "chore: untrack .ipynb_checkpoints; soften overclaim comments in src

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

## Task 4: Reproducibility freeze

### Task 4.1: `reproducibility.lock.yaml`

**Files:** `reproducibility.lock.yaml` (new).

- [ ] **Step 1: Write the lock file** with the verified facts (spec §3) and `not_recorded` + fill-commands for the two unreachable HF fields:

```yaml
# reproducibility.lock.yaml — machine-readable reproducibility snapshot for the
# published HunyuanOCR-ROCm results. Verified fields carry a source; unverified
# fields are `not_recorded` with a fill command. Do not invent values.

hunyuanocr_rocm:
  repo: https://github.com/AIwork4me/HunyuanOCR-ROCm
  commit: e17fab1d3c2586599b9ee0c845784e4b000e2101   # git rev-parse HEAD (verified)

llama_cpp:
  repo: https://github.com/ggml-org/llama.cpp
  commit: a320cbfcb7056b7b81fb854d97fe01d0ea77c4b5   # GitHub API (verified 2026-07-16)
  ggml_version: "0.16.0"                              # per HANDOFF (reported)
  contains_hunyuan_multimodal: true                   # tools/mtmd/models/hunyuanvl.cpp + src/models/hunyuan-vl.cpp present at this commit

model:
  hf_repo: tencent/HunyuanOCR
  hf_revision: not_recorded                           # huggingface.co API not reachable from this env
  # fill: curl -sL https://huggingface.co/api/models/tencent/HunyuanOCR/revision/main | jq -r .sha
  gguf_repo: ggml-org/HunyuanOCR-GGUF
  gguf_files: [HunyuanOCR-bf16.gguf, mmproj-HunyuanOCR-bf16.gguf]
  gguf_lfs_oid: not_recorded                          # HF API not reachable from this env
  # fill: curl -sL https://huggingface.co/api/models/ggml-org/HunyuanOCR-GGUF/tree/main | jq -r '.[]|select(.path|contains("bf16"))|{path,.lfs.oid}'

omnidocbench:
  version: v1.6
  repo: https://github.com/opendatalab/OmniDocBench
  repo_default_branch: main
  # repo commit (fill): curl -sL https://api.github.com/repos/opendatalab/OmniDocBench/commits/main | jq -r .sha
  gt_json_canary: OmniDocBench_150.json
  gt_json_canary_sha256: 3e3fbea07702084d9466e231260ad92141848a32631c9895d8e55b24e2c2f7b5   # sha256sum (verified, 148 pages)
  gt_json_full: OmniDocBench.json
  gt_json_full_sha256: a45cd84b04ad8b793e775089640e6b681209abea33ead54c1828ddca35fae496     # sha256sum (verified)
  scorer_repo: /root/ocr-eval/OmniDocBench
  scorer_commit: 2b161d010d2e3aff77a0edef359ea3a6411d23cd                              # git rev-parse HEAD (verified, local)

environment:
  python: "3.12.3"                                    # sys.version (verified)
  rocm_hip: "7.2.53211-e1a6bc5663"                    # torch.version.hip (verified)
  torch: "2.9.1+gitff65f5b"                           # torch.__version__ (verified)
  transformers:
    benchmark_venv: "5.13.0"                          # REPORTED in HANDOFF (/root/hunyuanocr-venvs/transformers); NOT re-verified this session
    current_opt_venv: "4.57.6"                        # transformers.__version__ here (verified)
    note: "The published benchmark was produced in the isolated 5.13.0 venv, NOT this /opt/venv."
  vllm: "0.16.1.dev0+g89a77b108.d20260317"            # vllm.__version__ (verified)
  gpu_arch: gfx1100                                   # RDNA3 (per repo docs)
  rocm_smi_device_id: "0x744b"                        # rocm-smi (verified)

benchmark:
  date: "2026-07-16"                                  # per reports
  hardware: "4x AMD gfx1100 (RDNA3, 48GB), ROCm 7.2"
  canary_148:
    vllm_overall: 94.81
    transformers_overall: 94.11
    llamacpp_overall: 93.33
  full_1651:
    llamacpp_overall: 92.09
    vllm_overall: invalid                            # ~780 ERROR pages; 46.31 is not a valid benchmark
```

- [ ] **Step 2: Verify it parses + verified SHA matches live file**

```bash
python -c "import yaml,sys; yaml.safe_load(open('reproducibility.lock.yaml')); print('yaml ok')"
sha256sum /workspace/OmniDocBench_data/OmniDocBench_150.json | grep -q 3e3fbea07702084d9466e231260ad92141848a32631c9895d8e55b24e2c2f7b5 && echo "canary sha matches"
```

- [ ] **Step 3: Commit**

```bash
git add reproducibility.lock.yaml
git commit -m "repro: reproducibility.lock.yaml with verified commits/SHA256/env; HF fields marked not_recorded

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

### Task 4.2: `scripts/create_canary_manifest.py` + generated manifest

**Files:**
- Create: `scripts/create_canary_manifest.py`
- Create: `eval/canary_148.manifest.json` (generated from the real GT)
- Test: `tests/test_canary_manifest.py`

**Interfaces:** `build_manifest(gt_json, *, name, dataset, dataset_version) -> dict`; `manifest_sha256(d) -> str` (sha over canonical JSON **without** the `manifest_sha256` field).

- [ ] **Step 1: Write the failing test** (`tests/test_canary_manifest.py`)

```python
# tests/test_canary_manifest.py
import json
import sys
from pathlib import Path


def _import():
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
    import create_canary_manifest as m
    return m


def test_build_manifest_structure_and_sha(tmp_path):
    m = _import()
    gt = tmp_path / "gt.json"
    pages = [{"page_info": {"image_path": f"page-{i}.png"}} for i in [3, 1, 2]]
    gt.write_text(json.dumps(pages), encoding="utf-8")
    d = m.build_manifest(str(gt), name="canary-test", dataset="OmniDocBench", dataset_version="v1.6")
    assert d["expected_count"] == 3
    assert d["subset_name"] == "canary-test"
    assert [p["stem"] for p in d["pages"]] == ["page-1", "page-2", "page-3"]  # sorted
    assert "source_json_sha256" in d and len(d["source_json_sha256"]) == 64
    # manifest_sha recomputes from the dict WITHOUT manifest_sha256
    sha = m.manifest_sha256(d)
    d2 = dict(d); d2["manifest_sha256"] = sha
    assert m.manifest_sha256(d) == sha
```

- [ ] **Step 2: Run to verify it fails** → FAIL (module missing).

- [ ] **Step 3: Create `scripts/create_canary_manifest.py`**

```python
#!/usr/bin/env python3
"""Generate a verifiable canary manifest from an OmniDocBench GT json.

Emits JSON: subset name, source dataset/version, expected_count, sorted page
stems + image_paths, source JSON SHA256, manifest SHA256. Deterministic.

The manifest_sha256 is the sha256 of the canonical JSON of the manifest WITH
that field omitted (so a reader can drop it and recompute to verify integrity).

Usage:
  python scripts/create_canary_manifest.py \
      --gt-json /workspace/OmniDocBench_data/OmniDocBench_150.json \
      --name canary-148 --dataset OmniDocBench --dataset-version v1.6 \
      --out eval/canary_148.manifest.json
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path


def sha256_file(p) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def build_manifest(gt_json, *, name, dataset, dataset_version) -> dict:
    pages = json.load(open(gt_json, encoding="utf-8"))
    entries = sorted((Path(p["page_info"]["image_path"]).stem,
                      p["page_info"]["image_path"]) for p in pages)
    return {
        "subset_name": name,
        "source_dataset": dataset,
        "source_dataset_version": dataset_version,
        "expected_count": len(entries),
        "pages": [{"stem": s, "image_path": ip} for s, ip in entries],
        "source_json_sha256": sha256_file(gt_json),
    }


def manifest_sha256(d: dict) -> str:
    body = {k: v for k, v in d.items() if k != "manifest_sha256"}
    text = json.dumps(body, ensure_ascii=False, sort_keys=True, indent=2)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--gt-json", required=True)
    p.add_argument("--name", required=True)
    p.add_argument("--dataset", default="OmniDocBench")
    p.add_argument("--dataset-version", default="v1.6")
    p.add_argument("--out", required=True)
    args = p.parse_args()

    m = build_manifest(args.gt_json, name=args.name, dataset=args.dataset,
                       dataset_version=args.dataset_version)
    m["manifest_sha256"] = manifest_sha256(m)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(m, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
                   encoding="utf-8")
    print(f"wrote {out}: {m['expected_count']} pages, "
          f"source_sha256={m['source_json_sha256'][:12]}..., "
          f"manifest_sha256={m['manifest_sha256'][:12]}...")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run tests to verify pass**

Run: `python -m pytest tests/test_canary_manifest.py -v` → PASS.

- [ ] **Step 5: Generate the real manifest from the local GT and commit it**

```bash
cd /workspace/HunyuanOCR-ROCm
python scripts/create_canary_manifest.py \
  --gt-json /workspace/OmniDocBench_data/OmniDocBench_150.json \
  --name canary-148 --dataset OmniDocBench --dataset-version v1.6 \
  --out eval/canary_148.manifest.json
python -c "import json; d=json.load(open('eval/canary_148.manifest.json')); assert d['expected_count']==148; print('ok', d['expected_count'])"
```

- [ ] **Step 6: Commit**

```bash
git add scripts/create_canary_manifest.py tests/test_canary_manifest.py eval/canary_148.manifest.json
git commit -m "repro: create_canary_manifest.py + generated 148-page canary manifest (verified SHA256)

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

### Task 4.3: Reproduce scripts

**Files:**
- Create: `scripts/reproduce_llamacpp_canary.sh`
- Create: `scripts/reproduce_llamacpp_full.sh`

- [ ] **Step 1: Create `scripts/reproduce_llamacpp_canary.sh`**

```bash
#!/usr/bin/env bash
# Reproduce the published llama.cpp canary (148-page) result on OmniDocBench v1.6.
# Pipeline: predict -> validate -> score. No auto-download; overwrite-guarded.
# Start the llama-servers yourself first (one/GPU), e.g.:
#   for g in 0 1 2 3; do CUDA_VISIBLE_DEVICES=$g $LLAMA_DIR/build/bin/llama-server \
#     --model $GGUF_DIR/HunyuanOCR-bf16.gguf --mmproj $GGUF_DIR/mmproj-HunyuanOCR-bf16.gguf \
#     --host 127.0.0.1 --port $((8081+g)) --alias HYVL -ngl 999 -c 65536 -n 32768 & done
set -euo pipefail

LLAMA_DIR="${LLAMA_DIR:?LLAMA_DIR must point to a llama.cpp checkout at the locked commit}"
GGUF_DIR="${GGUF_DIR:?GGUF_DIR must contain HunyuanOCR-bf16.gguf + mmproj-HunyuanOCR-bf16.gguf}"
DATA_DIR="${DATA_DIR:?DATA_DIR must be the OmniDocBench data dir}"
GT_JSON="${GT_JSON:-$DATA_DIR/OmniDocBench_150.json}"
OUT_DIR="${OUT_DIR:?OUT_DIR must be set (output predictions dir)}"
PORTS="${PORTS:-8081,8082,8083,8084}"
HOST="${HOST:-127.0.0.1}"
CONCURRENCY="${CONCURRENCY:-16}"
LOCKED_LLAMA_COMMIT="a320cbfcb7056b7b81fb854d97fe01d0ea77c4b5"

echo "[repro] canary 148 pages, llama.cpp @ $LOCKED_LLAMA_COMMIT"

for f in "$GT_JSON" "$DATA_DIR/images" "$GGUF_DIR/HunyuanOCR-bf16.gguf" \
         "$GGUF_DIR/mmproj-HunyuanOCR-bf16.gguf" "$LLAMA_DIR/build/bin/llama-server"; do
  [[ -e "$f" ]] || { echo "[fatal] missing: $f" >&2; exit 1; }
done

ACTUAL="$(git -C "$LLAMA_DIR" rev-parse HEAD)"
if [[ "$ACTUAL" != "$LOCKED_LLAMA_COMMIT" ]]; then
  echo "[fatal] llama.cpp HEAD=$ACTUAL != locked $LOCKED_LLAMA_COMMIT" >&2
  echo "        cd $LLAMA_DIR && git checkout $LOCKED_LLAMA_COMMIT && cmake --build build" >&2
  exit 1
fi

if compgen -G "$OUT_DIR/*.md" > /dev/null && [[ -z "${OVERWRITE:-}" ]]; then
  echo "[fatal] $OUT_DIR already has predictions; set OVERWRITE=1 to resume/redo" >&2; exit 1
fi
mkdir -p "$OUT_DIR"

REPO="$(cd "$(dirname "$0")/.." && pwd)"

echo "[repro] step 1/3: predict"
python "$REPO/scripts/run_phase2_vllm.py" \
  --gt-json "$GT_JSON" --images-dir "$DATA_DIR/images" \
  --pred-dir "$OUT_DIR" --host "$HOST" --ports "$PORTS" \
  --model HYVL --concurrency "$CONCURRENCY"

echo "[repro] step 2/3: validate"
python "$REPO/scripts/validate_predictions.py" --gt-json "$GT_JSON" --pred-dir "$OUT_DIR"

echo "[repro] step 3/3: score"
python "$REPO/scripts/score_predictions.py" --pred-dir "$OUT_DIR" --gt-json "$GT_JSON" \
  --label llamacpp-canary-148

echo "[repro] done."
```

- [ ] **Step 2: Create `scripts/reproduce_llamacpp_full.sh`** (same template; different defaults)

```bash
#!/usr/bin/env bash
# Reproduce the published llama.cpp full-set (1651-page) result on OmniDocBench v1.6.
# Pipeline: predict -> validate -> score. See reproduce_llamacpp_canary.sh for server start.
set -euo pipefail

LLAMA_DIR="${LLAMA_DIR:?LLAMA_DIR must point to a llama.cpp checkout at the locked commit}"
GGUF_DIR="${GGUF_DIR:?GGUF_DIR must contain HunyuanOCR-bf16.gguf + mmproj-HunyuanOCR-bf16.gguf}"
DATA_DIR="${DATA_DIR:?DATA_DIR must be the OmniDocBench data dir}"
GT_JSON="${GT_JSON:-$DATA_DIR/OmniDocBench.json}"
OUT_DIR="${OUT_DIR:?OUT_DIR must be set (output predictions dir)}"
PORTS="${PORTS:-8081,8082,8083,8084}"
HOST="${HOST:-127.0.0.1}"
CONCURRENCY="${CONCURRENCY:-16}"
LOCKED_LLAMA_COMMIT="a320cbfcb7056b7b81fb854d97fe01d0ea77c4b5"

echo "[repro] full 1651 pages, llama.cpp @ $LOCKED_LLAMA_COMMIT"

for f in "$GT_JSON" "$DATA_DIR/images" "$GGUF_DIR/HunyuanOCR-bf16.gguf" \
         "$GGUF_DIR/mmproj-HunyuanOCR-bf16.gguf" "$LLAMA_DIR/build/bin/llama-server"; do
  [[ -e "$f" ]] || { echo "[fatal] missing: $f" >&2; exit 1; }
done

ACTUAL="$(git -C "$LLAMA_DIR" rev-parse HEAD)"
[[ "$ACTUAL" == "$LOCKED_LLAMA_COMMIT" ]] || { echo "[fatal] llama.cpp HEAD=$ACTUAL != $LOCKED_LLAMA_COMMIT" >&2; exit 1; }

if compgen -G "$OUT_DIR/*.md" > /dev/null && [[ -z "${OVERWRITE:-}" ]]; then
  echo "[fatal] $OUT_DIR already has predictions; set OVERWRITE=1 to resume/redo" >&2; exit 1
fi
mkdir -p "$OUT_DIR"

REPO="$(cd "$(dirname "$0")/.." && pwd)"

echo "[repro] step 1/3: predict"
python "$REPO/scripts/run_phase2_vllm.py" \
  --gt-json "$GT_JSON" --images-dir "$DATA_DIR/images" \
  --pred-dir "$OUT_DIR" --host "$HOST" --ports "$PORTS" \
  --model HYVL --concurrency "$CONCURRENCY"

echo "[repro] step 2/3: validate"
python "$REPO/scripts/validate_predictions.py" --gt-json "$GT_JSON" --pred-dir "$OUT_DIR"

echo "[repro] step 3/3: score"
python "$REPO/scripts/score_predictions.py" --pred-dir "$OUT_DIR" --gt-json "$GT_JSON" \
  --label llamacpp-full-1651

echo "[repro] done."
```

- [ ] **Step 3: Make executable + syntax check**

```bash
chmod +x scripts/reproduce_llamacpp_canary.sh scripts/reproduce_llamacpp_full.sh
bash -n scripts/reproduce_llamacpp_canary.sh && bash -n scripts/reproduce_llamacpp_full.sh && echo "syntax ok"
grep -L "set -euo pipefail" scripts/reproduce_llamacpp_*.sh || echo "both have set -euo pipefail"
```

- [ ] **Step 4: Commit**

```bash
git add scripts/reproduce_llamacpp_canary.sh scripts/reproduce_llamacpp_full.sh
git commit -m "repro: parameterized canary + full reproduce scripts (locked commit, 127.0.0.1, validate before score)

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

### Task 4.4: README "Reproducibility" section

**Files:** `README.md` (add a new section; also pin the llama.cpp checkout in the Quick start).

- [ ] **Step 1: In Quick start, pin the llama.cpp checkout**

Change the clone block to:

```bash
git clone https://github.com/ggml-org/llama.cpp.git
cd llama.cpp
git checkout a320cbfcb7056b7b81fb854d97fe01d0ea77c4b5   # locked commit for published results
```

- [ ] **Step 2: Add a `## Reproducibility` section** (near Architecture / before License):

```markdown
## Reproducibility

- **Lock file:** [`reproducibility.lock.yaml`](reproducibility.lock.yaml) pins every
  verified input (repo + llama.cpp commits, GT SHA256, env versions) for the published
  results. Unreachable-from-our-env fields (HF model revision, GGUF LFS oid) are
  `not_recorded` with a fill command.
- **Canary manifest:** [`eval/canary_148.manifest.json`](eval/canary_148.manifest.json)
  lists the 148 canary pages with the source-GT SHA256. Regenerate with
  `scripts/create_canary_manifest.py`.
- **Reproduce scripts:** `scripts/reproduce_llamacpp_canary.sh` (148) and
  `scripts/reproduce_llamacpp_full.sh` (1651) run predict → validate → score against a
  locked llama.cpp commit, binding `127.0.0.1`. Set `LLAMA_DIR/GGUF_DIR/DATA_DIR/OUT_DIR`.
- **Which numbers are formal vs diagnostic:**
  - **Formal/reliable:** llama.cpp full 1651 = **92.09**; canary vLLM 94.81, transformers
    94.11, llama.cpp 93.33.
  - **Diagnostic only:** the >14k ViT isolation, throughput tuning, formula-CDM ablation.
  - **Invalid (excluded):** vLLM full-set 46.31 (server crashes, ~780 ERROR pages).
- **Why newer deps may differ:** the benchmark used transformers 5.13.0 in an isolated
  venv; a current install may have a different transformers/torch/vLLM and can produce
  different numbers, especially around the >14k ViT path and formula CDM.
```

- [ ] **Step 3: Commit**

```bash
git add README.md
git commit -m "docs(readme): Reproducibility section + pinned llama.cpp checkout

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

## Acceptance (run after Task 4; CPU-only)

- [ ] **A. Static checks**

```bash
cd /workspace/HunyuanOCR-ROCm
git diff --check                                                    # no whitespace errors
python -m compileall -q src scripts                                 # rc 0
python -m pytest -q                                                 # all pass (pre-existing + new)
(ruff check . 2>&1 | tail -20 || echo 'ruff optional — skipped')    # no NEW errors in non-vendored files
```

- [ ] **B. 10-step CPU functional harness** (covered by `tests/test_phase2_no_gpu.py` + `tests/test_validation.py` + `tests/test_score_gate.py`; restate for clarity):

1. `test_phase2_no_gpu.py::test_phase2_two_ok_one_failed_then_rerun` builds 3 pages, fakes 1 failure.
2. `infer_one` raises on `c` → asserts exactly 2 valid `.md` (a, b).
3. asserts `_errors/c.json` written.
4. asserts run 1 `SystemExit` with non-zero code.
5. run 2: asserts a,b skipped (complete), c retried and now succeeds → no SystemExit.
6. (fabricate empty/ERROR/.partial) — `test_validation.py::test_empty_error_partial_markers`.
7. `validate_predictions` catches empty/error_marker/partial/missing — same test.
8. `test_score_gate.py` asserts scorer not invoked on invalid dir.
9. `test_validation.py::test_clean_dir_passes` asserts PASS on a clean dir.
10. overall `pytest -q` green.

- [ ] **C. License acceptance**

```bash
test "$(grep -c APPENDIX LICENSE)" -ge 1
test -f LICENSES/Apache-2.0.txt && test -f LICENSES/Tencent-Hunyuan-Community-License.txt
grep -F "Tencent Hunyuan is licensed under the Tencent Hunyuan Community License" NOTICE
! grep -qi "Powered by Tencent Hunyuan" NOTICE && echo "check: 'Powered by' present as encouraged line" ; grep -qi ENCOURAGED NOTICE
grep -niE "License\.txt" README.md || echo "no stray License.txt link"
```

- [ ] **D. Doc acceptance**

```bash
! grep -rni "full run in progress" README.md reports/
grep -ni "46.31" README.md        # present only in the invalid-result note
grep -rniE "leading explanation" README.md
grep -rL "Historical" reports/*.md || echo "all reports bannered"
```

- [ ] **E. Repro acceptance**

```bash
python -c "import yaml; yaml.safe_load(open('reproducibility.lock.yaml'))"
python -c "import json; d=json.load(open('eval/canary_148.manifest.json')); assert d['expected_count']==148"
bash -n scripts/reproduce_llamacpp_canary.sh && bash -n scripts/reproduce_llamacpp_full.sh
```

- [ ] **F. Final commit (if any uncommitted verification artifacts)** — none expected; everything committed per task.

---

## Open items (not auto-resolvable in P0)

- **`postprocess.py` Group 2 (`process_one`)**: README says "verbatim port", file docstring says "original normalization". Flagged in NOTICE as needing maintainer ruling (treated conservatively as upstream-derivative).
- **HF model revision + GGUF LFS oid**: `not_recorded` (HF API unreachable here); fill commands in the lock file.
- **transformers benchmark venv (5.13.0)**: reported in HANDOFF, not re-verified this session; recorded as reported.
- **Phase-1 multiprocessing worker end-to-end**: not CPU-testable (imports torch + needs GPU). The shared pure helpers (`select_todo`, `decide_run_status`, `page_status`) ARE tested; the worker glue is GPU-only and re-verifiable only on hardware.
- **No GPU re-derivation of any score** in this round (by design).
```
```
