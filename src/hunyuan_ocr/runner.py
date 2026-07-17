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


# NOTE: --venv-python is redacted because it is a machine-local absolute path,
# not a credential — keeping it out of the portable manifest.
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
