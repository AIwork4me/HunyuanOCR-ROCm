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
