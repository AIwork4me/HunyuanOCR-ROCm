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
