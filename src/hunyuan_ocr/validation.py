# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 AIwork4me
"""Pre-score validation of a prediction directory against OmniDocBench GT.

Pure function: read GT json + pred dir -> structured Report. No GPU, no model.
A non-clean report blocks scoring (see scripts/validate_predictions.py and the
gate in scripts/score_predictions.py).
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

from .runner import _OWN_ARTIFACTS, ERROR_PREFIX


@dataclass
class Problem:
    severity: str  # "error" | "warning"
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
            problems.append(
                Problem("error", "duplicate_stem", f"GT maps {n} pages to stem '{stem}'", {"stem": stem, "count": n})
            )
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
        problems.append(Problem("error", "error_marker", f"'{stem}.md' starts with 'ERROR:'", {"stem": stem}))

    for p in sorted(pred_dir.glob("*.partial")):
        problems.append(Problem("error", "partial", f"leftover partial '{p.name}'", {"file": p.name}))

    edir = pred_dir / "_errors"
    if edir.is_dir():
        for ef in sorted(edir.glob("*.json")):
            problems.append(
                Problem("error", "unresolved_error", f"unresolved error record '_errors/{ef.name}'", {"stem": ef.stem})
            )

    if pred_dir.is_dir():
        for entry in sorted(pred_dir.iterdir()):
            if entry.is_dir():
                if entry.name not in _OWN_ARTIFACTS:
                    problems.append(
                        Problem("warning", "unexpected_dir", f"unexpected dir '{entry.name}/'", {"name": entry.name})
                    )
                continue
            if entry.name in _OWN_ARTIFACTS:
                continue
            if entry.name.endswith(".md") or entry.name.endswith(".partial"):
                continue
            problems.append(
                Problem("warning", "unexpected_file", f"unexpected file '{entry.name}'", {"name": entry.name})
            )

    return Report(expected=expected, valid=valid, problems=problems)
