#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 AIwork4me
"""Render the README "GENERATED RESULTS" block from reproducibility.lock.yaml.

The lock is the single source of truth for the formal results. This script emits
the lock-sourced block between the BEGIN/END GENERATED RESULTS markers in
README.md (writing by default; `--check` fails CI if README has drifted).

Only values present in the lock are emitted — no number is invented or hand-typed.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parents[1]
LOCK_PATH = REPO / "reproducibility.lock.yaml"
README_PATH = REPO / "README.md"
REGION_RE = re.compile(r"<!-- BEGIN GENERATED RESULTS -->.*?<!-- END GENERATED RESULTS -->", re.DOTALL)

BACKEND_DISPLAY = {"llamacpp": "llama.cpp", "transformers": "transformers", "vllm": "vLLM"}


def _fmt_value(v) -> str:
    if isinstance(v, str) and v.strip().lower() == "invalid":
        return "invalid (excluded; see reproducibility.lock.yaml)"
    return str(v)


def render_block(lock: dict) -> str:
    """Render the BEGIN..END block (without surrounding README text) from the lock."""
    bench = (lock or {}).get("benchmark", {}) or {}
    lines = [
        "<!-- BEGIN GENERATED RESULTS -->",
        "<!-- Auto-generated from reproducibility.lock.yaml by scripts/render_benchmark_tables.py (do not edit by hand). -->",
        "",
        "| Page set | Backend | Overall | Source |",
        "|---|---|---|---|",
    ]
    for page_key, label in (("canary_148", "canary 148"), ("full_1651", "full 1651")):
        section = bench.get(page_key, {}) or {}
        rows = []
        for k, v in section.items():
            if not k.endswith("_overall"):
                continue
            backend = k[: -len("_overall")]
            rows.append((BACKEND_DISPLAY.get(backend, backend), _fmt_value(v)))
        for display, value in sorted(rows):
            lines.append(f"| {label} | {display} | {value} | reproducibility.lock.yaml |")
    official = bench.get("official_reference", {}) or {}
    if official:
        engine = official.get("inference_engine", "official")
        overall = _fmt_value(official.get("omnidocbench_overall"))
        lines.append(f"| official | {engine} | {overall} | official HunyuanOCR table |")
    lines.append("")
    lines.append("<!-- END GENERATED RESULTS -->")
    return "\n".join(lines)


def current_region(readme: str) -> str | None:
    m = REGION_RE.search(readme)
    return m.group(0) if m else None


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--check", action="store_true", help="fail (exit 1) if README region != generated")
    p.add_argument("--lock", default=str(LOCK_PATH))
    p.add_argument("--readme", default=str(README_PATH))
    args = p.parse_args(argv)

    lock = yaml.safe_load(Path(args.lock).read_text(encoding="utf-8"))
    block = render_block(lock)
    readme = Path(args.readme).read_text(encoding="utf-8")
    region = current_region(readme)
    if args.check:
        if region is None:
            print("FAIL: README has no BEGIN/END GENERATED RESULTS region", file=sys.stderr)
            return 1
        if region != block:
            print("FAIL: README GENERATED RESULTS region differs from the lock. Run:", file=sys.stderr)
            print("  python scripts/render_benchmark_tables.py", file=sys.stderr)
            return 1
        print("OK: README GENERATED RESULTS matches reproducibility.lock.yaml")
        return 0
    # write: replace the region (or insert it after the Results intro if absent)
    if region is not None:
        new_readme = readme.replace(region, block)
    else:
        # insert before the first "### Table A" if no region yet
        idx = readme.find("### Table A")
        if idx == -1:
            print("FAIL: cannot place the block (no region and no '### Table A' anchor)", file=sys.stderr)
            return 1
        new_readme = readme[:idx] + block + "\n\n" + readme[idx:]
    Path(args.readme).write_text(new_readme, encoding="utf-8")
    print(f"wrote GENERATED RESULTS block into {args.readme}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
