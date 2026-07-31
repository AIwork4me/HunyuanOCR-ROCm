#!/usr/bin/env python
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 AIwork4me
"""Deterministic README results-block generator (ADR-0007/ADR-0012 single source
of truth). Reads ``model_card_v2.json`` and emits the markdown between the
ROCmDoc markers. Re-running MUST produce no diff (``--check`` exits 1 on drift).

    python scripts/generate_results_block.py            # rewrite README block in place
    python scripts/generate_results_block.py --check     # CI: exit 1 if README drifted

The README never hand-writes a benchmark score; this block is derived from the v2
card (which is derived from raw evidence). Markers (frozen in .rocmdoc/spec-lock.json):

    <!-- BEGIN ROCMDOC GENERATED RESULTS -->
    <!-- END ROCMDOC GENERATED RESULTS -->
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

BEGIN = "<!-- BEGIN ROCMDOC GENERATED RESULTS -->"
END = "<!-- END ROCMDOC GENERATED RESULTS -->"


def render(card: dict) -> str:
    results = sorted(card.get("results", []), key=lambda r: r.get("result_id", ""))
    primary = card.get("primary_result_id")
    rows = []
    for r in results:
        rid = r["result_id"]
        impl = r.get("implementation", {})
        cov = r.get("coverage", {})
        metrics = r.get("metrics", {})
        overall = metrics.get("overall")
        star = " *(primary)*" if rid == primary else ""
        rows.append(
            f"| {impl.get('backend', '?')} | {cov.get('platform', '?')} | "
            f"{cov.get('page_count', '?')} | {'' if overall is None else round(overall, 2)} | "
            f"{r.get('assurance', '?')} | {r.get('status', '?')} | "
            f"`{rid}`{star} |"
        )
    header = (
        "| Backend | Platform | Pages | Overall | Assurance | Status | Result ID |\n|---|---|---|---|---|---|---|\n"
    )
    body = "\n".join(rows) if rows else "| _(no results yet)_ |  |  |  |  |  |  |"
    note = (
        "\n\n> Generated from `model_card_v2.json` (single source of truth). "
        "Overall is the raw-evidence-derived value; external/experimental "
        "references live in `docs/benchmark-context.md`."
    )
    return header + body + note


def replace_block(text: str, block: str) -> str:
    i = text.find(BEGIN)
    j = text.find(END)
    if i == -1 or j == -1 or j < i:
        # append a new block at the end if markers absent
        return text.rstrip() + "\n\n" + BEGIN + "\n" + block + "\n" + END + "\n"
    return text[: i + len(BEGIN)] + "\n" + block + "\n" + text[j:]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--card", default="model_card_v2.json")
    ap.add_argument("--readme", default="README.md")
    ap.add_argument("--check", action="store_true", help="exit 1 if README block drifted")
    a = ap.parse_args(argv)
    card = json.loads(Path(a.card).read_text(encoding="utf-8"))
    block = render(card)
    p = Path(a.readme)
    text = p.read_text(encoding="utf-8") if p.exists() else ""
    new = replace_block(text, block)
    if a.check:
        if new != text:
            print(f"DRIFT: {a.readme} results block is stale; run scripts/generate_results_block.py", file=sys.stderr)
            return 1
        print(f"OK: {a.readme} results block is up to date")
        return 0
    p.write_text(new, encoding="utf-8")
    print(f"wrote generated results block -> {a.readme}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
