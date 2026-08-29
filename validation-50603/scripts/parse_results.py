#!/usr/bin/env python3
"""Aggregate validation-50603 evidence into a markdown results table.

Reads <state>/*.evidence.json + *.log + routing-markers.txt for each state dir
and prints a markdown table + per-run output snippets for SUMMARY.md.
"""

import json
import pathlib
import sys

HERE = pathlib.Path(__file__).resolve().parent.parent
STATES = ["baseline", "pr53856", "pr53856-pr54210"]
SCRIPTS = ["repro_determinism", "repro_warmup", "repro_control_short"]
GT_TITLE = "quarterly financial summary"
# Ground-truth lines rendered onto the page by the gist scripts; the short
# control page only renders the title plus five revenue lines.
GT_LINES_FULL = [
    "Quarterly Financial Summary",
    "Revenue: 12,480,000 USD",
    "Cost of Goods Sold: 4,210,000 USD",
    "Gross Profit: 8,270,000 USD",
    "Operating Expense: 3,150,000 USD",
    "Net Income: 5,120,000 USD",
    "Earnings Per Share: 2.84 USD",
    "Fiscal Year: 2026 Q2 Report",
    "Prepared by Finance Committee",
]
GT_LINES_SHORT = GT_LINES_FULL[:6]


def gt_score(text: str, gt_lines: list[str]) -> tuple[int, int]:
    t = text.lower()
    hits = sum(1 for g in gt_lines if g.lower() in t)
    return hits, len(gt_lines)


def main() -> None:
    rows: list[str] = []
    snippets: list[str] = []
    ok = True
    for state in STATES:
        sdir = HERE / state
        if not sdir.exists():
            continue
        head = (sdir / "vllm-head.txt").read_text().strip() if (sdir / "vllm-head.txt").exists() else "?"
        for script in SCRIPTS:
            ev_path = sdir / f"{script}.evidence.json"
            if not ev_path.exists():
                rows.append(f"| {state} | {script} | — | — | — | — | — | missing evidence |")
                ok = False
                continue
            ev = json.loads(ev_path.read_text())
            sha8s = ev["sha8s"]
            texts = ev.get("texts", [])
            det = "yes" if len(set(sha8s)) == 1 else "**NO**"
            gt_any = any(GT_TITLE in t.lower() for t in texts)
            gt_all = GT_LINES_SHORT if "control_short" in script else GT_LINES_FULL
            scores = [gt_score(t, gt_all) for t in texts]
            gt_lines_best = max((h for h, _ in scores), default=0)
            warm = ev.get("warmup_sha8", "")
            rows.append(
                f"| {state} (`{head[:9]}`) | {script} | {', '.join(sha8s)} | {det} |"
                f" {'yes' if gt_any else '**NO**'} | {gt_lines_best}/{len(gt_all)} |"
                f" {warm or '—'} |"
            )
            for i, t in enumerate(texts):
                snippets.append(
                    f"### {state} / {script} / run {i} — sha8 `{sha8s[i]}`\n\n"
                    + "```text\n"
                    + t[:400]
                    + ("\n...(truncated)" if len(t) > 400 else "")
                    + "\n```\n"
                )
        markers = sdir / "routing-markers.txt"
        if markers.exists() and markers.read_text().strip():
            snippets.insert(0, "")  # noop, markers are folded into SUMMARY text
    print("| state | script | 3× sha8 | deterministic | GT title in any run | GT lines best | warmup sha8 |")
    print("|---|---|---|---|---|---|---|")
    print("\n".join(rows))
    print()
    print("\n".join(snippets))
    if not ok:
        sys.exit(1)


if __name__ == "__main__":
    main()
