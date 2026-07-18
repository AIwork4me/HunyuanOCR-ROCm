#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 AIwork4me
"""Score a predictions dir against OmniDocBench v1.6 and print the score table.

Thin wrapper over :func:`hunyuan_ocr.scoring.score_directory` (shared with the
``hunyuan-ocr score`` CLI). VALIDATES the prediction dir first
(missing/empty/ERROR/.partial/unresolved-error pages block scoring). Use
``--skip-validation`` ONLY for debugging.

Usage:
  python scripts/score_predictions.py \\
      --pred-dir ./predictions --gt-json /path/to/OmniDocBench.json [--label x]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from hunyuan_ocr import scoring  # noqa: E402


def main_with_args(argv):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--pred-dir", required=True)
    p.add_argument("--gt-json", required=True)
    p.add_argument("--label", default="backend")
    p.add_argument("--omnidocbench-repo", default=scoring.DEFAULT_OMNIDOCBENCH_REPO)
    p.add_argument("--venv-python", default=scoring.DEFAULT_VENV_PYTHON)
    p.add_argument("--skip-validation", action="store_true", help="DANGEROUS: bypass pre-score validation")
    args = p.parse_args(argv)

    if args.skip_validation:
        print("WARNING: validation bypassed -- score may be invalid", file=sys.stderr)

    try:
        result = scoring.score_directory(
            gt_json=args.gt_json,
            pred_dir=args.pred_dir,
            omnidocbench_repo=args.omnidocbench_repo,
            venv_python=args.venv_python,
            skip_validation=args.skip_validation,
        )
    except scoring.ScoringError as exc:
        sys.exit(f"[error] {exc}")

    print(scoring.format_score_table(args.label, result["metrics"]))


def main():
    main_with_args(sys.argv[1:])


if __name__ == "__main__":
    main()
