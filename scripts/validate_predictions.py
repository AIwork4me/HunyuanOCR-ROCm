#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 AIwork4me
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
from hunyuan_ocr.validation import validate_predictions


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--gt-json", required=True)
    p.add_argument("--pred-dir", required=True)
    p.add_argument("--strict", action="store_true", default=True, help="warnings are fatal (default)")
    p.add_argument("--lenient", action="store_true", help="warnings are non-fatal")
    args = p.parse_args()
    strict = args.strict and not args.lenient

    r = validate_predictions(args.gt_json, args.pred_dir, strict=strict)
    print(f"expected={r.expected} valid={r.valid} errors={len(r.errors())} warnings={len(r.warnings())}")
    for prob in r.problems:
        tag = "ERROR" if prob.severity == "error" else "WARN "
        print(f"  [{tag}] {prob.code}: {prob.message}")
    ok = r.ok_strict if strict else r.ok
    print("RESULT: PASS" if ok else "RESULT: FAIL")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
