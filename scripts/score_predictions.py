#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 AIwork4me
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
import tempfile
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
    p.add_argument("--skip-validation", action="store_true", help="DANGEROUS: bypass pre-score validation")
    args = p.parse_args(argv)

    if args.skip_validation:
        print("WARNING: validation bypassed -- score may be invalid", file=sys.stderr)
    else:
        rep = validate_predictions(args.gt_json, args.pred_dir, strict=True)
        if not rep.ok_strict:
            print(
                f"[validation] {len(rep.errors())} error(s), {len(rep.warnings())} warning(s); refusing to score:",
                file=sys.stderr,
            )
            for prob in rep.problems:
                tag = "ERROR" if prob.severity == "error" else "WARN"
                print(f"  [{tag}] {prob.code}: {prob.message}", file=sys.stderr)
            sys.exit(
                "[error] predictions invalid; fix them or re-run the driver "
                "(use --skip-validation to override at your own risk)"
            )

    # Write the eval config into a PRIVATE temp dir, never the prediction dir,
    # so repeated scoring passes validation (no stray _eval_config.yaml that the
    # strict validator would flag as unexpected_file). The config still POINTS at
    # the real prediction dir; only the config file itself is ephemeral.
    save_name = f"{Path(args.pred_dir).name}_quick_match"
    with tempfile.TemporaryDirectory(prefix="hunyuanocr_eval_") as tmpd:
        cfg_path = Path(tmpd) / "_eval_config.yaml"
        scoring.write_eval_config(gt_json=args.gt_json, pred_dir=args.pred_dir, out_yaml=cfg_path)
        res = scoring.run_scorer(
            omnidocbench_repo=args.omnidocbench_repo, config_yaml=str(cfg_path), venv_python=args.venv_python
        )
        if res.returncode != 0:
            print(res.stdout[-4000:])
            print(res.stderr[-4000:], file=sys.stderr)
            sys.exit(f"[error] scorer failed (rc={res.returncode})")
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
    recomputed = scoring.overall_score(
        {"text_edit_dist": s["text_edit_dist"], "formula_cdm": s["formula_cdm"], "table_teds": s["table_teds"]}
    )
    print(f"  (overall recomputed: {'n/a' if recomputed is None else f'{recomputed:.2f}'})")


def main():
    main_with_args(sys.argv[1:])


if __name__ == "__main__":
    main()
