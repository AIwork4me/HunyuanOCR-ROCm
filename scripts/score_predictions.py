#!/usr/bin/env python3
"""Score a predictions dir against OmniDocBench v1.6 and print the score table.

Usage:
  python scripts/score_predictions.py \
      --pred-dir /root/hunyuanocr-results/phase1-transformers/preds \
      --gt-json /workspace/OmniDocBench_data/OmniDocBench.json \
      [--label transformers] [--no-cdm]
"""
from __future__ import annotations
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from hunyuan_ocr import scoring


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--pred-dir", required=True)
    p.add_argument("--gt-json", required=True)
    p.add_argument("--label", default="backend")
    p.add_argument("--omnidocbench-repo", default=scoring.DEFAULT_OMNIDOCBENCH_REPO)
    p.add_argument("--venv-python", default=scoring.DEFAULT_VENV_PYTHON)
    args = p.parse_args()

    cfg_path = Path(args.pred_dir) / "_eval_config.yaml"
    scoring.write_eval_config(gt_json=args.gt_json, pred_dir=args.pred_dir, out_yaml=cfg_path)
    res = scoring.run_scorer(omnidocbench_repo=args.omnidocbench_repo, config_yaml=str(cfg_path), venv_python=args.venv_python)
    if res.returncode != 0:
        print(res.stdout[-4000:]); print(res.stderr[-4000:], file=sys.stderr)
        sys.exit(f"[error] scorer failed (rc={res.returncode})")

    save_name = f"{Path(args.pred_dir).name}_quick_match"
    s = scoring.parse_run_summary(Path(args.omnidocbench_repo) / "result", save_name)
    print(f"\n=== {args.label} — OmniDocBench v1.6 ===")
    def fmt(v, pct=False):
        if v is None:
            return "n/a"
        return f"{v * 100:.2f}" if pct else f"{v:.4f}"
    ov = s["overall"]
    print(f"  Overall          : {'n/a (CDM missing on this subset)' if ov is None else f'{ov:.2f}'}")
    print(f"  text  EditDist   : {fmt(s['text_edit_dist'])}   -> {fmt(s['text_edit_dist'], pct=True)}")
    print(f"  formula CDM      : {fmt(s['formula_cdm'])}   -> {fmt(s['formula_cdm'], pct=True)}")
    print(f"  table  TEDS      : {fmt(s['table_teds'])}   -> {fmt(s['table_teds'], pct=True)}")
    print(f"  order  EditDist  : {fmt(s['reading_order_edit'])}")
    recomputed = scoring.overall_score({"text_edit_dist": s["text_edit_dist"], "formula_cdm": s["formula_cdm"], "table_teds": s["table_teds"]})
    print(f"  (overall recomputed: {'n/a' if recomputed is None else f'{recomputed:.2f}'})")


if __name__ == "__main__":
    main()
