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
    print(f"  Overall          : {s['overall']:.2f}")
    print(f"  text  EditDist   : {s['text_edit_dist']:.4f}   -> {(1-s['text_edit_dist'])*100:.2f}")
    print(f"  formula CDM      : {s['formula_cdm']:.4f}   -> {s['formula_cdm']*100:.2f}")
    print(f"  table  TEDS      : {s['table_teds']:.4f}   -> {s['table_teds']*100:.2f}")
    print(f"  order  EditDist  : {s['reading_order_edit']:.4f}")
    recomputed = scoring.overall_score({"text_edit_dist": s["text_edit_dist"], "formula_cdm": s["formula_cdm"], "table_teds": s["table_teds"]})
    print(f"  (overall recomputed: {recomputed:.2f})")


if __name__ == "__main__":
    main()
