#!/usr/bin/env python3
"""Run the transformers backend on the 150-page canary, then score it.

The canary is the project's minute-level regression oracle (Absorb-C):
later phases compare their canary score against this Phase-1 canary score.
Usage:
  python scripts/regression_canary.py --model /root/models/HunyuanOCR --gpu-ids 0,1,2
"""
from __future__ import annotations
import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = Path("/workspace/OmniDocBench_data")
CANARY_GT = DATA / "OmniDocBench_150.json"
PRED = Path("/root/hunyuanocr-results/canary-transformers/preds")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="/root/models/HunyuanOCR")
    p.add_argument("--gpu-ids", default="0,1,2")
    args = p.parse_args()

    subprocess.run([
        sys.executable, str(ROOT / "scripts" / "run_phase1_transformers.py"),
        "--gt-json", str(CANARY_GT), "--images-dir", str(DATA / "images"),
        "--pred-dir", str(PRED), "--model", args.model, "--gpu-ids", args.gpu_ids,
    ], check=True)

    subprocess.run([
        sys.executable, str(ROOT / "scripts" / "score_predictions.py"),
        "--pred-dir", str(PRED), "--gt-json", str(CANARY_GT), "--label", "transformers-canary-150",
    ], check=True)


if __name__ == "__main__":
    main()
