"""OmniDocBench v1.6 scoring driver.

Writes an eval config, invokes the OmniDocBench scorer (pdf_validation.py) in its
own 3.11 venv, and parses the resulting metric_result.json / run_summary.json.
Overall = ((1 - text_EditDist)*100 + formula_CDM*100 + table_TEDS*100) / 3
(reading-order EditDist is reported separately, NOT part of Overall).
"""
from __future__ import annotations
import json
import subprocess
from pathlib import Path

import yaml

DEFAULT_VENV_PYTHON = "/root/ocr-eval/OmniDocBench/.venv/bin/python"
DEFAULT_OMNIDOCBENCH_REPO = "/root/ocr-eval/OmniDocBench"
TEMPLATE_CONFIG = Path(__file__).resolve().parents[2] / "eval" / "configs" / "hunyuanocr-1.5_linux-rocm.yaml"


def overall_score(metrics: dict) -> float:
    """v1.6 Overall from raw metrics (each in [0,1])."""
    text = metrics["text_edit_dist"]
    cdm = metrics["formula_cdm"]
    teds = metrics["table_teds"]
    return ((1.0 - text) * 100.0 + cdm * 100.0 + teds * 100.0) / 3.0


def write_eval_config(*, gt_json: str, pred_dir: str, out_yaml: Path) -> None:
    """Materialize an eval config from the template, substituting GT + pred paths."""
    cfg = yaml.safe_load(TEMPLATE_CONFIG.read_text(encoding="utf-8"))
    cfg["end2end_eval"]["dataset"]["ground_truth"]["data_path"] = str(gt_json)
    cfg["end2end_eval"]["dataset"]["prediction"]["data_path"] = str(pred_dir)
    out_yaml = Path(out_yaml)
    out_yaml.parent.mkdir(parents=True, exist_ok=True)
    out_yaml.write_text(yaml.safe_dump(cfg, allow_unicode=True, sort_keys=False), encoding="utf-8")


def run_scorer(*, omnidocbench_repo: str, config_yaml: str, venv_python: str | None = None) -> subprocess.CompletedProcess:
    """Run pdf_validation.py --config <cfg> inside the OmniDocBench repo."""
    py = venv_python or DEFAULT_VENV_PYTHON
    cmd = [py, "pdf_validation.py", "--config", str(config_yaml)]
    return subprocess.run(cmd, cwd=omnidocbench_repo, capture_output=True, text=True, check=False)


def parse_run_summary(result_dir: str | Path, save_name: str) -> dict:
    """Read overall + per-task numbers. save_name = basename(pred_dir) + '_quick_match'."""
    result_dir = Path(result_dir)
    metric = json.loads((result_dir / f"{save_name}_metric_result.json").read_text(encoding="utf-8"))
    summary = json.loads((result_dir / f"{save_name}_run_summary.json").read_text(encoding="utf-8"))
    return {
        "overall": summary["notebook_metric_summary"]["overall_notebook"],
        "text_edit_dist": metric["text_block"]["all"]["Edit_dist"]["ALL_page_avg"],
        "formula_cdm": metric["display_formula"]["page"]["CDM"]["ALL"],
        "table_teds": metric["table"]["page"]["TEDS"]["ALL"],
        "reading_order_edit": metric["reading_order"]["all"]["Edit_dist"]["ALL_page_avg"],
    }
