import json
import subprocess
from pathlib import Path
from unittest.mock import patch

from hunyuan_ocr import scoring

FIX = Path(__file__).parent / "fixtures"


def test_overall_score_formula():
    # v1.6 Overall = ((1-text_edit)*100 + cdm*100 + teds*100) / 3
    metrics = {
        "text_edit_dist": 0.04,
        "formula_cdm": 0.94,
        "table_teds": 0.93,
    }
    assert abs(scoring.overall_score(metrics) - 94.33333333333333) < 1e-6


def test_write_eval_config_substitutes_pred_dir(tmp_path):
    gt = "/workspace/OmniDocBench_data/OmniDocBench.json"
    out = tmp_path / "c.yaml"
    scoring.write_eval_config(gt_json=gt, pred_dir="/tmp/preds", out_yaml=out)
    txt = out.read_text()
    assert "data_path: /tmp/preds" in txt
    assert "data_path: /workspace/OmniDocBench_data/OmniDocBench.json" in txt
    assert "quick_match" in txt


def test_parse_run_summary_reads_overall_and_per_task():
    res = scoring.parse_run_summary(FIX, save_name="mini")
    assert round(res["overall"], 2) == 94.33          # computed ((96+94+93)/3)=94.333
    assert res["text_edit_dist"] == 0.04
    assert res["formula_cdm"] == 0.94
    assert res["table_teds"] == 0.93
    assert res["reading_order_edit"] == 0.13


def test_overall_score_none_when_cdm_missing():
    # CDM absent (subset with no formula pages) -> Overall undefined
    assert scoring.overall_score({"text_edit_dist": 0.04, "formula_cdm": None, "table_teds": 0.93}) is None


def test_run_scorer_invokes_pdf_validation_with_venv_python():
    with patch("hunyuan_ocr.scoring.subprocess.run") as mock:
        mock.return_value = subprocess.CompletedProcess([], 0, "", "")
        scoring.run_scorer(
            omnidocbench_repo="/root/ocr-eval/OmniDocBench",
            config_yaml="/tmp/c.yaml",
            venv_python="/root/ocr-eval/OmniDocBench/.venv/bin/python",
        )
        cmd = mock.call_args[0][0]
        assert cmd[0] == "/root/ocr-eval/OmniDocBench/.venv/bin/python"
        assert cmd[1] == "pdf_validation.py"
        assert "--config" in cmd
