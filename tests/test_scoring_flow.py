# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 AIwork4me
"""Phase 2 acceptance: double-score works + backend name is recorded.

Mocks the OmniDocBench scorer (no GPU / no scorer venv needed).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

SCRIPTS = str(Path(__file__).resolve().parents[1] / "scripts")


def _valid_pred(tmp_path, stems=("a", "b")):
    gt = tmp_path / "gt.json"
    gt.write_text(json.dumps([{"page_info": {"image_path": f"{s}.png"}} for s in stems]), encoding="utf-8")
    pred = tmp_path / "pred"
    pred.mkdir()
    for s in stems:
        (pred / f"{s}.md").write_text(f"# output {s}")
    return gt, pred


def test_double_score_does_not_pollute_preddir(tmp_path, monkeypatch):
    """Scoring twice must both pass strict validation and leave no _eval_config.yaml
    behind in the prediction directory (Phase 2.1 regression)."""
    sys.path.insert(0, SCRIPTS)
    import score_predictions as sp
    from hunyuan_ocr import scoring

    gt, pred = _valid_pred(tmp_path)

    monkeypatch.setattr(scoring, "run_scorer", lambda **k: SimpleNamespace(returncode=0, stdout="", stderr=""))
    monkeypatch.setattr(
        scoring,
        "parse_run_summary",
        lambda *a, **k: {
            "overall": 90.0,
            "text_edit_dist": 0.05,
            "formula_cdm": 0.9,
            "table_teds": 0.9,
            "reading_order_edit": 0.1,
        },
    )

    sp.main_with_args(["--pred-dir", str(pred), "--gt-json", str(gt)])
    sp.main_with_args(["--pred-dir", str(pred), "--gt-json", str(gt)])  # second pass

    # the bug was: first pass wrote _eval_config.yaml into pred, second pass failed
    assert not (pred / "_eval_config.yaml").exists()
    assert (pred / "a.md").exists() and (pred / "b.md").exists()


def test_llamacpp_backend_recorded_in_manifest(tmp_path, monkeypatch):
    """A --backend-name llamacpp run must write backend=llamacpp, not vllm
    (Phase 2.2 regression)."""
    sys.path.insert(0, SCRIPTS)
    import run_phase2_vllm as drv

    gt = tmp_path / "gt.json"
    gt.write_text(json.dumps([{"page_info": {"image_path": "a.png"}}]), encoding="utf-8")
    img = tmp_path / "images"
    img.mkdir()
    (img / "a.png").write_bytes(b"x")
    pred = tmp_path / "pred"

    monkeypatch.setattr(drv, "infer_one", lambda client, image_path, prompt, **k: "# ok")
    monkeypatch.setattr(drv, "OpenAI", lambda *a, **k: object())
    monkeypatch.setattr(drv, "health_check", lambda url: True)

    drv.main_with_args(
        [
            "--backend-name",
            "llamacpp",
            "--server-alias",
            "HYVL",
            "--gt-json",
            str(gt),
            "--images-dir",
            str(img),
            "--pred-dir",
            str(pred),
            "--ports",
            "8081",
            "--concurrency",
            "1",
            "--max-retries",
            "1",
        ]
    )
    m = json.loads((pred / "run_manifest.json").read_text("utf-8"))
    assert m["backend"] == "llamacpp"
    assert m["backend_provenance"]["server_alias"] == "HYVL"
    assert "models" not in json.dumps(m)  # no credentials recorded
    assert m["run_counts"]["succeeded"] == 1
    assert m["final_state"]["complete"] == 1
