# tests/test_score_gate.py
import json
import sys
from pathlib import Path


def test_scorer_refuses_invalid_dir(tmp_path, monkeypatch):
    # invalid pred dir: a page missing
    gt = tmp_path / "gt.json"
    gt.write_text(json.dumps([{"page_info": {"image_path": "a.png"}},
                              {"page_info": {"image_path": "b.png"}}]), "utf-8")
    pred = tmp_path / "pred"; pred.mkdir()
    (pred / "a.md").write_text("ok")   # b.md missing

    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
    import score_predictions as sp

    called = {"n": 0}
    def must_not_run(*a, **k):
        called["n"] += 1
        raise AssertionError("scorer must not run on invalid dir")
    monkeypatch.setattr(sp.scoring, "run_scorer", must_not_run)
    monkeypatch.setattr(sp.scoring, "parse_run_summary", lambda *a, **k: {})

    import pytest
    with pytest.raises(SystemExit):
        sp.main_with_args(["--pred-dir", str(pred), "--gt-json", str(gt)])
    assert called["n"] == 0
