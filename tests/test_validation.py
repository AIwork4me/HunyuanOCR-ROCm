# tests/test_validation.py
import json

from hunyuan_ocr import validation


def _gt(tmp_path, stems):
    pages = [{"page_info": {"image_path": f"{s}.png"}} for s in stems]
    p = tmp_path / "gt.json"
    p.write_text(json.dumps(pages), encoding="utf-8")
    return p


def test_clean_dir_passes(tmp_path):
    gt = _gt(tmp_path, ["a", "b"])
    pred = tmp_path / "pred"
    pred.mkdir()
    (pred / "a.md").write_text("ok a")
    (pred / "b.md").write_text("ok b")
    r = validation.validate_predictions(gt, pred)
    assert r.ok and r.ok_strict and r.expected == 2 and r.valid == 2


def test_missing_pages(tmp_path):
    gt = _gt(tmp_path, ["a", "b", "c"])
    pred = tmp_path / "pred"
    pred.mkdir()
    (pred / "a.md").write_text("ok")
    r = validation.validate_predictions(gt, pred)
    assert not r.ok
    codes = {p.code for p in r.errors()}
    assert "missing" in codes


def test_empty_error_partial_markers(tmp_path):
    gt = _gt(tmp_path, ["a", "b", "c", "d"])
    pred = tmp_path / "pred"
    pred.mkdir()
    (pred / "a.md").write_text("")  # empty
    (pred / "b.md").write_text("ERROR: ValueError: x")  # error marker
    (pred / "c.md").write_text("ok")
    (pred / "d.md.partial").write_text("half")  # leftover partial
    r = validation.validate_predictions(gt, pred)
    assert not r.ok
    codes = {p.code for p in r.errors()}
    assert {"empty", "error_marker", "partial", "missing"} <= codes


def test_unresolved_error_record(tmp_path):
    gt = _gt(tmp_path, ["a"])
    pred = tmp_path / "pred"
    pred.mkdir()
    (pred / "_errors").mkdir()
    (pred / "_errors" / "a.json").write_text(json.dumps({"stem": "a"}))
    r = validation.validate_predictions(gt, pred)
    assert not r.ok and "unresolved_error" in {p.code for p in r.errors()}


def test_unexpected_file_warning(tmp_path):
    gt = _gt(tmp_path, ["a"])
    pred = tmp_path / "pred"
    pred.mkdir()
    (pred / "a.md").write_text("ok")
    (pred / "junk.txt").write_text("??")
    r = validation.validate_predictions(gt, pred)
    assert r.ok is True  # no hard error
    assert r.ok_strict is False  # warning present under strict
    assert "unexpected_file" in {p.code for p in r.warnings()}


def test_duplicate_stem_in_gt(tmp_path):
    gt = _gt(tmp_path, ["dup", "dup"])
    pred = tmp_path / "pred"
    pred.mkdir()
    r = validation.validate_predictions(gt, pred)
    assert not r.ok and "duplicate_stem" in {p.code for p in r.errors()}
