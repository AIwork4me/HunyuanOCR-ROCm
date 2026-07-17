# tests/test_runner.py
import json
from pathlib import Path
import pytest

from hunyuan_ocr import runner


def test_write_atomic_creates_final_and_no_partial(tmp_path):
    out = tmp_path / "page.md"
    runner.write_atomic(out, "# hello")
    assert out.read_text(encoding="utf-8") == "# hello"
    assert not (tmp_path / "page.md.partial").exists()


def test_write_atomic_is_atomic_on_error(tmp_path, monkeypatch):
    out = tmp_path / "page.md"
    import os as _os

    real_replace = _os.replace

    def boom(src, dst):
        # fail the rename step
        raise OSError("simulated rename failure")

    monkeypatch.setattr(_os, "replace", boom)
    with pytest.raises(OSError):
        runner.write_atomic(out, "data")
    # no final file, and the .partial was cleaned up
    assert not out.exists()
    assert not (tmp_path / "page.md.partial").exists()


def test_write_atomic_creates_parent_dir(tmp_path):
    out = tmp_path / "nested" / "deep" / "page.md"
    runner.write_atomic(out, "x")
    assert out.exists()


def test_record_error_writes_structured_record(tmp_path):
    try:
        raise ValueError("boom")
    except ValueError as e:
        runner.record_error(tmp_path, "stem1",
                            image_path="/x/y.png", backend="vllm",
                            endpoint="127.0.0.1:8080", exc=e, attempt=2, ts=1.5)
    rec = json.loads((tmp_path / "_errors" / "stem1.json").read_text("utf-8"))
    assert rec["exception_type"] == "ValueError"
    assert rec["exception_message"] == "boom"
    assert rec["attempt"] == 2
    assert rec["backend"] == "vllm"
    assert rec["image_path"] == "/x/y.png"


def test_commit_success_writes_md_and_clears_stale_error(tmp_path):
    try:
        raise RuntimeError("first try failed")
    except RuntimeError as e:
        runner.record_error(tmp_path, "s", image_path="i", backend="b",
                            endpoint="e", exc=e, attempt=1)
    assert not runner.is_complete(tmp_path, "s")  # has error record
    runner.commit_success(tmp_path, "s", "# real output")
    assert runner.is_complete(tmp_path, "s")
    assert not (tmp_path / "_errors" / "s.json").exists()


def test_is_complete_false_for_missing_empty_error_partial(tmp_path):
    assert not runner.is_complete(tmp_path, "missing")
    (tmp_path / "empty.md").write_text("")
    assert not runner.is_complete(tmp_path, "empty")
    (tmp_path / "err.md").write_text("ERROR: ValueError: x")
    assert not runner.is_complete(tmp_path, "err")
    (tmp_path / "good.md").write_text("# fine")
    assert runner.is_complete(tmp_path, "good")


def test_is_complete_false_if_partial_only(tmp_path):
    (tmp_path / "p.md.partial").write_text("half")
    assert not runner.is_complete(tmp_path, "p")


def test_page_status_states(tmp_path):
    assert runner.page_status(tmp_path, "n") == "pending"
    runner.commit_success(tmp_path, "ok", "x")
    assert runner.page_status(tmp_path, "ok") == "complete"
    try:
        raise ValueError("z")
    except ValueError as e:
        runner.record_error(tmp_path, "bad", image_path="i", backend="b",
                            endpoint="e", exc=e, attempt=2)
    assert runner.page_status(tmp_path, "bad") == "failed"


def test_select_todo_default_resumes_and_retries_failed(tmp_path):
    items = [("a", "a.png"), ("b", "b.png"), ("c", "c.png"), ("d", "d.png")]
    runner.commit_success(tmp_path, "a", "ok")          # complete -> skip
    try:
        raise ValueError("x")
    except ValueError as e:
        runner.record_error(tmp_path, "b", image_path="b.png", backend="b",
                            endpoint="e", exc=e, attempt=1)  # failed -> retry
    # c pending, d pending
    todo, skipped = runner.select_todo(items, tmp_path)
    assert {s for s, _ in todo} == {"b", "c", "d"}
    assert skipped == 1


def test_select_todo_retry_failed_only(tmp_path):
    items = [("a", "a.png"), ("b", "b.png"), ("c", "c.png")]
    runner.commit_success(tmp_path, "a", "ok")
    try:
        raise ValueError("x")
    except ValueError as e:
        runner.record_error(tmp_path, "b", image_path="b.png", backend="b",
                            endpoint="e", exc=e, attempt=1)
    todo, skipped = runner.select_todo(items, tmp_path, retry_failed=True)
    assert {s for s, _ in todo} == {"b"}
    assert skipped == 2


def test_select_todo_overwrite(tmp_path):
    items = [("a", "a.png")]
    runner.commit_success(tmp_path, "a", "ok")
    todo, skipped = runner.select_todo(items, tmp_path, overwrite=True)
    assert todo == [("a", "a.png")] and skipped == 0


def test_detect_stem_conflicts(tmp_path):
    conflicts = runner.detect_stem_conflicts(["dirA/page-1.png", "dirB/page-1.png", "page-2.png"])
    assert len(conflicts) == 1
    stem, srcs = conflicts[0]
    assert stem == "page-1" and len(srcs) == 2


def test_decide_run_status():
    assert runner.decide_run_status(0, 0) == "ok"
    assert runner.decide_run_status(1, 0) == "failed"
    assert runner.decide_run_status(0, 1) == "failed"
    assert runner.decide_run_status(0, 0, worker_errors=1) == "failed"
    assert runner.decide_run_status(0, 0, crashed=1) == "failed"


def test_aggregate_errors_concatenates_records(tmp_path):
    for stem, msg in [("a", "e1"), ("b", "e2")]:
        try:
            raise ValueError(msg)
        except ValueError as e:
            runner.record_error(tmp_path, stem, image_path=stem + ".png",
                                backend="b", endpoint="e", exc=e, attempt=1)
    out = runner.aggregate_errors(tmp_path)
    lines = [json.loads(l) for l in out.read_text("utf-8").splitlines() if l.strip()]
    assert {l["exception_message"] for l in lines} == {"e1", "e2"}


def test_safe_argv_redacts_secrets():
    argv = ["--gt-json", "x.json", "--hf-token", "SECRET123",
            "--api-key=TOPSECRET", "--ports", "8000"]
    redacted = runner.safe_argv(argv)
    assert "SECRET123" not in redacted
    assert "TOPSECRET" not in redacted
    assert "--gt-json" in redacted and "x.json" in redacted and "8000" in redacted


def test_safe_argv_no_false_positive_on_monkey():
    # 'monkey' contains 'key' substring but is not a secret flag
    redacted = runner.safe_argv(["--monkey", "tail"])
    assert redacted == ["--monkey", "tail"]


def test_write_run_manifest_structure_and_no_secret(tmp_path):
    p = runner.write_run_manifest(
        tmp_path, backend="vllm", model="HYVL",
        counts={"expected": 3, "succeeded": 2, "failed": 1, "skipped": 0},
        ports=[8000, 8001], max_pixels=0, max_tokens=32768, status="failed")
    m = json.loads(p.read_text("utf-8"))
    assert m["backend"] == "vllm" and m["status"] == "failed"
    assert m["counts"] == {"expected": 3, "succeeded": 2, "failed": 1, "skipped": 0}
    assert m["ports"] == [8000, 8001]
    assert "torch" in m["env"]  # current env has torch
