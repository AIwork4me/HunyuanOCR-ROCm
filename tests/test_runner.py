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
