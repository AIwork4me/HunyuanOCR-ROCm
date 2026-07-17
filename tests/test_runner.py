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
