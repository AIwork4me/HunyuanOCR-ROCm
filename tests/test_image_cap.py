# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 AIwork4me
"""Phase 3.3: client-side image cap must not pollute the (possibly read-only)
source dataset directory."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

pytest.importorskip("PIL")  # pillow is a core dep; skip if absent in minimal envs

from hunyuan_ocr.backends.vllm_client import _maybe_cap_image  # noqa: E402


def _make_png(path: Path, w: int, h: int):
    from PIL import Image

    Image.new("RGB", (w, h), (255, 0, 0)).save(path)


def test_cap_does_not_write_next_to_source(tmp_path, monkeypatch):
    monkeypatch.setenv("HUNYUANOCR_CAP_CACHE", str(tmp_path / "cache"))
    src = tmp_path / "dataset" / "img.png"
    src.parent.mkdir()
    _make_png(src, 2000, 2000)  # 4M px > 1M cap
    out = _maybe_cap_image(str(src), 1_000_000)
    # output lives in the cache, never beside the source
    assert str(tmp_path / "cache") in out
    assert not (src.parent / "img.png.cap1000000.png").exists()
    assert os.path.dirname(out) != str(src.parent)


def test_small_image_returned_unchanged(tmp_path, monkeypatch):
    monkeypatch.setenv("HUNYUANOCR_CAP_CACHE", str(tmp_path / "cache"))
    src = tmp_path / "small.png"
    _make_png(src, 100, 100)
    assert _maybe_cap_image(str(src), 1_000_000) == str(src)


def test_cap_caches_reuse(tmp_path, monkeypatch):
    monkeypatch.setenv("HUNYUANOCR_CAP_CACHE", str(tmp_path / "cache"))
    src = tmp_path / "big.png"
    _make_png(src, 2000, 2000)
    out1 = _maybe_cap_image(str(src), 1_000_000)
    out2 = _maybe_cap_image(str(src), 1_000_000)
    assert out1 == out2  # stable cache key -> same path
