# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 AIwork4me
"""Phase 6.1: canary materializer round-trips byte-identically from the full GT
using the manifest's page order. Uses synthetic data (no real dataset needed)."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from hunyuan_ocr import canary  # noqa: E402
import create_canary_manifest as cm  # noqa: E402


def _synthetic_full(tmp_path, names):
    pages = [{"page_info": {"image_path": n}, "layout_dets": [{"x": i}]} for i, n in enumerate(names)]
    full = tmp_path / "OmniDocBench.json"
    full.write_text(json.dumps(pages, ensure_ascii=False), encoding="utf-8")
    return full, pages


def test_materialize_byte_identical_roundtrip(tmp_path):
    full, all_pages = _synthetic_full(tmp_path, [f"p{i}.png" for i in range(5)])
    # canary subset = pages 2, 0, 4 in that deliberate (non-sorted) order
    subset_pages = [all_pages[2], all_pages[0], all_pages[4]]
    canon = tmp_path / "OmniDocBench_150.json"
    canon_bytes = json.dumps(subset_pages, ensure_ascii=False).encode("utf-8")
    canon.write_bytes(canon_bytes)

    manifest = tmp_path / "canary.manifest.json"
    d = cm.build_manifest(str(canon), name="canary-test", dataset="OmniDocBench", dataset_version="v1.6")
    d["manifest_sha256"] = cm.manifest_sha256(d)
    manifest.write_text(json.dumps(d, ensure_ascii=False, sort_keys=True, indent=2), encoding="utf-8")

    out = tmp_path / "materialized.json"
    sha = canary.materialize(str(full), str(manifest), str(out))
    import hashlib

    assert sha == hashlib.sha256(canon_bytes).hexdigest()
    assert out.read_bytes() == canon_bytes  # byte-identical


def test_materialize_rejects_missing_page(tmp_path):
    full, _ = _synthetic_full(tmp_path, ["a.png", "b.png"])
    # manifest references a page not in the full GT
    d = {
        "subset_name": "x",
        "expected_count": 1,
        "serialization": "json_compact_utf8",
        "pages": [{"stem": "ghost", "image_path": "ghost.png"}],
        "source_json_sha256": "0" * 64,
    }
    manifest = tmp_path / "m.json"
    manifest.write_text(json.dumps(d), encoding="utf-8")
    with pytest.raises(canary.CanaryError, match="not found"):
        canary.materialize(str(full), str(manifest), str(tmp_path / "out.json"))


def test_materialize_rejects_sha_mismatch(tmp_path):
    full, _ = _synthetic_full(tmp_path, ["a.png"])
    d = {
        "subset_name": "x",
        "expected_count": 1,
        "serialization": "json_compact_utf8",
        "pages": [{"stem": "a", "image_path": "a.png"}],
        "source_json_sha256": "deadbeef" + "0" * 56,
    }  # wrong sha
    manifest = tmp_path / "m.json"
    manifest.write_text(json.dumps(d), encoding="utf-8")
    with pytest.raises(canary.CanaryError, match="not byte-identical"):
        canary.materialize(str(full), str(manifest), str(tmp_path / "out.json"))
