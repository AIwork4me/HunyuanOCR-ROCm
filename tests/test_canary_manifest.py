# tests/test_canary_manifest.py
import json
import sys
from pathlib import Path


def _import():
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
    import create_canary_manifest as m
    return m


def test_build_manifest_structure_and_sha(tmp_path):
    m = _import()
    gt = tmp_path / "gt.json"
    pages = [{"page_info": {"image_path": f"page-{i}.png"}} for i in [3, 1, 2]]
    gt.write_text(json.dumps(pages), encoding="utf-8")
    d = m.build_manifest(str(gt), name="canary-test", dataset="OmniDocBench", dataset_version="v1.6")
    assert d["expected_count"] == 3
    assert d["subset_name"] == "canary-test"
    assert [p["stem"] for p in d["pages"]] == ["page-1", "page-2", "page-3"]  # sorted
    assert "source_json_sha256" in d and len(d["source_json_sha256"]) == 64
    # manifest_sha recomputes from the dict WITHOUT manifest_sha256
    sha = m.manifest_sha256(d)
    d2 = dict(d); d2["manifest_sha256"] = sha
    assert m.manifest_sha256(d) == sha
