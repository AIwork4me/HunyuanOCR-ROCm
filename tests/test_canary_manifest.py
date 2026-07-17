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
    # file order preserved (load-bearing for byte-identical materialization)
    assert [p["stem"] for p in d["pages"]] == ["page-3", "page-1", "page-2"]
    assert d["serialization"] == "json_compact_utf8"
    assert "source_json_sha256" in d and len(d["source_json_sha256"]) == 64
    # manifest_sha256 MUST be the sha of the canonical JSON with the field
    # omitted, so a reader can drop it and recompute to verify integrity.
    sha = m.manifest_sha256(d)  # d has no manifest_sha256 field
    d_with = dict(d)
    d_with["manifest_sha256"] = "deadbeef"  # field present with any value
    # manifest_sha256 MUST ignore its own field -> same sha whether or not it's present
    assert m.manifest_sha256(d_with) == sha
