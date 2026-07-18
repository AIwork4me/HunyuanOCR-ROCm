# tests/test_phase2_no_gpu.py
import json
import sys
from pathlib import Path

import pytest


def _make_gt(tmp_path, stems):
    gt = tmp_path / "gt.json"
    gt.write_text(json.dumps([{"page_info": {"image_path": f"{s}.png"}} for s in stems]), encoding="utf-8")
    img = tmp_path / "images"
    img.mkdir()
    for s in stems:
        (img / f"{s}.png").write_bytes(b"x")
    return gt, img


def _import_driver():
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
    import run_phase2_vllm as drv

    return drv


def test_phase2_two_ok_one_failed_then_rerun(tmp_path, monkeypatch):
    drv = _import_driver()
    gt, img = _make_gt(tmp_path, ["a", "b", "c"])
    pred = tmp_path / "pred"

    # fake infer: succeed for a,b; raise for c on attempt 1, succeed on attempt 2 (rerun)
    state = {"c_tries": 0}

    def fake_infer(client, image_path, prompt, *, model, max_pixels):
        stem = Path(image_path).stem
        if stem == "c":
            state["c_tries"] += 1
            if state["c_tries"] == 1:
                raise RuntimeError("server 500")
        return f"# output for {stem}"

    monkeypatch.setattr(drv, "infer_one", fake_infer)
    # The driver imported OpenAI into its own namespace at module load, so patch
    # drv.OpenAI (not openai.OpenAI) to avoid even constructing a real client.
    monkeypatch.setattr(drv, "OpenAI", lambda *a, **k: object())
    # No live server in this CPU test: pretend every endpoint is healthy so the
    # circuit-breaking pool dispatches instead of fast-failing.
    monkeypatch.setattr(drv, "health_check", lambda url: True)

    # run 1: c fails (max-retries=1 -> one attempt) -> non-zero exit
    with pytest.raises(SystemExit) as ei:
        drv.main_with_args(
            [
                "--gt-json",
                str(gt),
                "--images-dir",
                str(img),
                "--pred-dir",
                str(pred),
                "--ports",
                "9999",
                "--concurrency",
                "2",
                "--max-retries",
                "1",
            ]
        )
    assert ei.value.code != 0
    assert (pred / "a.md").exists() and (pred / "b.md").exists()
    assert not (pred / "c.md").exists()
    assert (pred / "_errors" / "c.json").exists()
    assert (pred / "run_manifest.json").exists()

    # run 2: default resume skips a,b; retries c (now succeeds) -> exit 0
    drv.main_with_args(
        [
            "--gt-json",
            str(gt),
            "--images-dir",
            str(img),
            "--pred-dir",
            str(pred),
            "--ports",
            "9999",
            "--concurrency",
            "2",
            "--max-retries",
            "1",
        ]
    )
    assert (pred / "c.md").read_text("utf-8") == "# output for c"
    # stale error record must be cleared on success
    assert not (pred / "_errors" / "c.json").exists()


def test_phase2_conflict_aborts(tmp_path, monkeypatch):
    drv = _import_driver()
    # two distinct image paths with same stem
    gt = tmp_path / "gt.json"
    gt.write_text(
        json.dumps(
            [
                {"page_info": {"image_path": "dir1/x.png"}},
                {"page_info": {"image_path": "dir2/x.png"}},
            ]
        ),
        encoding="utf-8",
    )
    (tmp_path / "dir1").mkdir()
    (tmp_path / "dir1" / "x.png").write_bytes(b"")
    (tmp_path / "dir2").mkdir()
    (tmp_path / "dir2" / "x.png").write_bytes(b"")
    with pytest.raises(SystemExit):
        drv.main_with_args(
            [
                "--gt-json",
                str(gt),
                "--images-dir",
                str(tmp_path),
                "--pred-dir",
                str(tmp_path / "pred"),
                "--ports",
                "9999",
                "--max-retries",
                "1",
            ]
        )


def test_phase2_unexpected_crash_still_writes_manifest(tmp_path, monkeypatch):
    """A worker unexpected-exception must not abort before the manifest is written:
    the run_manifest.json exists, status is 'crashed', and the crash is recorded."""
    import hunyuan_ocr.runner as runner_mod

    drv = _import_driver()
    gt, img = _make_gt(tmp_path, ["a", "b"])
    pred = tmp_path / "pred"

    # infer_one fails for both pages (max-retries=1 -> one attempt each). Then the
    # crash is forced by making record_error raise an UNEXPECTED error, which is
    # outside work()'s bounded inner try -> propagates via f.result() -> captured.
    def failing_infer(client, image_path, prompt, *, model, max_pixels):
        raise RuntimeError("server 500")

    def exploding_record_error(*a, **k):
        raise RuntimeError("disk full while recording error")

    monkeypatch.setattr(drv, "infer_one", failing_infer)
    monkeypatch.setattr(drv, "OpenAI", lambda *a, **k: object())
    monkeypatch.setattr(drv, "health_check", lambda url: True)
    monkeypatch.setattr(runner_mod, "record_error", exploding_record_error)

    with pytest.raises(SystemExit) as ei:
        drv.main_with_args(
            [
                "--gt-json",
                str(gt),
                "--images-dir",
                str(img),
                "--pred-dir",
                str(pred),
                "--ports",
                "9999",
                "--concurrency",
                "1",
                "--max-retries",
                "1",
            ]
        )
    assert ei.value.code != 0
    # the manifest MUST exist even though the run crashed
    mp = pred / "run_manifest.json"
    assert mp.exists(), "run_manifest.json was not written after the crash"
    m = json.loads(mp.read_text("utf-8"))
    assert m["status"] == "crashed"
    assert m["extensions"]["crash"]["exception_type"] == "RuntimeError"
    assert "disk full" in m["extensions"]["crash"]["exception_message"]
    assert m["run_counts"]["interrupted"] >= 1
    # and it still satisfies the conservation laws
    assert runner_mod.validate_manifest(m) == []
