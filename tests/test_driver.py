# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 AIwork4me
"""CPU unit tests for the package-resident OpenAI-compatible driver (crash safety).

These exercise the crash/interrupt paths directly with fakes — no network, no
GPU, no torch. The normal predict path is covered by test_phase2_no_gpu.py.
"""

from __future__ import annotations

import json
from types import SimpleNamespace

from hunyuan_ocr import driver, runner


def test_run_workers_captures_unexpected_exception():
    """A future that raises an unexpected exception must not propagate; the crash
    is captured and the partial results returned so the manifest can be written."""
    todo = [("a", "a.png"), ("b", "b.png"), ("c", "c.png")]
    seen = {"n": 0}

    def work(item):
        idx, (stem, img) = item
        del idx, img
        seen["n"] += 1
        if seen["n"] == 3:
            raise RuntimeError("boom unexpected")
        return {"stem": stem, "status": "complete"}

    results, crash = driver.run_workers(todo, concurrency=1, work=work)
    assert crash is not None
    assert crash["kind"] == "crashed"
    assert crash["exception_type"] == "RuntimeError"
    assert "boom unexpected" in crash["exception_message"]
    assert crash.get("traceback_tail")
    # the two pages before the crash were collected
    assert len(results) == 2 and all(r["status"] == "complete" for r in results)


def test_run_workers_captures_keyboard_interrupt():
    def work(item):
        raise KeyboardInterrupt("ctrl-c")

    _results, crash = driver.run_workers([("a", "a.png")], concurrency=1, work=work)
    assert crash is not None
    assert crash["kind"] == "interrupted"


class _FakePool:
    def snapshot(self):
        return [{"alias": "port-1", "state": "closed", "half_open_in_flight": False}]


def _args(pred_dir):
    return SimpleNamespace(
        pred_dir=str(pred_dir),
        backend_name="llamacpp",
        model="HYVL",
        max_pixels=0,
        server_alias="HYVL",
        protocol="openai-chat-v1",
        host="127.0.0.1",
    )


def test_finalize_crashed_manifest_is_consistent(tmp_path):
    """A crashed run writes a manifest that still passes validate_manifest and
    records the crash, with `interrupted` accounting for the unresolved pages."""
    pred = tmp_path / "pred"
    pred.mkdir()
    pages = [("a", str(pred / "a.png")), ("b", str(pred / "b.png"))]
    runner.commit_success(pred, "a", "# a")  # 'a' completed on disk before the crash
    todo = pages  # both were dispatched
    results = [{"stem": "a", "status": "complete"}]  # only 'a' returned; 'b' interrupted
    crash = {
        "kind": "crashed",
        "exception_type": "RuntimeError",
        "exception_message": "boom",
        "traceback_tail": "Traceback ...\nRuntimeError: boom",
    }
    status = driver.finalize(
        _args(pred),
        _FakePool(),
        [("port-1", "http://127.0.0.1:8081/v1")],
        pages,
        todo,
        skipped=0,
        results=results,
        crash=crash,
        ports=[8081],
    )
    assert status == "crashed"
    m = json.loads((pred / "run_manifest.json").read_text("utf-8"))
    assert m["status"] == "crashed"
    assert m["run_counts"]["interrupted"] == 1  # 'b' dispatched but unresolved
    assert m["final_state"]["pending"] == 1  # 'b' left pending on disk
    assert m["extensions"]["crash"]["exception_type"] == "RuntimeError"
    # the crash record carries no secret-style keys and the manifest is self-consistent
    assert "extensions" in m and "endpoints" in m["extensions"]
    assert runner.validate_manifest(m) == []


def test_finalize_clean_run_is_ok(tmp_path):
    """The normal (non-crash) path still produces status=ok with interrupted=0."""
    pred = tmp_path / "pred"
    pred.mkdir()
    pages = [("a", str(pred / "a.png")), ("b", str(pred / "b.png"))]
    for s, _ in pages:
        runner.commit_success(pred, s, f"# {s}")
    results = [{"stem": s, "status": "complete"} for s, _ in pages]
    status = driver.finalize(
        _args(pred),
        _FakePool(),
        [("port-1", "http://127.0.0.1:8081/v1")],
        pages,
        pages,
        skipped=0,
        results=results,
        crash=None,
        ports=[8081],
    )
    assert status == "ok"
    m = json.loads((pred / "run_manifest.json").read_text("utf-8"))
    assert m["run_counts"]["interrupted"] == 0
    assert "crash" not in m["extensions"]
    assert runner.validate_manifest(m) == []
