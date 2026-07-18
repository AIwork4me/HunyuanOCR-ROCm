# tests/test_runner.py
import json
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
        runner.record_error(
            tmp_path,
            "stem1",
            image_path="/x/y.png",
            backend="vllm",
            endpoint="127.0.0.1:8080",
            exc=e,
            attempt=2,
            ts=1.5,
        )
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
        runner.record_error(tmp_path, "s", image_path="i", backend="b", endpoint="e", exc=e, attempt=1)
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
        runner.record_error(tmp_path, "bad", image_path="i", backend="b", endpoint="e", exc=e, attempt=2)
    assert runner.page_status(tmp_path, "bad") == "failed"


def test_select_todo_default_resumes_and_retries_failed(tmp_path):
    items = [("a", "a.png"), ("b", "b.png"), ("c", "c.png"), ("d", "d.png")]
    runner.commit_success(tmp_path, "a", "ok")  # complete -> skip
    try:
        raise ValueError("x")
    except ValueError as e:
        runner.record_error(
            tmp_path, "b", image_path="b.png", backend="b", endpoint="e", exc=e, attempt=1
        )  # failed -> retry
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
        runner.record_error(tmp_path, "b", image_path="b.png", backend="b", endpoint="e", exc=e, attempt=1)
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
            runner.record_error(tmp_path, stem, image_path=stem + ".png", backend="b", endpoint="e", exc=e, attempt=1)
    out = runner.aggregate_errors(tmp_path)
    lines = [json.loads(row) for row in out.read_text("utf-8").splitlines() if row.strip()]
    assert {row["exception_message"] for row in lines} == {"e1", "e2"}


def test_safe_argv_redacts_secrets():
    argv = ["--gt-json", "x.json", "--hf-token", "SECRET123", "--api-key=TOPSECRET", "--ports", "8000"]
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
        tmp_path,
        backend="vllm",
        model="HYVL",
        run_counts={"attempted": 3, "succeeded": 2, "failed": 1, "skipped": 0},
        final_state={"expected": 3, "complete": 2, "failed": 1, "pending": 0},
        ports=[8000, 8001],
        max_pixels=0,
        max_tokens=32768,
        status="failed",
    )
    m = json.loads(p.read_text("utf-8"))
    assert m["schema_version"] == runner.MANIFEST_SCHEMA_VERSION
    assert m["backend"] == "vllm" and m["status"] == "failed"
    assert m["run_counts"] == {"attempted": 3, "succeeded": 2, "failed": 1, "skipped": 0, "interrupted": 0}
    assert m["final_state"] == {"expected": 3, "complete": 2, "failed": 1, "pending": 0}
    assert m["ports"] == [8000, 8001]
    assert "timestamp_iso" in m and m["timestamp_iso"].endswith("+00:00")
    assert m["extensions"] == {}  # extra is namespaced, never top-level
    # env is best-effort and OPTIONAL — present only when the dep is installed.
    # Never assert a specific package exists (env-independent test).
    assert isinstance(m["env"], dict)
    assert isinstance(m["platform"], dict) and "python" in m["platform"]
    # no secrets: a token-bearing flag value must be redacted
    assert all("TOPSECRET" != str(v) for v in m["command"])


def test_write_run_manifest_extra_is_namespaced_and_collision_rejected(tmp_path):
    # extra lands under 'extensions', not at the top level
    p = runner.write_run_manifest(
        tmp_path,
        backend="llamacpp",
        model="HYVL",
        run_counts={"attempted": 1, "succeeded": 1, "failed": 0, "skipped": 0},
        final_state={"expected": 1, "complete": 1, "failed": 0, "pending": 0},
        extra={"endpoints": [{"alias": "p1", "state": "closed"}]},
    )
    m = json.loads(p.read_text("utf-8"))
    assert "endpoints" not in m  # not top-level
    assert m["extensions"]["endpoints"] == [{"alias": "p1", "state": "closed"}]
    assert runner.validate_manifest(m) == []
    # a reserved core key in extra must be rejected, not silently overwrite it
    with pytest.raises(ValueError, match="reserved core field"):
        runner.write_run_manifest(
            tmp_path / "x",
            backend="llamacpp",
            model="HYVL",
            run_counts={"attempted": 1, "succeeded": 1, "failed": 0, "skipped": 0},
            final_state={"expected": 1, "complete": 1, "failed": 0, "pending": 0},
            extra={"status": "ok"},  # would clobber the real status -> rejected
        )


def test_manifest_invariants_hold(tmp_path):
    # Conservation laws: attempted == succeeded+failed+interrupted;
    # expected == attempted+skipped; expected == complete+failed+pending.
    for rc, fs in [
        (
            {"attempted": 5, "succeeded": 5, "failed": 0, "skipped": 0},
            {"expected": 5, "complete": 5, "failed": 0, "pending": 0},
        ),
        (
            {"attempted": 1, "succeeded": 1, "failed": 0, "skipped": 2},
            {"expected": 3, "complete": 3, "failed": 0, "pending": 0},
        ),  # partial resume
        (
            {"attempted": 2, "succeeded": 1, "failed": 1, "skipped": 0},
            {"expected": 2, "complete": 1, "failed": 1, "pending": 0},
        ),  # retry-failed (one page failed -> status is honestly 'failed')
    ]:
        # status must match the counts: 'ok' only when nothing failed or is pending.
        status = "ok" if (fs["failed"] == 0 and fs["pending"] == 0) else "failed"
        runner.write_run_manifest(tmp_path, backend="vllm", model="m", run_counts=rc, final_state=fs, status=status)
        m = json.loads((tmp_path / "run_manifest.json").read_text("utf-8"))
        assert runner.validate_manifest(m) == [], (rc, fs)


def test_manifest_invariants_violated(tmp_path):
    # The broken pre-fix shape: expected=3,succeeded=3,skipped=2 must be rejected.
    runner.write_run_manifest(
        tmp_path,
        backend="vllm",
        model="m",
        run_counts={"attempted": 3, "succeeded": 3, "failed": 0, "skipped": 2},
        final_state={"expected": 3, "complete": 3, "failed": 0, "pending": 0},
    )
    m = json.loads((tmp_path / "run_manifest.json").read_text("utf-8"))
    errs = runner.validate_manifest(m)
    assert errs and any("expected" in e and "attempted" in e for e in errs)


def test_manifest_works_without_torch(tmp_path):
    # Generating a manifest must not require torch/transformers/vllm. This test
    # runs in both the GPU env (where env may list them) and the no-torch CI venv
    # (where env is empty); either way manifest generation + validation must work.
    runner.write_run_manifest(
        tmp_path,
        backend="llamacpp",
        model="HYVL",
        run_counts={"attempted": 1, "succeeded": 1, "failed": 0, "skipped": 0},
        final_state={"expected": 1, "complete": 1, "failed": 0, "pending": 0},
    )
    m = json.loads((tmp_path / "run_manifest.json").read_text("utf-8"))
    assert m["backend"] == "llamacpp"
    assert runner.validate_manifest(m) == []


def _valid_manifest(**overrides):
    m = {
        "schema_version": 2,
        "repo_commit": "abc123",
        "backend": "llamacpp",
        "model": "HYVL",
        "timestamp_iso": "2026-07-17T12:00:00+00:00",
        "status": "ok",
        "run_counts": {"attempted": 2, "succeeded": 2, "failed": 0, "skipped": 0, "interrupted": 0},
        "final_state": {"expected": 2, "complete": 2, "failed": 0, "pending": 0},
    }
    m.update(overrides)
    return m


def test_validate_manifest_accepts_valid():
    assert runner.validate_manifest(_valid_manifest()) == []


def test_validate_manifest_rejects_non_object():
    assert runner.validate_manifest("not a dict")  # non-empty errors list
    assert runner.validate_manifest([1, 2, 3])


def test_validate_manifest_unknown_schema_version():
    errs = runner.validate_manifest(_valid_manifest(schema_version=99))
    assert any("schema_version" in e for e in errs)


def test_validate_manifest_v1_read_compat():
    # A legacy v1 manifest (no interrupted key, extra formerly top-level) must
    # still validate on read. interrupted defaults to 0.
    m = _valid_manifest()
    m["schema_version"] = 1
    del m["run_counts"]["interrupted"]
    assert runner.validate_manifest(m) == []


def test_validate_manifest_missing_run_counts():
    m = _valid_manifest()
    del m["run_counts"]
    errs = runner.validate_manifest(m)
    assert any("run_counts" in e for e in errs)


def test_validate_manifest_missing_single_count():
    m = _valid_manifest()
    del m["run_counts"]["failed"]
    errs = runner.validate_manifest(m)
    assert any("run_counts.failed is missing" in e for e in errs)


def test_validate_manifest_rejects_string_count():
    m = _valid_manifest()
    m["run_counts"]["attempted"] = "3"
    errs = runner.validate_manifest(m)
    assert any("run_counts.attempted" in e and "non-negative integer" in e for e in errs)


def test_validate_manifest_rejects_float_count():
    m = _valid_manifest()
    m["final_state"]["complete"] = 2.0
    errs = runner.validate_manifest(m)
    assert any("final_state.complete" in e and "non-negative integer" in e for e in errs)


def test_validate_manifest_rejects_bool_count():
    # booleans are a subclass of int in Python; they must NOT count as integers.
    m = _valid_manifest()
    m["run_counts"]["succeeded"] = True
    errs = runner.validate_manifest(m)
    assert any("run_counts.succeeded" in e and "non-negative integer" in e for e in errs)


def test_validate_manifest_rejects_empty_backend_or_model():
    for bad in ("backend", "model"):
        m = _valid_manifest(**{bad: ""})
        errs = runner.validate_manifest(m)
        assert any(bad in e and "non-empty string" in e for e in errs), bad


def test_validate_manifest_rejects_unparseable_timestamp():
    m = _valid_manifest(timestamp_iso="not-a-date")
    errs = runner.validate_manifest(m)
    assert any("timestamp_iso" in e for e in errs)


def test_validate_manifest_ok_with_failed_is_invalid():
    # status == ok must imply final_state.failed == 0 and pending == 0.
    m = _valid_manifest(status="ok")
    m["final_state"]["failed"] = 1
    m["run_counts"]["failed"] = 1  # keep run_counts arithmetic honest in isolation
    errs = runner.validate_manifest(m)
    assert any("status is 'ok' but final_state.failed" in e for e in errs)


def test_validate_manifest_conservation_with_interrupted():
    # A crashed run: dispatched 5, resolved 3 (2 ok, 1 failed), 2 interrupted.
    m = _valid_manifest(
        status="crashed",
        run_counts={"attempted": 5, "succeeded": 2, "failed": 1, "skipped": 0, "interrupted": 2},
        final_state={"expected": 5, "complete": 2, "failed": 1, "pending": 2},
    )
    assert runner.validate_manifest(m) == []
    # ...but dropping interrupted makes attempted(5) != succeeded(2)+failed(1)
    m2 = _valid_manifest(
        status="crashed",
        run_counts={"attempted": 5, "succeeded": 2, "failed": 1, "skipped": 0, "interrupted": 0},
        final_state={"expected": 5, "complete": 2, "failed": 1, "pending": 2},
    )
    errs = runner.validate_manifest(m2)
    assert any("attempted" in e and "interrupted" in e for e in errs)
