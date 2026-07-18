# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 AIwork4me
"""CPU tests for the GPU-CI bridge poller (no network, no GPU)."""

from __future__ import annotations

from pathlib import Path

REPO = Path(__file__).resolve().parents[1]

from hunyuan_ocr.ci.models import (  # noqa: E402
    CHECK_NAME,
    POLL_INTERVAL_SEC,
    SMOKE_TIMEOUT_SEC,
    STALE_AFTER_SEC,
    CheckRun,
    SmokeResult,
    _parse_iso,
)


def test_constants_have_spec_values():
    assert CHECK_NAME == "gpu-smoke (gfx1100)"
    assert POLL_INTERVAL_SEC == 180
    assert STALE_AFTER_SEC == 30 * 60
    assert SMOKE_TIMEOUT_SEC == 20 * 60


def test_checkrun_smoke_dataclasses_hold_fields():
    cr = CheckRun(
        id=7,
        head_sha="abc",
        status="queued",
        conclusion=None,
        started_at=None,
        external_id="abc",
        name=CHECK_NAME,
        created_at="2026-07-18T03:00:00Z",
    )
    assert cr.status == "queued" and cr.external_id == "abc"
    sr = SmokeResult(
        ok=True,
        sha="abc",
        env_summary={"rocm": "7.2"},
        manifest={"status": "ok"},
        latency_sec=12.3,
        log_tail="",
    )
    assert sr.ok and sr.manifest["status"] == "ok"


def test_parse_iso_handles_z_suffix():
    # 2026-07-18T03:00:00Z == epoch 1784343600.0 (UTC)
    assert _parse_iso("2026-07-18T03:00:00Z") == 1784343600.0
    assert _parse_iso("garbage") == 0.0  # unparseable -> 0, never raises
    assert _parse_iso(None) == 0.0


from hunyuan_ocr.ci.poller import decide  # noqa: E402

NOW = 1784343600.0  # == 2026-07-18T03:00:00Z


def _queued(created_minutes_before_now: float, cid: int = 1, sha: str = "abc") -> CheckRun:
    import datetime as _dt

    created = (
        _dt.datetime.fromtimestamp(NOW, tz=_dt.timezone.utc) - _dt.timedelta(minutes=created_minutes_before_now)
    ).strftime("%Y-%m-%dT%H:%M:%SZ")
    return CheckRun(cid, sha, "queued", None, None, sha, CHECK_NAME, created)


def test_decide_runs_for_fresh_queued():
    assert decide(_queued(5), has_completed_for_sha=False, now=NOW + 5 * 60) == "run"


def test_decide_skip_done_when_completed_exists():
    assert decide(_queued(5), has_completed_for_sha=True, now=NOW + 5 * 60) == "skip_done"


def test_decide_timeout_when_older_than_stale():
    # 45 min old > 30 min STALE_AFTER_SEC
    assert decide(_queued(45), has_completed_for_sha=False, now=NOW) == "timeout"


def test_decide_never_timeout_if_completed_exists():
    assert decide(_queued(45), has_completed_for_sha=True, now=NOW) == "skip_done"


from hunyuan_ocr.ci.poller import build_output  # noqa: E402


def test_build_output_success_includes_env_manifest_latency():
    sr = SmokeResult(
        ok=True,
        sha="abc1234",
        env_summary={"rocm": "7.2.1", "torch": "2.9.1", "llama_cpp_commit": "a320cbf", "gpu": "gfx1100"},
        manifest={
            "status": "ok",
            "run_counts": {"attempted": 1, "succeeded": 1, "failed": 0, "skipped": 0, "interrupted": 0},
            "final_state": {"expected": 1, "complete": 1, "failed": 0, "pending": 0},
        },
        latency_sec=42.5,
        log_tail="",
    )
    title, summary = build_output(sr)
    assert title == "gpu-smoke PASSED"
    assert "gfx1100" in summary and "ROCm 7.2.1" in summary
    assert "complete=1" in summary and "42.5s" in summary


def test_build_output_failure_includes_log_tail():
    sr = SmokeResult(
        ok=False, sha="abc", env_summary={}, manifest=None, latency_sec=5.0, log_tail="server did not become healthy"
    )
    title, summary = build_output(sr)
    assert title == "gpu-smoke FAILED"
    assert "server did not become healthy" in summary


import json  # noqa: E402

from hunyuan_ocr.ci.github import GitHubClient  # noqa: E402


class FakeGH:
    """Records the argv passed to `gh api` and returns canned JSON."""

    def __init__(self, responses):
        self.responses = list(responses)
        self.calls: list[list[str]] = []

    def __call__(self, argv):
        self.calls.append(list(argv))
        return self.responses.pop(0)


def test_ref_to_sha_calls_gh_and_parses():
    gh = FakeGH([json.dumps({"object": {"sha": "deadbeef"}})])
    c = GitHubClient("AIwork4me", "HunyuanOCR-ROCm", runner=gh)
    assert c.ref_to_sha("main") == "deadbeef"
    assert gh.calls[0][:3] == ["api", "repos/AIwork4me/HunyuanOCR-ROCm/git/refs/heads/main"]


def test_ref_to_sha_passthrough_for_full_sha():
    gh = FakeGH([])
    assert GitHubClient("o", "r", runner=gh).ref_to_sha("a" * 40) == "a" * 40
    assert gh.calls == []  # no API call for a raw SHA


def test_latest_tag_returns_name_and_sha():
    gh = FakeGH([json.dumps([{"ref": "refs/tags/v0.1.1", "object": {"sha": "tagsha"}}])])
    assert GitHubClient("o", "r", runner=gh).latest_tag() == ("v0.1.1", "tagsha")


def test_latest_tag_none_when_no_tags():
    gh = FakeGH([json.dumps([])])
    assert GitHubClient("o", "r", runner=gh).latest_tag() is None


def test_latest_tag_filters_non_v_and_picks_last():
    gh = FakeGH(
        [
            json.dumps(
                [
                    {"ref": "refs/tags/old", "object": {"sha": "x"}},
                    {"ref": "refs/tags/v0.1.1", "object": {"sha": "s1"}},
                    {"ref": "refs/tags/v0.1.2", "object": {"sha": "s2"}},
                ]
            )
        ]
    )
    assert GitHubClient("o", "r", runner=gh).latest_tag() == ("v0.1.2", "s2")


def test_latest_tag_degrades_to_none_on_api_error():
    class BoomGH:
        def __call__(self, argv):
            raise RuntimeError("HTTP 404")

    assert GitHubClient("o", "r", runner=BoomGH()).latest_tag() is None


def test_latest_tag_dereferences_annotated_tag_to_commit():
    # refs/tags/v0.1.1 points at a tag object (type=tag); git/tags/<obj> -> commit
    gh = FakeGH(
        [
            json.dumps([{"ref": "refs/tags/v0.1.1", "object": {"sha": "tagobj", "type": "tag"}}]),
            json.dumps({"object": {"sha": "commitsha", "type": "commit"}}),
        ]
    )
    assert GitHubClient("o", "r", runner=gh).latest_tag() == ("v0.1.1", "commitsha")


def test_list_check_runs_filters_to_check_name():
    payload = {
        "check_runs": [
            {
                "id": 1,
                "head_sha": "s",
                "status": "queued",
                "conclusion": None,
                "started_at": "2026-07-18T03:00:00Z",
                "external_id": "s",
                "name": "gpu-smoke (gfx1100)",
            },
            {
                "id": 2,
                "head_sha": "s",
                "status": "completed",
                "conclusion": "success",
                "started_at": None,
                "external_id": "s",
                "name": "other-check",
            },
        ]
    }
    gh = FakeGH([json.dumps(payload)])
    runs = GitHubClient("o", "r", runner=gh).list_check_runs("s")
    assert [r.id for r in runs] == [1]  # only our check name
    assert runs[0].created_at == "2026-07-18T03:00:00Z"  # created_at tracks started_at (for stale-sweep)


def test_complete_sends_conclusion_title_summary():
    gh = FakeGH(["{}"])
    GitHubClient("o", "r", runner=gh).complete(99, conclusion="success", title="T", summary="S")
    joined = " ".join(gh.calls[0])
    assert "check-runs/99" in joined and "success" in joined and "T" in joined and "S" in joined


import os  # noqa: E402

from hunyuan_ocr.ci.poller import run_smoke  # noqa: E402


def _fake_harness(tmp_path: Path, *, fail: bool = False, slow: bool = False) -> Path:
    if fail:
        body = "#!/usr/bin/env bash\necho 'server did not become healthy' >&2\nexit 1\n"
    elif slow:
        body = "#!/usr/bin/env bash\nsleep 30\n"
    else:
        body = (
            "#!/usr/bin/env bash\nset -euo pipefail\n"
            'mkdir -p "$HUNYUANOCR_SMOKE_OUT"\n'
            'echo "$REPO" > "$HUNYUANOCR_SMOKE_OUT/repo_used.txt"\n'
            'echo "ROCm 7.2.1"; echo "torch 2.9.1"; echo "llama.cpp a320cbf"; echo "gpu gfx1100"\n'
            'mkdir -p "$HUNYUANOCR_SMOKE_OUT/predictions"\n'
            'echo "# markdown" > "$HUNYUANOCR_SMOKE_OUT/predictions/page.md"\n'
            'echo \'{"status":"ok","run_counts":{"attempted":1,"succeeded":1,"failed":0,"skipped":0,"interrupted":0},'
            '"final_state":{"expected":1,"complete":1,"failed":0,"pending":0}}\' > "$HUNYUANOCR_SMOKE_OUT/predictions/run_manifest.json"\n'
        )
    script = tmp_path / "rocm_smoke.sh"
    script.write_text(body, encoding="utf-8")
    os.chmod(script, 0o755)
    return script


def test_run_smoke_success_records_trust_split_workdir(tmp_path, monkeypatch):
    harness = _fake_harness(tmp_path)
    out_dir = tmp_path / "out"
    monkeypatch.setattr("hunyuan_ocr.ci.poller._checkout_sha", lambda sha, dest: None)
    res = run_smoke(
        "abc1234",
        trusted_smoke_script=harness,
        workdir_parent=tmp_path,
        env={"HUNYUANOCR_SMOKE_OUT": str(out_dir)},
        timeout_s=30,
    )
    assert res.ok is True and res.sha == "abc1234"
    assert res.env_summary["gpu"] == "gfx1100" and res.manifest["status"] == "ok"
    # trust split: REPO was the per-run workdir (exists) and NOT the trusted harness location
    repo_used = Path((out_dir / "repo_used.txt").read_text(encoding="utf-8").strip())
    assert repo_used.is_dir()
    assert repo_used != harness.parent


def test_run_smoke_failure_captures_log_tail(tmp_path, monkeypatch):
    harness = _fake_harness(tmp_path, fail=True)
    monkeypatch.setattr("hunyuan_ocr.ci.poller._checkout_sha", lambda sha, dest: None)
    res = run_smoke(
        "abc",
        trusted_smoke_script=harness,
        workdir_parent=tmp_path,
        env={"HUNYUANOCR_SMOKE_OUT": str(tmp_path)},
        timeout_s=30,
    )
    assert res.ok is False
    assert "server did not become healthy" in res.log_tail


def test_run_smoke_timeout_is_failure(tmp_path, monkeypatch):
    harness = _fake_harness(tmp_path, slow=True)
    monkeypatch.setattr("hunyuan_ocr.ci.poller._checkout_sha", lambda sha, dest: None)
    res = run_smoke(
        "abc",
        trusted_smoke_script=harness,
        workdir_parent=tmp_path,
        env={"HUNYUANOCR_SMOKE_OUT": str(tmp_path)},
        timeout_s=1,
    )
    assert res.ok is False
    assert "timed out" in res.log_tail.lower()


import sys  # noqa: E402

import pytest  # noqa: E402

from hunyuan_ocr.ci.poller import _acquire_lock, once  # noqa: E402


@pytest.mark.skipif(sys.platform == "win32", reason="fcntl flock is POSIX")
def test_acquire_lock_second_call_returns_none(tmp_path):
    lock = tmp_path / "x.lock"
    fd1 = _acquire_lock(lock)
    assert fd1 is not None
    fd2 = _acquire_lock(lock)
    assert fd2 is None


class FakeClient:
    def __init__(self, runs_by_sha, main_sha="mainsha", tags=None):
        self.runs_by_sha = runs_by_sha
        self._main_sha = main_sha
        self.tags = tags or []
        self.completions: list = []

    def ref_to_sha(self, ref):
        return self._main_sha

    def latest_tag(self, prefix="v"):
        return self.tags[-1] if self.tags else None

    def list_check_runs(self, sha):
        return self.runs_by_sha.get(sha, [])

    def set_in_progress(self, cid):
        pass

    def complete(self, cid, *, conclusion, title, summary):
        self.completions.append((cid, conclusion))


def _cr(cid, sha, status, created="2026-07-18T03:00:00Z"):
    conclusion = "success" if status == "completed" else None
    return CheckRun(cid, sha, status, conclusion, None, sha, CHECK_NAME, created)


def test_once_runs_queued_and_completes_success(tmp_path, monkeypatch):
    client = FakeClient({"mainsha": [_cr(1, "mainsha", "queued")]})
    monkeypatch.setattr(
        "hunyuan_ocr.ci.poller.run_smoke",
        lambda sha, **k: SmokeResult(True, sha, {"gpu": "gfx1100"}, {"status": "ok"}, 1.0, ""),
    )
    s = once(
        client,
        trusted_smoke_script=tmp_path / "s.sh",
        workdir_parent=tmp_path,
        env={"HUNYUANOCR_SMOKE_OUT": str(tmp_path)},
        now=NOW,
    )
    assert s["ran"] == ["mainsha"]
    assert client.completions == [(1, "success")]


def test_once_skip_done_when_completed_exists(tmp_path, monkeypatch):
    client = FakeClient({"mainsha": [_cr(2, "mainsha", "completed"), _cr(1, "mainsha", "queued")]})
    ran = []
    monkeypatch.setattr(
        "hunyuan_ocr.ci.poller.run_smoke", lambda sha, **k: ran.append(sha) or SmokeResult(True, sha, {}, None, 1.0, "")
    )
    s = once(
        client,
        trusted_smoke_script=tmp_path / "s.sh",
        workdir_parent=tmp_path,
        env={"HUNYUANOCR_SMOKE_OUT": str(tmp_path)},
        now=NOW,
    )
    assert s["skipped_done"] == ["mainsha"] and ran == []
    assert client.completions == []


def test_once_stale_sweep_times_out_queued(tmp_path, monkeypatch):
    import datetime as _dt

    old_created = (_dt.datetime.fromtimestamp(NOW, tz=_dt.timezone.utc) - _dt.timedelta(minutes=45)).strftime(
        "%Y-%m-%dT%H:%M:%SZ"
    )
    client = FakeClient({"mainsha": [_cr(1, "mainsha", "queued", old_created)]})

    def boom(sha, **k):
        pytest.fail("should not run a stale job")

    monkeypatch.setattr("hunyuan_ocr.ci.poller.run_smoke", boom)
    s = once(
        client,
        trusted_smoke_script=tmp_path / "s.sh",
        workdir_parent=tmp_path,
        env={"HUNYUANOCR_SMOKE_OUT": str(tmp_path)},
        now=NOW,
    )
    assert s["timed_out"] == ["mainsha"]
    assert client.completions == [(1, "failure")]


from hunyuan_ocr.ci import poller as poller_mod  # noqa: E402


def _fresh_created() -> str:
    import datetime as _dt

    return _dt.datetime.now(_dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def test_main_dry_run_does_not_mutate(tmp_path, monkeypatch, capsys):
    client = FakeClient({"mainsha": [_cr(1, "mainsha", "queued", _fresh_created())]})
    monkeypatch.setattr(poller_mod, "GitHubClient", lambda *a, **k: client)
    monkeypatch.setattr(poller_mod, "run_smoke", lambda sha, **k: SmokeResult(True, sha, {}, None, 1.0, ""))
    rc = poller_mod.main(
        [
            "--owner",
            "o",
            "--repo",
            "r",
            "--dry-run",
            "--workdir-parent",
            str(tmp_path),
            "--smoke-script",
            str(tmp_path / "s.sh"),
            "--lock",
            str(tmp_path / "l.lock"),
        ]
    )
    assert rc == 0
    assert client.completions == []  # dry-run never completes
    out = capsys.readouterr().out
    assert "would run" in out and "mainsha" in out


def test_main_once_runs_one_pass(tmp_path, monkeypatch):
    client = FakeClient({"mainsha": [_cr(1, "mainsha", "queued", _fresh_created())]})
    monkeypatch.setattr(poller_mod, "GitHubClient", lambda *a, **k: client)
    monkeypatch.setattr(
        poller_mod, "run_smoke", lambda sha, **k: SmokeResult(True, sha, {"gpu": "gfx1100"}, {"status": "ok"}, 1.0, "")
    )
    rc = poller_mod.main(
        [
            "--owner",
            "o",
            "--repo",
            "r",
            "--once",
            "--workdir-parent",
            str(tmp_path),
            "--smoke-script",
            str(tmp_path / "s.sh"),
            "--lock",
            str(tmp_path / "l.lock"),
        ]
    )
    assert rc == 0
    assert client.completions == [(1, "success")]


# --- scripts/make_smoke_input.py (1-page slicer) ------------------------------
import importlib.util  # noqa: E402

_msi_spec = importlib.util.spec_from_file_location("make_smoke_input", REPO / "scripts" / "make_smoke_input.py")
msi = importlib.util.module_from_spec(_msi_spec)
_msi_spec.loader.exec_module(msi)


def test_select_one_page_picks_canary_first_page_and_verifies_image(tmp_path):
    full = [{"page_info": {"image_path": "a.png"}}, {"page_info": {"image_path": "b.png"}}]
    manifest = {"pages": [{"image_path": "b.png"}, {"image_path": "a.png"}]}
    imgs = tmp_path / "images"
    imgs.mkdir()
    (imgs / "a.png").write_bytes(b"x")
    (imgs / "b.png").write_bytes(b"x")
    page = msi.select_one_page(full, manifest, imgs)
    assert page["page_info"]["image_path"] == "b.png"  # manifest order, first page


def test_select_one_page_raises_if_image_missing(tmp_path):
    full = [{"page_info": {"image_path": "a.png"}}]
    manifest = {"pages": [{"image_path": "a.png"}]}
    with pytest.raises(FileNotFoundError):
        msi.select_one_page(full, manifest, tmp_path / "nope")
