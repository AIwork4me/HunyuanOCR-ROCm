# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 AIwork4me
"""CPU tests for the GPU-CI bridge poller (commit-status model; no network, no GPU)."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]

from hunyuan_ocr.ci.models import (  # noqa: E402
    CHECK_NAME,
    POLL_INTERVAL_SEC,
    SMOKE_TIMEOUT_SEC,
    STALE_AFTER_SEC,
    SmokeResult,
    SmokeStatus,
    _parse_iso,
)


def test_constants_have_spec_values():
    assert CHECK_NAME == "gpu-smoke (gfx1100)"
    assert POLL_INTERVAL_SEC == 180
    assert STALE_AFTER_SEC == 30 * 60
    assert SMOKE_TIMEOUT_SEC == 20 * 60


def test_dataclasses_hold_fields():
    st = SmokeStatus(sha="abc", context=CHECK_NAME, state="pending", created_at="2026-07-18T03:00:00Z", target_url=None)
    assert st.state == "pending"
    sr = SmokeResult(
        ok=True, sha="abc", env_summary={"rocm": "7.2"}, manifest={"status": "ok"}, latency_sec=12.3, log_tail=""
    )
    assert sr.ok and sr.manifest["status"] == "ok"


def test_parse_iso_handles_z_suffix():
    assert _parse_iso("2026-07-18T03:00:00Z") == 1784343600.0
    assert _parse_iso("garbage") == 0.0
    assert _parse_iso(None) == 0.0


# --- decide() -----------------------------------------------------------------
from hunyuan_ocr.ci.poller import decide  # noqa: E402


def test_decide_runs_for_fresh_pending():
    assert decide("pending", age_sec=60) == "run"


def test_decide_skip_done_on_terminal():
    for terminal in ("success", "failure", "error"):
        assert decide(terminal, age_sec=60) == "skip_done"


def test_decide_timeout_when_stale_pending():
    assert decide("pending", age_sec=STALE_AFTER_SEC + 1) == "timeout"


# --- build_description() ------------------------------------------------------
from hunyuan_ocr.ci.poller import build_description  # noqa: E402


def test_build_description_success_under_140_with_env():
    sr = SmokeResult(
        ok=True,
        sha="abc",
        env_summary={"rocm": "7.2.1", "torch": "2.9.1", "gpu": "gfx1100"},
        manifest={"status": "ok", "final_state": {"expected": 1, "complete": 1, "failed": 0, "pending": 0}},
        latency_sec=42.5,
        log_tail="",
    )
    desc = build_description(sr)
    assert desc.startswith("PASSED") and "gfx1100" in desc and "complete=1" in desc
    assert len(desc) <= 140


def test_build_description_failure_includes_reason():
    sr = SmokeResult(
        ok=False,
        sha="abc",
        env_summary={},
        manifest=None,
        latency_sec=5.0,
        log_tail="line1\nserver did not become healthy",
    )
    desc = build_description(sr)
    assert desc.startswith("FAILED") and "server did not become healthy" in desc
    assert len(desc) <= 140


# --- GitHubClient (statuses) --------------------------------------------------
from hunyuan_ocr.ci.github import GitHubClient  # noqa: E402


class FakeGH:
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
    assert GitHubClient("o", "r", runner=FakeGH([])).ref_to_sha("a" * 40) == "a" * 40


def test_latest_tag_dereferences_annotated_tag_to_commit():
    gh = FakeGH(
        [
            json.dumps([{"ref": "refs/tags/v0.1.1", "object": {"sha": "tagobj", "type": "tag"}}]),
            json.dumps({"object": {"sha": "commitsha", "type": "commit"}}),
        ]
    )
    assert GitHubClient("o", "r", runner=gh).latest_tag() == ("v0.1.1", "commitsha")


def test_latest_tag_none_on_error():
    class Boom:
        def __call__(self, argv):
            raise RuntimeError("HTTP 404")

    assert GitHubClient("o", "r", runner=Boom()).latest_tag() is None


def test_list_smoke_statuses_filters_context_most_recent_first():
    payload = [
        {
            "context": "gpu-smoke (gfx1100)",
            "state": "pending",
            "updated_at": "2026-07-18T03:01:00Z",
            "target_url": None,
        },
        {"context": "other", "state": "success", "updated_at": "2026-07-18T03:00:00Z", "target_url": None},
        {"context": "gpu-smoke (gfx1100)", "state": "success", "updated_at": "2026-07-18T02:00:00Z", "target_url": "u"},
    ]
    runs = GitHubClient("o", "r", runner=FakeGH([json.dumps(payload)])).list_smoke_statuses("s")
    assert [r.state for r in runs] == ["pending", "success"]  # only our context, recent first


def test_create_status_sends_state_and_description():
    gh = FakeGH(["{}"])
    GitHubClient("o", "r", runner=gh).create_status("s1", state="success", description="d", target_url="https://u")
    joined = " ".join(gh.calls[0])
    assert "/statuses/s1" in joined and "state=success" in joined and "context=gpu-smoke (gfx1100)" in joined
    assert "description=d" in joined and "target_url=https://u" in joined


# --- run_smoke() --------------------------------------------------------------
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
    repo_used = Path((out_dir / "repo_used.txt").read_text(encoding="utf-8").strip())
    assert repo_used.is_dir() and repo_used != harness.parent  # trust split


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
    assert res.ok is False and "server did not become healthy" in res.log_tail


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
    assert res.ok is False and "timed out" in res.log_tail.lower()


# --- once() + flock -----------------------------------------------------------
from hunyuan_ocr.ci.poller import _acquire_lock, once  # noqa: E402

NOW = 1784343600.0  # == 2026-07-18T03:00:00Z


def _iso(minutes_before_now: float) -> str:
    import datetime as _dt

    return (_dt.datetime.fromtimestamp(NOW, tz=_dt.timezone.utc) - _dt.timedelta(minutes=minutes_before_now)).strftime(
        "%Y-%m-%dT%H:%M:%SZ"
    )


@pytest.mark.skipif(sys.platform == "win32", reason="fcntl flock is POSIX")
def test_acquire_lock_second_call_returns_none(tmp_path):
    lock = tmp_path / "x.lock"
    assert _acquire_lock(lock) is not None
    assert _acquire_lock(lock) is None


class FakeClient:
    def __init__(self, statuses_by_sha, main_sha="mainsha", tags=None):
        self.statuses_by_sha = statuses_by_sha
        self._main_sha = main_sha
        self.tags = tags or []
        self.created: list = []

    def ref_to_sha(self, ref):
        return self._main_sha

    def latest_tag(self, prefix="v"):
        return self.tags[-1] if self.tags else None

    def list_smoke_statuses(self, sha):
        return self.statuses_by_sha.get(sha, [])

    def create_status(self, sha, *, state, description, target_url=""):
        self.created.append((sha, state))


def _st(state, created_minutes_before_now=0.0):
    return SmokeStatus("mainsha", CHECK_NAME, state, _iso(created_minutes_before_now), None)


def test_once_runs_pending_and_completes_success(tmp_path, monkeypatch):
    client = FakeClient({"mainsha": [_st("pending", 1)]})  # most-recent first
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
    assert client.created == [("mainsha", "success")]


def test_once_skip_done_when_terminal(tmp_path, monkeypatch):
    client = FakeClient({"mainsha": [_st("success", 1)]})
    monkeypatch.setattr("hunyuan_ocr.ci.poller.run_smoke", lambda sha, **k: pytest.fail("should not run"))
    s = once(
        client,
        trusted_smoke_script=tmp_path / "s.sh",
        workdir_parent=tmp_path,
        env={"HUNYUANOCR_SMOKE_OUT": str(tmp_path)},
        now=NOW,
    )
    assert s["skipped_done"] == ["mainsha"] and client.created == []


def test_once_stale_pending_times_out(tmp_path, monkeypatch):
    client = FakeClient({"mainsha": [_st("pending", 45)]})  # 45 min > 30 min STALE
    monkeypatch.setattr("hunyuan_ocr.ci.poller.run_smoke", lambda sha, **k: pytest.fail("should not run a stale job"))
    s = once(
        client,
        trusted_smoke_script=tmp_path / "s.sh",
        workdir_parent=tmp_path,
        env={"HUNYUANOCR_SMOKE_OUT": str(tmp_path)},
        now=NOW,
    )
    assert s["timed_out"] == ["mainsha"]
    assert client.created == [("mainsha", "failure")]


# --- main() -------------------------------------------------------------------
from hunyuan_ocr.ci import poller as poller_mod  # noqa: E402


def _fresh_st(state: str) -> SmokeStatus:
    import datetime as _dt

    return SmokeStatus(
        "mainsha", CHECK_NAME, state, _dt.datetime.now(_dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"), None
    )


def test_main_dry_run_does_not_mutate(tmp_path, monkeypatch, capsys):
    client = FakeClient({"mainsha": [_fresh_st("pending")]})
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
    out = capsys.readouterr().out
    assert rc == 0 and client.created == []
    assert "would run" in out and "mainsha" in out


def test_main_once_runs_one_pass(tmp_path, monkeypatch):
    client = FakeClient({"mainsha": [_fresh_st("pending")]})
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
    assert rc == 0 and client.created == [("mainsha", "success")]


# --- scripts/make_smoke_input.py ---------------------------------------------
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
    assert msi.select_one_page(full, manifest, imgs)["page_info"]["image_path"] == "b.png"


def test_select_one_page_raises_if_image_missing(tmp_path):
    with pytest.raises(FileNotFoundError):
        msi.select_one_page(
            [{"page_info": {"image_path": "a.png"}}], {"pages": [{"image_path": "a.png"}]}, tmp_path / "nope"
        )
