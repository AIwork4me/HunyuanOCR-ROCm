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
        id=7, head_sha="abc", status="queued", conclusion=None,
        started_at=None, external_id="abc", name=CHECK_NAME, created_at="2026-07-18T03:00:00Z",
    )
    assert cr.status == "queued" and cr.external_id == "abc"
    sr = SmokeResult(
        ok=True, sha="abc", env_summary={"rocm": "7.2"}, manifest={"status": "ok"},
        latency_sec=12.3, log_tail="",
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

    created = (_dt.datetime.fromtimestamp(NOW, tz=_dt.timezone.utc) - _dt.timedelta(minutes=created_minutes_before_now)).strftime("%Y-%m-%dT%H:%M:%SZ")
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

