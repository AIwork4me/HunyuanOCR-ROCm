# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 AIwork4me
"""Dataclasses + constants for the GPU-CI bridge. No I/O.

Uses GitHub **commit statuses** (not check-runs) because a user OAuth token can
read AND write statuses, whereas check-run writes require a GitHub App."""

from __future__ import annotations

import datetime
from dataclasses import dataclass

CHECK_NAME = "gpu-smoke (gfx1100)"
POLL_INTERVAL_SEC = 180
STALE_AFTER_SEC = 30 * 60
SMOKE_TIMEOUT_SEC = 20 * 60


@dataclass
class SmokeStatus:
    """One commit-status row for our context on a SHA. GitHub returns statuses
    most-recent-first, so statuses[0] is the latest state for the SHA."""

    sha: str
    context: str
    state: str  # "pending" | "success" | "failure" | "error"
    created_at: str | None
    target_url: str | None


@dataclass
class SmokeResult:
    ok: bool
    sha: str
    env_summary: dict
    manifest: dict | None
    latency_sec: float
    log_tail: str


def _parse_iso(iso) -> float:
    """Parse an ISO-8601 timestamp to epoch seconds. Returns 0.0 if unparseable
    (never raises), so callers can compute ages safely."""
    if not isinstance(iso, str) or not iso:
        return 0.0
    try:
        return datetime.datetime.fromisoformat(iso).timestamp()
    except ValueError:
        return 0.0
