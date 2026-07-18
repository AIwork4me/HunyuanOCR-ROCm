# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 AIwork4me
"""Dataclasses + constants for the GPU-CI bridge. No I/O."""

from __future__ import annotations

import datetime
from dataclasses import dataclass

CHECK_NAME = "gpu-smoke (gfx1100)"
POLL_INTERVAL_SEC = 180
STALE_AFTER_SEC = 30 * 60
SMOKE_TIMEOUT_SEC = 20 * 60


@dataclass
class CheckRun:
    id: int
    head_sha: str
    status: str  # "queued" | "in_progress" | "completed"
    conclusion: str | None
    started_at: str | None
    external_id: str | None
    name: str
    created_at: str | None


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
        return datetime.datetime.fromisoformat(iso.replace("Z", "+00:00")).timestamp()
    except ValueError:
        return 0.0
