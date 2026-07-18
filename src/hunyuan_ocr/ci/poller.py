# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 AIwork4me
"""GPU-CI bridge poller orchestration (pure logic; I/O via GitHubClient)."""

from __future__ import annotations

from hunyuan_ocr.ci.models import STALE_AFTER_SEC, CheckRun, _parse_iso


def decide(queued_run: CheckRun, has_completed_for_sha: bool, now: float) -> str:
    """Decide what to do with a queued gpu-smoke Check Run.

    Returns 'skip_done' if a completed smoke already exists for this SHA
    (idempotency), 'timeout' if it has waited longer than STALE_AFTER_SEC
    (no silent hangs), else 'run'.
    """
    if has_completed_for_sha:
        return "skip_done"
    created = _parse_iso(queued_run.created_at)
    if created and now - created > STALE_AFTER_SEC:
        return "timeout"
    return "run"
