# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 AIwork4me
"""GPU-CI bridge poller orchestration (pure logic; I/O via GitHubClient)."""

from __future__ import annotations

from hunyuan_ocr.ci.models import STALE_AFTER_SEC, CheckRun, SmokeResult, _parse_iso


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


def build_output(result: SmokeResult) -> tuple[str, str]:
    """Render the Check Run output (title, markdown summary) from a smoke result."""
    env = result.env_summary or {}
    env_line = ", ".join(
        f"{label} {val}" for label, val in (
            ("ROCm", env.get("rocm")),
            ("torch", env.get("torch")),
            ("llama.cpp", env.get("llama_cpp_commit")),
            ("gpu", env.get("gpu")),
        ) if val
    )
    if result.manifest:
        rc = result.manifest.get("run_counts", {})
        fs = result.manifest.get("final_state", {})
        manifest_line = (
            f"status={result.manifest.get('status')} "
            f"attempted={rc.get('attempted')} succeeded={rc.get('succeeded')} "
            f"failed={rc.get('failed')} complete={fs.get('complete')} pending={fs.get('pending')}"
        )
    else:
        manifest_line = "manifest: (none)"
    title = "gpu-smoke PASSED" if result.ok else "gpu-smoke FAILED"
    summary = (
        f"- sha: `{result.sha}`\n"
        f"- env: {env_line or '(unrecorded)'}\n"
        f"- {manifest_line}\n"
        f"- latency: {result.latency_sec:.1f}s\n"
    )
    if not result.ok and result.log_tail:
        summary += f"\n**log tail:**\n```\n{result.log_tail[-1500:]}\n```"
    return title, summary

