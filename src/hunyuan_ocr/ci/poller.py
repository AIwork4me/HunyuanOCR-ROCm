# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 AIwork4me
"""GPU-CI bridge poller orchestration (pure logic; I/O via GitHubClient)."""

from __future__ import annotations

import json
import os
import re
import subprocess
import tempfile
import time
from pathlib import Path

from hunyuan_ocr.ci.models import SMOKE_TIMEOUT_SEC, STALE_AFTER_SEC, CheckRun, SmokeResult, _parse_iso


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


def _checkout_sha(sha: str, dest: Path) -> None:
    """Materialize <sha> of the box repo into dest as a detached worktree.
    Runtime-only; patched in tests."""
    repo = Path(os.environ.get("HUNYUANOCR_ROCM_DIR", "/workspace/HunyuanOCR-ROCm"))
    subprocess.run(["git", "-C", str(repo), "fetch", "origin", sha], check=False, capture_output=True)
    subprocess.run(["git", "-C", str(repo), "worktree", "add", "--detach", str(dest), sha], check=True)


def _parse_env_summary(log: str) -> dict:
    """Scrape env markers the harness prints (ROCm x.y / torch x / llama.cpp <sha> / gpu gfxXXXX)."""
    out: dict[str, str] = {}
    for key, pat in (
        ("rocm", r"ROCm\s+([\d.]+)"),
        ("torch", r"torch\s+([\d.\w+]+)"),
        ("llama_cpp_commit", r"llama\.cpp\s+([0-9a-f]{7,40})"),
        ("gpu", r"gpu\s+(gfx\d+)"),
    ):
        m = re.search(pat, log)
        if m:
            out[key] = m.group(1)
    return out


def run_smoke(sha, *, trusted_smoke_script, workdir_parent, env, timeout_s=SMOKE_TIMEOUT_SEC):
    """Checkout <sha> into a temp workdir and run the TRUSTED smoke script with
    REPO=<workdir>. The harness (lifecycle/assertions) is trusted; the model
    driver under the workdir is the dispatched code. Returns a SmokeResult."""
    start = time.monotonic()
    workdir = Path(tempfile.mkdtemp(prefix="gpu-smoke-", dir=str(workdir_parent)))
    log_tail = ""
    try:
        _checkout_sha(sha, workdir)
        full_env = {**os.environ, **env, "REPO": str(workdir)}
        cp = subprocess.run(
            ["bash", str(trusted_smoke_script)], env=full_env,
            capture_output=True, text=True, timeout=timeout_s,
        )
        combined = (cp.stdout or "") + (cp.stderr or "")
        ok = cp.returncode == 0
        if not ok:
            log_tail = combined
        env_summary = _parse_env_summary(combined)
        manifest = None
        out_dir = Path(env.get("HUNYUANOCR_SMOKE_OUT", "")) / "predictions"
        mp = out_dir / "run_manifest.json"
        if mp.is_file():
            try:
                manifest = json.loads(mp.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                manifest = None
        return SmokeResult(ok=ok, sha=sha, env_summary=env_summary, manifest=manifest,
                            latency_sec=time.monotonic() - start, log_tail=log_tail)
    except subprocess.TimeoutExpired as exc:
        return SmokeResult(ok=False, sha=sha, env_summary={}, manifest=None,
                            latency_sec=time.monotonic() - start,
                            log_tail=f"smoke timed out after {timeout_s}s: {exc}")
    finally:
        subprocess.run(["git", "worktree", "remove", "--force", str(workdir)],
                       check=False, capture_output=True)


