# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 AIwork4me
"""GPU-CI bridge poller orchestration (pure logic; I/O via GitHubClient)."""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import tempfile
import time
from pathlib import Path

from hunyuan_ocr.ci.github import GitHubClient
from hunyuan_ocr.ci.models import (
    POLL_INTERVAL_SEC,
    SMOKE_TIMEOUT_SEC,
    STALE_AFTER_SEC,
    CheckRun,
    SmokeResult,
    _parse_iso,
)


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
        f"{label} {val}"
        for label, val in (
            ("ROCm", env.get("rocm")),
            ("torch", env.get("torch")),
            ("llama.cpp", env.get("llama_cpp_commit")),
            ("gpu", env.get("gpu")),
        )
        if val
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
    Runtime-only (uses HUNYUANOCR_ROCM_DIR); patched in tests."""
    repo = os.environ.get("HUNYUANOCR_ROCM_DIR")
    if not repo:
        raise RuntimeError("HUNYUANOCR_ROCM_DIR must point at the box repo checkout to materialize a SHA")
    subprocess.run(["git", "-C", repo, "fetch", "origin", sha], check=False, capture_output=True)
    subprocess.run(["git", "-C", repo, "worktree", "add", "--detach", str(dest), sha], check=True)


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
            ["bash", str(trusted_smoke_script)],
            env=full_env,
            capture_output=True,
            text=True,
            timeout=timeout_s,
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
        return SmokeResult(
            ok=ok,
            sha=sha,
            env_summary=env_summary,
            manifest=manifest,
            latency_sec=time.monotonic() - start,
            log_tail=log_tail,
        )
    except subprocess.TimeoutExpired as exc:
        return SmokeResult(
            ok=False,
            sha=sha,
            env_summary={},
            manifest=None,
            latency_sec=time.monotonic() - start,
            log_tail=f"smoke timed out after {timeout_s}s: {exc}",
        )
    finally:
        subprocess.run(["git", "worktree", "remove", "--force", str(workdir)], check=False, capture_output=True)


def _acquire_lock(path):
    """Non-blocking exclusive flock. Returns the held fd (close to release) or None."""
    import fcntl

    try:
        fd = os.open(str(path), os.O_CREAT | os.O_RDWR, 0o644)
    except OSError:
        return None
    try:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        os.close(fd)
        return None
    return fd


def once(client, *, trusted_smoke_script, workdir_parent, env, now):
    """One polling pass. Returns a summary dict {ran, skipped_done, timed_out}."""
    summary = {"ran": [], "skipped_done": [], "timed_out": []}
    watched = [("main", client.ref_to_sha("main"))]
    tag = client.latest_tag()
    if tag:
        watched.append(tag)  # (name, sha)

    seen: set[str] = set()
    for _name, sha in watched:
        if sha in seen:
            continue
        seen.add(sha)
        runs = client.list_check_runs(sha)
        has_completed = any(r.status == "completed" for r in runs)
        for q in [r for r in runs if r.status == "queued"]:
            action = decide(q, has_completed_for_sha=has_completed, now=now)
            if action == "skip_done":
                summary["skipped_done"].append(sha)
                continue
            if action == "timeout":
                client.complete(
                    q.id,
                    conclusion="failure",
                    title="gpu-smoke FAILED",
                    summary="timed out waiting for the gfx1100 runner — is the Radeon Cloud box online?",
                )
                summary["timed_out"].append(sha)
                continue
            client.set_in_progress(q.id)
            result = run_smoke(sha, trusted_smoke_script=trusted_smoke_script, workdir_parent=workdir_parent, env=env)
            title, smry = build_output(result)
            client.complete(q.id, conclusion="success" if result.ok else "failure", title=title, summary=smry)
            summary["ran"].append(sha)
    return summary


def _driven_once(client, *, trusted_smoke_script, workdir_parent, env, now, dry_run):
    """Dry-run-aware wrapper: prints would-do instead of mutating when dry_run."""
    if not dry_run:
        return once(client, trusted_smoke_script=trusted_smoke_script, workdir_parent=workdir_parent, env=env, now=now)
    watched = [("main", client.ref_to_sha("main"))]
    tag = client.latest_tag()
    if tag:
        watched.append(tag)
    for _n, sha in watched:
        for r in client.list_check_runs(sha):
            if r.status == "queued":
                print(f"[dry-run] would run smoke for {sha} (check-run {r.id})", flush=True)
    return {"ran": [], "skipped_done": [], "timed_out": [], "dry_run": True}


def _resolve_smoke_script(arg: str | None) -> str:
    if arg:
        return arg
    repo = os.environ.get("HUNYUANOCR_ROCM_DIR")
    if not repo:
        raise SystemExit("[fatal] --smoke-script not given and HUNYUANOCR_ROCM_DIR is unset")
    return str(Path(repo) / "scripts" / "rocm_smoke.sh")


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="hunyuan_ocr.ci.poller", description="GPU-CI bridge poller")
    p.add_argument("--owner", default="AIwork4me")
    p.add_argument("--repo", default="HunyuanOCR-ROCm")
    p.add_argument(
        "--smoke-script",
        default=None,
        help="trusted rocm_smoke.sh (default: $HUNYUANOCR_ROCM_DIR/scripts/rocm_smoke.sh)",
    )
    p.add_argument("--workdir-parent", default="/tmp")
    p.add_argument("--lock", default=os.path.expanduser("~/.rocm_ci_poller.lock"))
    p.add_argument("--interval", type=int, default=POLL_INTERVAL_SEC)
    p.add_argument("--once", action="store_true", help="run a single pass and exit")
    p.add_argument("--dry-run", action="store_true", help="report what would run; no mutations")
    args = p.parse_args(argv)

    smoke_script = _resolve_smoke_script(args.smoke_script)
    client = GitHubClient(args.owner, args.repo)
    env = dict(os.environ)
    while True:
        fd = _acquire_lock(args.lock)
        if fd is None:
            print("[poller] another pass holds the lock; skipping", flush=True)
        else:
            try:
                _driven_once(
                    client,
                    trusted_smoke_script=smoke_script,
                    workdir_parent=args.workdir_parent,
                    env=env,
                    now=time.time(),
                    dry_run=args.dry_run,
                )
            except Exception as exc:  # noqa: BLE001 — a daemon must not die on one bad pass
                print(f"[poller] pass failed (will retry next interval): {exc}", flush=True)
            finally:
                os.close(fd)
        if args.once or args.dry_run:
            return 0
        time.sleep(args.interval)


if __name__ == "__main__":
    raise SystemExit(main())
