# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 AIwork4me
"""GPU-CI bridge poller orchestration over commit statuses (pure logic; I/O via
GitHubClient)."""

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
    SmokeResult,
    _parse_iso,
)

TERMINAL_STATES = ("success", "failure", "error")


def decide(latest_state: str, age_sec: float) -> str:
    """Decide what to do given the latest smoke status for a SHA.

    'skip_done' if the latest is already terminal (idempotency), 'timeout' if a
    pending has waited longer than STALE_AFTER_SEC (no silent hangs), else 'run'.
    """
    if latest_state in TERMINAL_STATES:
        return "skip_done"
    if age_sec > STALE_AFTER_SEC:
        return "timeout"
    return "run"


def build_description(result: SmokeResult) -> str:
    """Render a <=140-char commit-status description from a smoke result."""
    if not result.ok:
        reason = ""
        tail = result.log_tail.strip()
        if tail:
            reason = tail.splitlines()[-1][:80]
        return f"FAILED gfx1100: {reason}".strip()[:140]
    env = result.env_summary or {}
    bits = ["PASSED", "gfx1100"]
    if env.get("rocm"):
        bits.append(f"ROCm{env['rocm']}")
    if env.get("torch"):
        bits.append(f"torch{env['torch']}")
    if result.manifest:
        fs = result.manifest.get("final_state", {})
        bits.append(f"complete={fs.get('complete')}")
    bits.append(f"{result.latency_sec:.0f}s")
    return " ".join(bits)[:140]


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
    """One polling pass over commit statuses. Returns {ran, skipped_done, timed_out}."""
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
        statuses = client.list_smoke_statuses(sha)
        if not statuses:
            continue
        latest = statuses[0]  # most-recent-first
        age = now - _parse_iso(latest.created_at)
        action = decide(latest.state, age)
        if action == "skip_done":
            summary["skipped_done"].append(sha)
            continue
        if action == "timeout":
            client.create_status(
                sha,
                state="failure",
                description="timed out waiting for the gfx1100 runner — is the Radeon Cloud box online?",
            )
            summary["timed_out"].append(sha)
            continue
        result = run_smoke(sha, trusted_smoke_script=trusted_smoke_script, workdir_parent=workdir_parent, env=env)
        client.create_status(
            sha,
            state="success" if result.ok else "failure",
            description=build_description(result),
        )
        summary["ran"].append(sha)
    return summary


def _driven_once(client, *, trusted_smoke_script, workdir_parent, env, now, dry_run):
    """Dry-run-aware wrapper: prints would-do instead of mutating when dry_run."""
    if not dry_run:
        return once(
            client,
            trusted_smoke_script=trusted_smoke_script,
            workdir_parent=workdir_parent,
            env=env,
            now=now,
        )
    watched = [("main", client.ref_to_sha("main"))]
    tag = client.latest_tag()
    if tag:
        watched.append(tag)
    for _n, sha in watched:
        statuses = client.list_smoke_statuses(sha)
        if statuses and statuses[0].state == "pending":
            print(f"[dry-run] would run smoke for {sha} (pending status)", flush=True)
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
