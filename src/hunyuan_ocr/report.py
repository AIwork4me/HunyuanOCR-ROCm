# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 AIwork4me
"""Assemble a benchmark release-artifact bundle from a run manifest + the lock.

See docs/release-artifact.md for the layout. This module is pure filesystem +
stdlib (no GPU, no scorer); it packages the reproducibility evidence for a run:
the manifest, its environment + command, the lock, and a tamper-evident checksum
file. Metrics from the scorer are intentionally NOT produced here — run the
scorer separately and drop its output into the bundle.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    h.update(path.read_bytes())
    return h.hexdigest()


def assemble_release_artifact(pred_dir, out_dir, repo_root) -> Path:
    """Build the bundle at ``out_dir`` from ``pred_dir/run_manifest.json`` + the
    repo's ``reproducibility.lock.yaml``. Returns the output directory."""
    pred_dir = Path(pred_dir)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    manifest_path = pred_dir / "run_manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"no run_manifest.json in {pred_dir}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    # run_manifest.json (verbatim copy)
    (out_dir / "run_manifest.json").write_text(manifest_path.read_text(encoding="utf-8"), encoding="utf-8")

    # environment.json (best-effort env + platform from the manifest)
    (out_dir / "environment.json").write_text(
        json.dumps({"env": manifest.get("env", {}), "platform": manifest.get("platform", {})}, indent=2),
        encoding="utf-8",
    )

    # commands.txt (redacted argv already stored in the manifest)
    cmd = manifest.get("command")
    cmd_text = " ".join(cmd) + "\n" if isinstance(cmd, list) else str(cmd) + "\n"
    (out_dir / "commands.txt").write_text(cmd_text, encoding="utf-8")

    # REPRO.yaml (or legacy reproducibility.lock.yaml) — copy from the repo
    lock_src = Path(repo_root) / "REPRO.yaml"
    legacy_src = Path(repo_root) / "reproducibility.lock.yaml"
    if not lock_src.is_file() and legacy_src.is_file():
        lock_src = legacy_src
    if lock_src.is_file():
        (out_dir / lock_src.name).write_text(lock_src.read_text(encoding="utf-8"), encoding="utf-8")

    # README.md describing the bundle
    backend = manifest.get("backend", "?")
    status = manifest.get("status", "?")
    sha = manifest.get("repo_commit") or "?"
    (out_dir / "README.md").write_text(
        "# Benchmark release artifact\n\n"
        f"- backend: `{backend}`\n- status: `{status}`\n- repo commit: `{sha}`\n\n"
        "Reproduce by checking out the pinned commits + weights in "
        "`REPRO.yaml`, then running the commands in `commands.txt`.\n"
        "Verify integrity with `sha256sum -c checksums.sha256`.\n",
        encoding="utf-8",
    )

    # checksums.sha256 (over every other file in the bundle)
    lines = []
    for f in sorted(out_dir.iterdir()):
        if f.name == "checksums.sha256":
            continue
        lines.append(f"{_sha256(f)}  {f.name}")
    (out_dir / "checksums.sha256").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return out_dir
