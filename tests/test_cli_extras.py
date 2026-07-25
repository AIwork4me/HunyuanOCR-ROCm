# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 AIwork4me
"""CPU tests for the hunyuan-ocr benchmark + report subcommands (no network/GPU)."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from hunyuan_ocr import cli

# --- benchmark ---------------------------------------------------------------


def test_benchmark_prints_lock_results(tmp_path, capsys):
    lock = tmp_path / "reproducibility.lock.yaml"
    lock.write_text(
        "benchmark:\n  canary_148:\n    vllm_overall: 94.81\n    llamacpp_overall: 93.33\n",
        encoding="utf-8",
    )
    rc = cli.main(["benchmark", "--lock", str(lock)])
    assert rc == 0
    out = capsys.readouterr().out
    assert "94.81" in out and "93.33" in out and "BEGIN GENERATED RESULTS" in out


def test_benchmark_missing_lock(tmp_path):
    rc = cli.main(["benchmark", "--lock", str(tmp_path / "nope.yaml")])
    assert rc == 2


# --- report ------------------------------------------------------------------


def _write_manifest(pred_dir: Path):
    manifest = {
        "schema_version": 2,
        "repo_commit": "abc123",
        "backend": "llamacpp",
        "model": "HYVL",
        "timestamp_iso": "2026-07-18T03:00:00Z",
        "status": "ok",
        "run_counts": {"attempted": 1, "succeeded": 1, "failed": 0, "skipped": 0, "interrupted": 0},
        "final_state": {"expected": 1, "complete": 1, "failed": 0, "pending": 0},
        "command": ["run_inference.py", "--backend-name", "llamacpp"],
        "env": {"torch": "2.9.1"},
        "platform": {"python": "3.12.3"},
    }
    pred_dir.mkdir(parents=True, exist_ok=True)
    (pred_dir / "run_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def test_report_assembles_bundle_with_checksums(tmp_path):
    pred = tmp_path / "pred"
    _write_manifest(pred)
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    (repo_root / "reproducibility.lock.yaml").write_text("hunyuanocr_rocm:\n  commit: x\n", encoding="utf-8")
    out = tmp_path / "artifact"

    rc = cli.main(["report", "--pred-dir", str(pred), "--out", str(out), "--repo-root", str(repo_root)])
    assert rc == 0

    assert (out / "run_manifest.json").is_file()
    assert (out / "environment.json").is_file()
    assert (out / "commands.txt").is_file()
    assert (out / "reproducibility.lock.yaml").is_file()
    assert (out / "README.md").is_file()
    # checksums cover every other file and verify
    sums = (out / "checksums.sha256").read_text(encoding="utf-8").strip().splitlines()
    assert len(sums) == 5
    for line in sums:
        digest, name = line.split("  ", 1)
        assert hashlib.sha256((out / name).read_bytes()).hexdigest() == digest


def test_report_missing_manifest(tmp_path):
    pred = tmp_path / "empty"
    pred.mkdir()
    rc = cli.main(["report", "--pred-dir", str(pred), "--out", str(tmp_path / "o"), "--repo-root", str(tmp_path)])
    assert rc == 2
