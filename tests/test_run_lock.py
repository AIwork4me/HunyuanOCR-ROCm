# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 AIwork4me
"""CPU unit tests for the prediction-directory writer lock (Phase 2.5).

POSIX only (fcntl.flock); these tests are skipped on non-POSIX platforms.
"""

from __future__ import annotations

import json
import sys

import pytest

from hunyuan_ocr import runner

pytestmark = pytest.mark.skipif(sys.platform == "win32", reason="RunLock uses fcntl.flock (POSIX only)")


def test_two_writers_second_refused(tmp_path):
    a = runner.acquire_run_lock(tmp_path, command=["w1"])
    a.__enter__()
    try:
        b = runner.acquire_run_lock(tmp_path, command=["w2"])
        with pytest.raises(runner.RunLockHeld):
            b.__enter__()
    finally:
        a.__exit__(None, None, None)
    # after release, a fresh acquire succeeds
    with runner.acquire_run_lock(tmp_path, command=["w3"]):
        pass


def test_stale_lock_file_reclaimed(tmp_path):
    # a lock file whose holder is long dead (no live flock) must be reclaimed,
    # not treated as a permanent blocker.
    (tmp_path / runner.RunLock.LOCK_NAME).write_text(
        json.dumps({"pid": 999999, "host": "dead", "started_iso": "2026-01-01T00:00:00+00:00", "command": "old"}),
        encoding="utf-8",
    )
    with runner.acquire_run_lock(tmp_path, command=["fresh"]):
        info = json.loads((tmp_path / runner.RunLock.LOCK_NAME).read_text("utf-8"))
        assert info["command"] == "fresh"


def test_release_removes_lock_file(tmp_path):
    with runner.acquire_run_lock(tmp_path, command=["x"]):
        assert (tmp_path / runner.RunLock.LOCK_NAME).exists()
    assert not (tmp_path / runner.RunLock.LOCK_NAME).exists()


def test_lockinfo_records_pid_host_command(tmp_path):
    with runner.acquire_run_lock(tmp_path, command=["run", "--flag", "v"]):
        info = json.loads((tmp_path / runner.RunLock.LOCK_NAME).read_text("utf-8"))
    assert info["pid"] > 0
    assert info["host"]
    assert info["started_iso"].endswith("+00:00")
    assert "run" in info["command"]
