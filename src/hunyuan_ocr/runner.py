# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 AIwork4me
"""Prediction-integrity primitives shared by the phase drivers.

Centralizes the rules that prevent "false completion":
  * atomic .md writes (partial -> fsync -> rename), never an ERROR: file
  * structured per-page error records (_errors/<stem>.json)
  * resumability that skips only genuinely-complete pages
  * output-name conflict detection
  * per-run manifest

No GPU, no model deps. Pure filesystem + stdlib.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

ERROR_PREFIX = "ERROR:"
# Files/dirs this project itself may write into a prediction directory. The
# validator treats these as expected (never flags them as unexpected_file/dir).
_OWN_ARTIFACTS = {
    "_errors",  # per-page error records dir
    "_errors.jsonl",  # aggregated error log
    "run_manifest.json",  # per-run manifest
    "_metadata",  # project metadata dir (e.g. eval-config snapshots)
    "_eval_config.yaml",  # legacy: scorer used to write this here; tolerated if present
    ".run.lock",  # writer mutual-exclusion lock
}


def _partial_of(path: Path) -> Path:
    return Path(path).with_suffix(Path(path).suffix + ".partial")


def _fsync_dir(path: Path) -> None:
    """Best-effort fsync of the parent directory so the rename is durable.

    POSIX-only; silently skips on platforms without ``os.open``/directory fsync
    (e.g. some network filesystems reject fsync on a directory).
    """
    try:
        fd = os.open(str(path), os.O_RDONLY)
    except (OSError, ValueError):
        return
    try:
        os.fsync(fd)
    except OSError:
        pass
    finally:
        os.close(fd)


def write_atomic(path: Path, content: str) -> None:
    """Write ``content`` to ``path`` atomically.

    Writes ``<path>.partial`` first, flushes + fsyncs, then ``os.replace`` onto
    the final path, then fsyncs the parent directory so the rename survives a
    crash. On any error the ``.partial`` is removed and the exception re-raised.
    Callers that see the final path can trust it is complete.

    Concurrent writes to the same path are prevented by the caller holding the
    prediction-directory write lock (see :func:`acquire_run_lock`).
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = _partial_of(path)
    try:
        with open(partial, "w", encoding="utf-8") as f:
            f.write(content)
            f.flush()
            os.fsync(f.fileno())
        os.replace(partial, path)
        _fsync_dir(path.parent)
    except BaseException:
        try:
            partial.unlink()
        except FileNotFoundError:
            pass
        raise


def _error_path(pred_dir, stem: str, ext: str = ".md") -> Path:
    return Path(pred_dir) / "_errors" / f"{stem}.json"


def record_error(
    pred_dir, stem: str, *, image_path, backend, endpoint, exc, attempt: int, ts: float | None = None
) -> None:
    """Write ``_errors/<stem>.json`` (one file per page -> no concurrent-write race).

    The presence of this file means the page is FAILED. ``write_atomic`` is used
    so the record is never half-written.
    """
    ts = time.time() if ts is None else ts
    rec = {
        "image_path": str(image_path),
        "stem": stem,
        "backend": backend,
        "endpoint": str(endpoint),
        "exception_type": type(exc).__name__,
        "exception_message": str(exc),
        "attempt": attempt,
        "timestamp": ts,
    }
    write_atomic(_error_path(pred_dir, stem), json.dumps(rec, ensure_ascii=False, indent=2))


def commit_success(pred_dir, stem: str, md: str, *, ext: str = ".md") -> Path:
    """Atomically write the final prediction AND clear any stale error record.

    Preserves the invariant  COMPLETE <=> valid .md present AND no _errors/<stem>.json
    across retries: a page that failed attempt 1 then succeeded attempt 2 must not
    retain a stale error file. All success paths go through here, never raw write_atomic.
    """
    out = Path(pred_dir) / f"{stem}{ext}"
    write_atomic(out, md)
    try:
        _error_path(pred_dir, stem, ext).unlink()
    except FileNotFoundError:
        pass
    return out


def is_complete(pred_dir, stem: str, ext: str = ".md") -> bool:
    """True iff a valid prediction exists (non-empty, not ERROR:) and no unresolved error."""
    out = Path(pred_dir) / f"{stem}{ext}"
    if not out.is_file():
        return False
    try:
        if out.stat().st_size == 0:
            return False
        with open(out, "r", encoding="utf-8") as f:
            head = f.read(len(ERROR_PREFIX) + 32)
    except OSError:
        return False
    if head.lstrip().startswith(ERROR_PREFIX):
        return False
    if _error_path(pred_dir, stem, ext).exists():
        return False
    return True


def page_status(pred_dir, stem: str, ext: str = ".md") -> str:
    """'failed' | 'complete' | 'pending'."""
    if _error_path(pred_dir, stem, ext).exists():
        return "failed"
    if is_complete(pred_dir, stem, ext):
        return "complete"
    return "pending"


def select_todo(items, pred_dir, *, overwrite: bool = False, retry_failed: bool = False, ext: str = ".md"):
    """Build the run's todo list per the resume policy.

    items: iterable of (stem, image_path). Returns (todo, n_skipped).
      default      -> skip COMPLETE; run FAILED + PENDING (failed retried across runs)
      retry_failed -> run FAILED only
      overwrite    -> run everything
    """
    todo: list[tuple[str, str]] = []
    skipped = 0
    for stem, img in items:
        st = page_status(pred_dir, stem, ext)
        if overwrite:
            todo.append((stem, img))
        elif retry_failed:
            if st == "failed":
                todo.append((stem, img))
            else:
                skipped += 1
        else:
            if st == "complete":
                skipped += 1
            else:
                todo.append((stem, img))
    return todo, skipped


def detect_stem_conflicts(image_paths) -> list:
    """Return [(stem, [source_paths...])] for any stem produced by >1 distinct image."""
    seen: dict[str, list[str]] = {}
    for p in image_paths:
        stem = Path(p).stem
        seen.setdefault(stem, []).append(str(p))
    return [(stem, srcs) for stem, srcs in seen.items() if len(srcs) > 1]


def decide_run_status(final_failed: int, final_pending: int, worker_errors: int = 0, crashed: int = 0) -> str:
    """Pure exit decision shared by both drivers."""
    if final_failed or final_pending or worker_errors or crashed:
        return "failed"
    return "ok"


def aggregate_errors(pred_dir, out_name: str = "_errors.jsonl") -> Path:
    """Concatenate ``_errors/*.json`` into ``_errors.jsonl``. Call ONCE from main
    after all workers join (single writer). Uses write_atomic for safety."""
    edir = Path(pred_dir) / "_errors"
    out = Path(pred_dir) / out_name
    rows = []
    if edir.is_dir():
        for f in sorted(edir.glob("*.json")):
            try:
                rows.append(json.loads(f.read_text(encoding="utf-8")))
            except (OSError, json.JSONDecodeError):
                continue
    body = "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows)
    write_atomic(out, body)
    return out


# NOTE: --venv-python is redacted because it is a machine-local absolute path,
# not a credential — keeping it out of the portable manifest.
_SECRET_FLAGS = {
    "--token",
    "--api-key",
    "--apikey",
    "--key",
    "--password",
    "--secret",
    "--hf-token",
    "--hugging-face-token",
    "--venv-python",
}

MANIFEST_SCHEMA_VERSION = 1


def safe_argv(argv=None) -> list[str]:
    """Return argv with secret-bearing flag values redacted (exact flag match only)."""
    argv = list(sys.argv[1:] if argv is None else argv)
    out: list[str] = []
    i = 0
    while i < len(argv):
        tok = argv[i]
        if "=" in tok and tok.split("=", 1)[0] in _SECRET_FLAGS:
            out.append(f"{tok.split('=', 1)[0]}=<redacted>")
        elif tok in _SECRET_FLAGS and i + 1 < len(argv):
            out.append(tok)
            out.append("<redacted>")
            i += 1
        else:
            out.append(tok)
        i += 1
    return out


def _repo_root() -> Path:
    """Resolve the repository root from this package's location, not the cwd.

    Walks up from this file looking for a ``.git`` dir so the recorded commit is
    stable regardless of where the driver is launched from. Falls back to the
    conventional ``parents[2]`` (the dir above ``src/``) for editable installs.
    """
    here = Path(__file__).resolve()
    for parent in here.parents:
        if (parent / ".git").exists():
            return parent
    return here.parents[2]


def _git_head(repo=None) -> str | None:
    """Current HEAD of the repo containing this package (None if not a git repo)."""
    repo = Path(repo) if repo else _repo_root()
    try:
        cp = subprocess.run(["git", "-C", str(repo), "rev-parse", "HEAD"], capture_output=True, text=True, timeout=10)
        if cp.returncode == 0:
            return cp.stdout.strip() or None
    except (OSError, subprocess.SubprocessError):
        pass
    return None


def iso_utc(epoch=None) -> str:
    """ISO-8601 UTC timestamp for an epoch (default: now)."""
    import datetime

    epoch = time.time() if epoch is None else epoch
    return datetime.datetime.fromtimestamp(epoch, tz=datetime.timezone.utc).isoformat()


def _platform_info() -> dict:
    import platform

    return {
        "python": platform.python_version(),
        "system": platform.system(),
        "machine": platform.machine(),
    }


def _env_versions() -> dict:
    """Best-effort versions of optional ML deps. Missing packages are simply
    omitted — never raise — so manifest generation works in a torch-free env."""
    v: dict[str, str] = {}
    try:
        import torch  # type: ignore

        v["torch"] = getattr(torch, "__version__", None)
        hip = getattr(getattr(torch, "version", None), "hip", None)
        if hip:
            v["hip"] = hip
    except Exception:
        pass
    try:
        import transformers  # type: ignore

        v["transformers"] = getattr(transformers, "__version__", None)
    except Exception:
        pass
    try:
        import vllm  # type: ignore

        v["vllm"] = getattr(vllm, "__version__", None)
    except Exception:
        pass
    return {k: val for k, val in v.items() if val}


def validate_manifest(m: dict) -> list[str]:
    """Return a list of conservation-law violations in a manifest (empty == valid).

    Enforced invariants:
      run_counts:  attempted == succeeded + failed
      cross:       expected == attempted + skipped
      final_state: expected == complete + failed + pending
    All counts must be present (non-None) and non-negative.
    """
    errs: list[str] = []
    if m.get("schema_version") != MANIFEST_SCHEMA_VERSION:
        errs.append(f"schema_version must be {MANIFEST_SCHEMA_VERSION} (got {m.get('schema_version')!r})")
    rc = m.get("run_counts") or {}
    fs = m.get("final_state") or {}
    for label, d in (("run_counts", rc), ("final_state", fs)):
        for k, val in d.items():
            if val is None:
                errs.append(f"{label}.{k} is None")
            elif not isinstance(val, int) or isinstance(val, bool):
                errs.append(f"{label}.{k} is not an int ({val!r})")
            elif val < 0:
                errs.append(f"{label}.{k} is negative ({val})")
    if errs:
        return errs
    a, s, f, sk = rc["attempted"], rc["succeeded"], rc["failed"], rc["skipped"]
    exp, c, ff, p = fs["expected"], fs["complete"], fs["failed"], fs["pending"]
    if a != s + f:
        errs.append(f"run_counts: attempted({a}) != succeeded({s}) + failed({f})")
    if exp != a + sk:
        errs.append(f"expected({exp}) != attempted({a}) + skipped({sk})")
    if exp != c + ff + p:
        errs.append(f"final_state: expected({exp}) != complete({c}) + failed({ff}) + pending({p})")
    return errs


def write_run_manifest(
    pred_dir,
    *,
    backend: str,
    model: str,
    run_counts: dict,
    final_state: dict,
    model_revision: str | None = None,
    backend_provenance: dict | None = None,
    command: list[str] | None = None,
    ports=None,
    gpu_ids=None,
    max_pixels=None,
    max_tokens=None,
    status: str = "ok",
    extra: dict | None = None,
) -> Path:
    """Write ``run_manifest.json`` (atomic) with a conservation-checked schema.

    ``run_counts`` describes THIS run (attempted/succeeded/failed/skipped);
    ``final_state`` describes the directory AFTER this run
    (expected/complete/failed/pending). Invariants are enforced by
    :func:`validate_manifest`. No secrets are recorded (command via safe_argv).
    """
    rc = run_counts or {}
    fs = final_state or {}
    manifest = {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "repo_commit": _git_head(),
        "backend": backend,
        "backend_provenance": backend_provenance or {},
        "model": model,
        "model_revision": model_revision,
        "command": safe_argv() if command is None else command,
        "timestamp": time.time(),
        "timestamp_iso": iso_utc(),
        "run_counts": {
            "attempted": rc.get("attempted"),
            "succeeded": rc.get("succeeded"),
            "failed": rc.get("failed"),
            "skipped": rc.get("skipped"),
        },
        "final_state": {
            "expected": fs.get("expected"),
            "complete": fs.get("complete"),
            "failed": fs.get("failed"),
            "pending": fs.get("pending"),
        },
        "ports": ports,
        "gpu_ids": gpu_ids,
        "pixel_cap": max_pixels,
        "max_tokens": max_tokens,
        "env": _env_versions(),
        "platform": _platform_info(),
        "status": status,
    }
    if extra:
        manifest.update(extra)
    out = Path(pred_dir) / "run_manifest.json"
    write_atomic(out, json.dumps(manifest, ensure_ascii=False, indent=2))
    return out


# --- writer mutual-exclusion lock for a prediction directory -----------------
# POSIX (Linux/ROCm) only: uses fcntl.flock. On a platform without fcntl,
# acquire() raises a clear error rather than silently allowing concurrent writes.


class RunLockHeld(RuntimeError):
    """Another live writer holds the prediction-directory lock."""


@dataclass
class LockInfo:
    pid: int
    host: str
    started_iso: str
    command: str

    def is_alive(self) -> bool:
        try:
            os.kill(self.pid, 0)
        except ProcessLookupError:
            return False
        except PermissionError:
            return True  # process exists but is not owned by us
        except OSError:
            return False
        return True


class RunLock:
    """Exclusive writer lock on a prediction directory.

    The authority is ``fcntl.flock(LOCK_EX | LOCK_NB)``: if the holder process
    dies, the kernel releases the flock automatically, so the next acquire
    succeeds and overwrites the stale PID record — no manual cleanup needed for a
    crashed holder. The ``.run.lock`` file records PID/host/start/command for
    human inspection and is removed on graceful release. A still-running holder is
    never silently displaced (flock refuses the second acquire -> RunLockHeld).
    """

    LOCK_NAME = ".run.lock"

    def __init__(self, pred_dir, command=None) -> None:
        self.pred_dir = Path(pred_dir)
        self.command = " ".join(str(x) for x in command) if isinstance(command, (list, tuple)) else (command or "")
        self.path = self.pred_dir / self.LOCK_NAME
        self._fd: int | None = None

    def _read_holder(self) -> LockInfo | None:
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
            return LockInfo(**data)
        except (OSError, ValueError, TypeError, json.JSONDecodeError):
            return None

    def acquire(self) -> "RunLock":
        import fcntl  # POSIX only

        self.pred_dir.mkdir(parents=True, exist_ok=True)
        fd = os.open(str(self.path), os.O_CREAT | os.O_RDWR, 0o644)
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as exc:
            os.close(fd)
            holder = self._read_holder()
            detail = f" (held by pid={holder.pid} host={holder.host} started={holder.started_iso})" if holder else ""
            raise RunLockHeld(
                f"prediction directory '{self.pred_dir}' is locked by another "
                f"live writer{detail}; refusing to write concurrently. "
                f"If the holder is dead, remove '{self.path}'."
            ) from exc
        self._fd = fd
        info = LockInfo(pid=os.getpid(), host=_hostname(), started_iso=iso_utc(), command=self.command)
        os.write(fd, json.dumps(asdict(info), ensure_ascii=False).encode("utf-8"))
        os.fsync(fd)
        return self

    def release(self) -> None:
        if self._fd is None:
            return
        import fcntl

        try:
            fcntl.flock(self._fd, fcntl.LOCK_UN)
        finally:
            os.close(self._fd)
            self._fd = None
            try:
                self.path.unlink()
            except FileNotFoundError:
                pass

    def __enter__(self) -> "RunLock":
        return self.acquire()

    def __exit__(self, *exc) -> None:
        self.release()


def _hostname() -> str:
    import socket

    try:
        return socket.gethostname() or "unknown"
    except OSError:
        return "unknown"


def acquire_run_lock(pred_dir, command=None) -> RunLock:
    """Context manager: acquire the exclusive writer lock for ``pred_dir``.

    Use as ``with acquire_run_lock(pred_dir, command=[...]): ...``. Raises
    :class:`RunLockHeld` immediately if another live writer holds it.
    """
    return RunLock(pred_dir, command=command)
