# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 AIwork4me
"""Pre-flight input validation + sharding helpers shared by the drivers.

Fails fast, BEFORE any model load or server request, on bad arguments or missing
inputs: empty/invalid GT, missing images, empty/duplicate ports or GPU ids,
out-of-range numerics, unwritable output dir, unknown backend. Also fixes the
shard bug where GPUs > pages caused an IndexError (``shard`` now always returns
exactly ``n`` buckets, some possibly empty).

No GPU, no model deps. Pure stdlib.
"""

from __future__ import annotations

import json
from pathlib import Path


class PreflightError(ValueError):
    """Raised when pre-flight checks fail. ``errors`` lists every problem."""

    def __init__(self, errors):
        if isinstance(errors, str):
            errors = [errors]
        self.errors = list(errors)
        super().__init__("; ".join(self.errors))


def load_gt(gt_json) -> list[dict]:
    """Load + structurally validate an OmniDocBench GT json. Raises PreflightError."""
    path = Path(gt_json)
    if not path.is_file():
        raise PreflightError([f"GT json not found: {gt_json}"])
    try:
        pages = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise PreflightError([f"GT json not parseable: {exc}"]) from exc
    if not isinstance(pages, list):
        raise PreflightError([f"GT json top-level must be a list, got {type(pages).__name__}"])
    if not pages:
        raise PreflightError(["GT json is an empty list (no pages)"])
    problems = []
    for i, p in enumerate(pages):
        if not isinstance(p, dict) or "page_info" not in p:
            problems.append(f"GT page[{i}] missing 'page_info'")
            continue
        if "image_path" not in (p["page_info"] or {}):
            problems.append(f"GT page[{i}].page_info missing 'image_path'")
    if problems:
        raise PreflightError(problems)
    return pages


def pages_with_images(gt_json, images_dir) -> list[tuple[str, str]]:
    """Return [(stem, abs_image_path), ...]; verify every image exists."""
    pages = load_gt(gt_json)
    images_dir = Path(images_dir)
    if not images_dir.is_dir():
        raise PreflightError([f"images-dir not found: {images_dir}"])
    out: list[tuple[str, str]] = []
    missing: list[str] = []
    for p in pages:
        rel = p["page_info"]["image_path"]
        abs_path = images_dir / rel
        if not abs_path.is_file():
            missing.append(rel)
        out.append((Path(rel).stem, str(abs_path)))
    if missing:
        sample = ", ".join(missing[:5])
        raise PreflightError([f"{len(missing)} input image(s) missing under {images_dir}; first: {sample}"])
    return out


def shard(items: list, n: int) -> list[list]:
    """Split ``items`` into exactly ``n`` buckets (stride distribution).

    Always returns ``n`` buckets — some possibly empty — so a worker assigned 0
    pages loops zero times instead of raising IndexError. Handles ``len(items)
    < n`` (more GPUs than pages) and ``n == 0``/negative (raises).
    """
    if n <= 0:
        raise ValueError(f"shard count must be >= 1, got {n}")
    return [list(items[i::n]) for i in range(n)]


def _split_ids(s, *, name) -> list[int]:
    if s is None or str(s).strip() == "":
        raise PreflightError([f"{name} is empty"])
    try:
        ids = [int(x) for x in str(s).split(",") if x.strip() != ""]
    except ValueError:
        raise PreflightError([f"{name} has non-integer entries: {s!r}"])
    if not ids:
        raise PreflightError([f"{name} is empty"])
    if len(set(ids)) != len(ids):
        raise PreflightError([f"{name} has duplicates: {ids}"])
    if any(i < 0 for i in ids):
        raise PreflightError([f"{name} has negative entries: {ids}"])
    return ids


def check_prediction_inputs(
    *,
    gt_json,
    images_dir,
    ports,
    gpu_ids,
    concurrency,
    max_retries,
    retry_backoff,
    max_pixels,
    model,
    pred_dir,
    backend_name=None,
    allowed_backends=None,
) -> list[tuple[str, str]]:
    """Return a list of (field, message) problems for an OpenAI-compatible run.

    Empty list == ok. Does NOT load the GT here (the driver does, to get pages);
    this validates argument ranges and the output dir writability.
    """
    problems: list[tuple[str, str]] = []

    # ports (required for OpenAI-compatible server backends; the in-process
    # transformers backend has no server, so ports are not required there)
    if backend_name != "transformers":
        try:
            _split_ids(ports, name="ports")
        except PreflightError as exc:
            problems.extend(("ports", e) for e in exc.errors)

    # gpu_ids (optional for the OAI driver, required for transformers)
    if gpu_ids is not None:
        try:
            _split_ids(gpu_ids, name="gpu-ids")
        except PreflightError as exc:
            problems.extend(("gpu-ids", e) for e in exc.errors)

    for field_name, val, lo in (
        ("concurrency", concurrency, 1),
        ("max-retries", max_retries, 1),
        ("max-pixels", max_pixels, 0),
    ):
        if not isinstance(val, int) or isinstance(val, bool) or val < lo:
            problems.append((field_name, f"must be an int >= {lo}, got {val!r}"))
    if not isinstance(retry_backoff, (int, float)) or isinstance(retry_backoff, bool) or retry_backoff < 0:
        problems.append(("retry-backoff", f"must be a number >= 0, got {retry_backoff!r}"))
    if not model or not str(model).strip():
        problems.append(("model", "must be non-empty"))

    if backend_name and allowed_backends and backend_name not in allowed_backends:
        problems.append(("backend-name", f"{backend_name!r} not in {sorted(allowed_backends)}"))

    # output dir writability (create if absent; check we can write a probe)
    pred = Path(pred_dir)
    try:
        pred.mkdir(parents=True, exist_ok=True)
        probe = pred / ".write_probe"
        probe.write_text("ok", encoding="utf-8")
        probe.unlink()
    except OSError as exc:
        problems.append(("pred-dir", f"not writable: {exc}"))
    return problems


def assert_ok(problems: list[tuple[str, str]]) -> None:
    """Raise PreflightError listing every problem, or return if there are none."""
    if problems:
        lines = [f"[{fld}] {msg}" for fld, msg in problems]
        raise PreflightError(lines)
