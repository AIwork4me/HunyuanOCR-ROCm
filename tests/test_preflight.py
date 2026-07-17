# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 AIwork4me
"""CPU unit tests for pre-flight input validation + sharding (Phase 3.2)."""

from __future__ import annotations

import json

import pytest

from hunyuan_ocr import preflight


def _write_gt(tmp_path, stems):
    gt = tmp_path / "gt.json"
    gt.write_text(json.dumps([{"page_info": {"image_path": f"{s}.png"}} for s in stems]), encoding="utf-8")
    img = tmp_path / "images"
    img.mkdir()
    for s in stems:
        (img / f"{s}.png").write_bytes(b"x")
    return gt, img


def test_shard_more_gpus_than_pages_no_indexerror():
    # The pre-fix bug: 1 page, 4 GPUs -> chunks[1..3] IndexError. Now 4 buckets.
    chunks = preflight.shard(["only"], 4)
    assert len(chunks) == 4
    assert chunks[0] == ["only"]
    assert chunks[1] == [] and chunks[2] == [] and chunks[3] == []


def test_shard_balanced():
    chunks = preflight.shard(list(range(7)), 3)
    assert len(chunks) == 3
    assert sum(len(c) for c in chunks) == 7


def test_shard_zero_raises():
    with pytest.raises(ValueError):
        preflight.shard(["a"], 0)


def test_empty_gt_rejected(tmp_path):
    gt = tmp_path / "gt.json"
    gt.write_text("[]", encoding="utf-8")
    with pytest.raises(preflight.PreflightError):
        preflight.load_gt(gt)


def test_missing_images_rejected(tmp_path):
    gt = tmp_path / "gt.json"
    gt.write_text(json.dumps([{"page_info": {"image_path": "ghost.png"}}]), encoding="utf-8")
    (tmp_path / "images").mkdir()
    with pytest.raises(preflight.PreflightError):
        preflight.pages_with_images(gt, tmp_path / "images")


def test_bad_ports_and_ranges(tmp_path):
    gt, img = _write_gt(tmp_path, ["a"])
    problems = preflight.check_prediction_inputs(
        gt_json=gt,
        images_dir=img,
        ports="",
        gpu_ids=None,
        concurrency=0,
        max_retries=0,
        retry_backoff=-1,
        max_pixels=-5,
        model="",
        pred_dir=tmp_path / "pred",
    )
    fields = {f for f, _ in problems}
    assert {"ports", "concurrency", "max-retries", "retry-backoff", "max-pixels", "model"} <= fields


def test_duplicate_ports_rejected(tmp_path):
    gt, img = _write_gt(tmp_path, ["a"])
    problems = preflight.check_prediction_inputs(
        gt_json=gt,
        images_dir=img,
        ports="8000,8000",
        gpu_ids=None,
        concurrency=1,
        max_retries=1,
        retry_backoff=0,
        max_pixels=0,
        model="HYVL",
        pred_dir=tmp_path / "pred",
    )
    assert any("ports" in f and "duplicate" in m for f, m in problems)


def test_unknown_backend_rejected(tmp_path):
    gt, img = _write_gt(tmp_path, ["a"])
    problems = preflight.check_prediction_inputs(
        gt_json=gt,
        images_dir=img,
        ports="8000",
        gpu_ids=None,
        concurrency=1,
        max_retries=1,
        retry_backoff=0,
        max_pixels=0,
        model="HYVL",
        pred_dir=tmp_path / "pred",
        backend_name="triton",
        allowed_backends={"vllm", "llamacpp"},
    )
    assert any(f == "backend-name" for f, _ in problems)


def test_valid_inputs_pass(tmp_path):
    gt, img = _write_gt(tmp_path, ["a", "b"])
    problems = preflight.check_prediction_inputs(
        gt_json=gt,
        images_dir=img,
        ports="8000,8001",
        gpu_ids=None,
        concurrency=4,
        max_retries=2,
        retry_backoff=1.0,
        max_pixels=0,
        model="HYVL",
        pred_dir=tmp_path / "pred",
    )
    assert problems == []
