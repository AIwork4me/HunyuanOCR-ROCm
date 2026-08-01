# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 AIwork4me
"""ADR-0003: the gfx1100 ROCm floor guard (the pixel-area cap is removed; an env
below the proven #6416-fix floor must fail fast). Tests the pure version logic so
they run in CI without a ROCm torch install."""

from __future__ import annotations

from hunyuan_ocr.backends.transformers import _below_gfx1100_floor


def test_below_floor_old_default_env():
    # the pre-migration default env (torch 2.9.1 + hip 7.2) is below the floor
    assert _below_gfx1100_floor("2.9.1+rocm7.2.1.gitff65f5bc", "7.2.53211-e1a6bc5663")


def test_below_floor_ga_714_passes():
    # the proven-fix combination (GA 7.14) is at/above the floor
    assert not _below_gfx1100_floor("2.11.0+rocm7.14.0", "7.14.60850")


def test_below_floor_torch_too_old():
    assert _below_gfx1100_floor("2.10.0+rocm7.14.0", "7.14.60850")


def test_below_floor_hip_too_old():
    assert _below_gfx1100_floor("2.11.0+rocm7.13.0", "7.13.1")


def test_below_floor_non_rocm_is_noop():
    # non-ROCm builds (hip is None) are not subject to #6416 -> never below floor
    assert not _below_gfx1100_floor("2.11.0", None)
    assert not _below_gfx1100_floor("2.9.1", None)
