#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 AIwork4me
"""Canonical entry for the OpenAI-compatible multi-server driver.

Thin alias over ``run_phase2_vllm.py`` (same implementation) under the generic
name, so users are not led to think it only works with vLLM. Supports
``--backend-name {vllm,llamacpp,openai}``.

Usage:
  python scripts/run_openai_compatible.py --backend-name llamacpp ...
"""

from __future__ import annotations
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_phase2_vllm as drv  # noqa: E402


def main():
    drv.main_with_args(sys.argv[1:])


if __name__ == "__main__":
    main()
