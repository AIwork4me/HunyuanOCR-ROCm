#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 AIwork4me
"""Backward-compatibility shim — delegates to ``scripts/run_inference.py``.

``run_inference.py`` is the canonical name for the OpenAI-compatible inference
driver (llamacpp / vllm / openai). This file keeps the historical
``run_phase2_vllm`` name so existing ``python scripts/run_phase2_vllm.py ...``
commands keep working. New code and docs use ``run_inference.py``.
"""

from __future__ import annotations

from run_inference import main

if __name__ == "__main__":
    main()
