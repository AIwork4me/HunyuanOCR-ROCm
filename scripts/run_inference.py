#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 AIwork4me
"""OpenAI-compatible inference driver: run HunyuanOCR-1.5 over OmniDocBench.

Thin wrapper over :mod:`hunyuan_ocr.driver` (the package-resident, wheel-runnable
orchestration). Supports the OpenAI-compatible backends ``llamacpp`` (llama.cpp
``llama-server``), ``vllm``, and any other ``openai`` server via ``--backend-name``.

The real callables — the OpenAI client factory, the per-image ``infer_one``, and
the ``/v1/models`` health check — are imported here at module scope so tests can
monkeypatch them on this module (``drv.infer_one`` / ``drv.OpenAI`` /
``drv.health_check``); ``main_with_args`` resolves them as late-bound globals, so
the patches take effect.

``scripts/run_phase2_vllm.py`` is a backward-compatibility shim that delegates
here, so existing ``python scripts/run_phase2_vllm.py ...`` commands keep working.

Usage:
  # start servers first (one/GPU), then:
  python scripts/run_inference.py --backend-name llamacpp \\
      --gt-json GT.json --images-dir images --pred-dir ./predictions \\
      --ports 8081,8082,8083,8084 --model HYVL --concurrency 16
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from openai import OpenAI  # noqa: E402

from hunyuan_ocr import driver  # noqa: E402
from hunyuan_ocr.backends.vllm_client import infer_one  # noqa: E402

# Re-exported so tests/users that patch these on this module keep working.
health_check = driver.health_check
SUPPORTED_BACKENDS = driver.SUPPORTED_BACKENDS
_provenance = driver._provenance
_endpoints = driver._endpoints


def main_with_args(argv) -> int:
    """Parse argv, dispatch via the package driver, and ``sys.exit`` on failure.

    On a clean run returns ``0`` (no SystemExit); on any non-zero outcome it
    raises ``SystemExit(code)`` so shell/CI callers see the failure.
    """
    args = driver.parse_args(argv)
    code = driver.dispatch(args, infer_one=infer_one, client_factory=OpenAI, health_check_fn=health_check)
    if code:
        sys.exit(code)
    return 0


def main():
    main_with_args(sys.argv[1:])


if __name__ == "__main__":
    main()
