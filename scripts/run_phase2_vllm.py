#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 AIwork4me
"""OpenAI-compatible driver: run HunyuanOCR-1.5 over OmniDocBench via a server.

Thin wrapper over :mod:`hunyuan_ocr.driver` (the package-resident, wheel-runnable
orchestration). The real callables — the OpenAI client factory, the per-image
``infer_one``, and the ``/v1/models`` health check — are imported here at module
scope so existing tests and users can monkeypatch them on this module
(``drv.infer_one`` / ``drv.OpenAI`` / ``drv.health_check``); ``main_with_args``
resolves them as late-bound globals, so the patches take effect.

``run_openai_compatible.py`` is the same program under its generic name; this
file keeps the historical ``run_phase2_vllm`` name for existing users/Makefile.

Usage:
  # start servers first (one/GPU), then:
  python scripts/run_phase2_vllm.py --backend-name llamacpp \\
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

# Re-exported so tests/users that patch these on this module keep working, and so
# `import run_phase2_vllm as drv; drv.health_check` still resolves.
health_check = driver.health_check
SUPPORTED_BACKENDS = driver.SUPPORTED_BACKENDS
_provenance = driver._provenance
_endpoints = driver._endpoints


def main_with_args(argv) -> int:
    """Parse argv, dispatch via the package driver, and ``sys.exit`` on failure.

    On a clean run returns ``0`` (no SystemExit, matching the historical
    contract that a successful invocation simply returns); on any non-zero
    outcome it raises ``SystemExit(code)`` so shell/CI callers see the failure.
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
