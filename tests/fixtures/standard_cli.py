#!/usr/bin/env python
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 AIwork4me
"""Thin entry shim so the central conformance runner can invoke the REAL
``hunyuan-ocr`` standard CLI as a subprocess:

    omnidocbench-rocm conformance-profiles base        --cli tests/fixtures/standard_cli.py
    omnidocbench-rocm conformance-profiles runtime-core --cli tests/fixtures/standard_cli.py

The real console script is ``hunyuan-ocr`` (``hunyuan_ocr.cli:main``); this file
only exists because the runner executes ``python <cli_path> <args>``. It adds NO
behavior of its own. Run with ``PYTHONPATH=src`` (or a wheel install).
"""

import sys

from hunyuan_ocr.cli import main

if __name__ == "__main__":
    sys.exit(main())
