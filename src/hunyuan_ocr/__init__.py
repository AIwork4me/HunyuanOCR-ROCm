# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 AIwork4me
"""HunyuanOCR-ROCm: evaluation-backed AMD ROCm port of HunyuanOCR-1.5.

Runs the model across three OpenAI/transformers backends on AMD gfx1100 and
scores them on OmniDocBench v1.6. Not a precision-aligned port — see
docs/benchmark-methodology.md.
"""

__version__ = "0.1.0"
