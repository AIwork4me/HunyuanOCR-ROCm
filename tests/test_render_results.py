# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 AIwork4me
"""CPU tests for scripts/render_benchmark_tables.py (no network)."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]

_spec = importlib.util.spec_from_file_location(
    "render_benchmark_tables", REPO / "scripts" / "render_benchmark_tables.py"
)
rr = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(rr)
sys.modules["render_benchmark_tables"] = rr


FIXTURE_LOCK = {
    "benchmark": {
        "canary_148": {"vllm_overall": 94.81, "transformers_overall": 94.11, "llamacpp_overall": 93.33},
        "full_1651": {"llamacpp_overall": 92.09, "vllm_overall": "invalid"},
        "official_reference": {"omnidocbench_overall": "94.10", "inference_engine": "TensorRT"},
    }
}


def test_render_block_contains_all_lock_rows_in_order():
    block = rr.render_block(FIXTURE_LOCK)
    # canary backends sorted alphabetically by display name
    assert "| canary 148 | llama.cpp | 93.33 |" in block
    assert "| canary 148 | transformers | 94.11 |" in block
    assert "| canary 148 | vLLM | 94.81 |" in block
    assert "| full 1651 | llama.cpp | 92.09 |" in block
    assert "| full 1651 | vLLM | invalid (excluded; see REPRO.yaml) |" in block
    assert "| official | TensorRT | 94.10 | official HunyuanOCR table |" in block
    # the trailing-zero official figure is preserved (not 94.1)
    assert "94.1 |" not in block and "94.10 |" in block


def test_check_detects_drift(tmp_path):
    lock = tmp_path / "lock.yaml"
    lock.write_text("benchmark:\n  canary_148:\n    vllm_overall: 99.99\n", encoding="utf-8")
    readme = tmp_path / "README.md"
    readme.write_text(
        "## Results\n\n"
        "<!-- BEGIN GENERATED RESULTS -->\n"
        "| Page set | Backend | Overall | Source |\n|---|---|---|---|\n"
        "| canary 148 | vLLM | 94.81 | old |\n"
        "<!-- END GENERATED RESULTS -->\n",
        encoding="utf-8",
    )
    rc = rr.main(["--check", "--lock", str(lock), "--readme", str(readme)])
    assert rc == 1  # drift detected


def test_check_passes_when_in_sync(tmp_path):
    lock = tmp_path / "lock.yaml"
    lock.write_text("benchmark:\n  canary_148:\n    vllm_overall: 94.81\n", encoding="utf-8")
    readme = tmp_path / "README.md"
    readme.write_text(
        "## Results\n\n<!-- BEGIN GENERATED RESULTS -->\nOLD\n<!-- END GENERATED RESULTS -->\n### Table A\n",
        encoding="utf-8",
    )
    rr.main(["--lock", str(lock), "--readme", str(readme)])  # replace region with canonical
    assert rr.main(["--check", "--lock", str(lock), "--readme", str(readme)]) == 0
