# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 AIwork4me
"""Render the verified benchmark results from ``REPRO.yaml``.

Single source of truth for the headline numbers: the lock. Both the README
generator (``scripts/render_benchmark_tables.py``) and the ``hunyuan-ocr
benchmark`` CLI render through this module, so they can never disagree. No number
is invented — only values present in the lock are emitted.
"""

from __future__ import annotations

BACKEND_DISPLAY = {"llamacpp": "llama.cpp", "transformers": "transformers", "vllm": "vLLM"}


def _fmt_value(v) -> str:
    if isinstance(v, str) and v.strip().lower() == "invalid":
        return "invalid (excluded; see REPRO.yaml)"
    return str(v)


def render_results_block(lock: dict) -> str:
    """Render the BEGIN..END GENERATED RESULTS block (markdown) from the lock."""
    bench = (lock or {}).get("benchmark", {}) or {}
    lines = [
        "<!-- BEGIN GENERATED RESULTS -->",
        "<!-- Auto-generated from REPRO.yaml by scripts/render_benchmark_tables.py (do not edit by hand). -->",
        "",
        "| Page set | Backend | Overall | Source |",
        "|---|---|---|---|",
    ]
    for page_key, label in (("canary_148", "canary 148"), ("full_1651", "full 1651")):
        section = bench.get(page_key, {}) or {}
        rows = []
        for k, v in section.items():
            if not k.endswith("_overall"):
                continue
            backend = k[: -len("_overall")]
            rows.append((BACKEND_DISPLAY.get(backend, backend), _fmt_value(v)))
        for display, value in sorted(rows):
            lines.append(f"| {label} | {display} | {value} | REPRO.yaml |")
    official = bench.get("official_reference", {}) or {}
    if official:
        engine = official.get("inference_engine", "official")
        overall = _fmt_value(official.get("omnidocbench_overall"))
        lines.append(f"| official | {engine} | {overall} | official HunyuanOCR table |")
    lines.append("")
    lines.append("<!-- END GENERATED RESULTS -->")
    return "\n".join(lines)
