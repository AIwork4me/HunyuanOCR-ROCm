# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 AIwork4me
"""OmniDocBench v1.6 scoring driver.

Writes an eval config, invokes the OmniDocBench scorer (pdf_validation.py) in its
own 3.11 venv, and parses the resulting metric_result.json / run_summary.json.
Overall = ((1 - text_EditDist)*100 + formula_CDM*100 + table_TEDS*100) / 3
(reading-order EditDist is reported separately, NOT part of Overall).
"""

from __future__ import annotations

import json
import os
import subprocess
import tempfile
from pathlib import Path

import yaml

# Machine-local defaults from the benchmark environment. Override per-host via the
# OMNIDOCBENCH_VENV / OMNIDOCBENCH_REPO env vars (or --venv-python /
# --omnidocbench-repo) so `hunyuan-ocr score` works from a wheel install anywhere.
DEFAULT_VENV_PYTHON = os.environ.get("OMNIDOCBENCH_VENV", "/root/ocr-eval/OmniDocBench/.venv/bin/python")
DEFAULT_OMNIDOCBENCH_REPO = os.environ.get("OMNIDOCBENCH_REPO", "/root/ocr-eval/OmniDocBench")


class ScoringError(RuntimeError):
    """Raised when a prediction directory cannot be scored (validation failure or
    scorer non-zero exit). Carries a structured message so the CLI can present a
    friendly error instead of a traceback."""


# The template is bundled inside the package (data/eval_config.yaml) so it ships
# in the wheel; the repo's eval/configs/ copy is kept for human reference.
_REPO_TEMPLATE = Path(__file__).resolve().parents[2] / "eval" / "configs" / "hunyuanocr-1.5_linux-rocm.yaml"


def _load_eval_template() -> str:
    """Return the OmniDocBench eval-config template text.

    Prefers the resource bundled in the installed package (works under a wheel
    install); falls back to the repo-relative file for editable/src layouts.
    """
    try:
        from importlib.resources import files

        return (files("hunyuan_ocr") / "data" / "eval_config.yaml").read_text(encoding="utf-8")
    except Exception:
        return _REPO_TEMPLATE.read_text(encoding="utf-8")


def overall_score(metrics: dict) -> float | None:
    """v1.6 Overall = ((1-text_edit)*100 + cdm*100 + teds*100)/3.

    Returns None when any of the three is missing (e.g. CDM is null on a subset
    with no display-formula pages), since the 3-metric Overall is undefined then.
    """
    text = metrics["text_edit_dist"]
    cdm = metrics["formula_cdm"]
    teds = metrics["table_teds"]
    if text is None or cdm is None or teds is None:
        return None
    return ((1.0 - text) * 100.0 + cdm * 100.0 + teds * 100.0) / 3.0


def write_eval_config(*, gt_json: str, pred_dir: str, out_yaml: Path) -> None:
    """Materialize an eval config from the template, substituting GT + pred paths."""
    cfg = yaml.safe_load(_load_eval_template())
    cfg["end2end_eval"]["dataset"]["ground_truth"]["data_path"] = str(gt_json)
    cfg["end2end_eval"]["dataset"]["prediction"]["data_path"] = str(pred_dir)
    out_yaml = Path(out_yaml)
    out_yaml.parent.mkdir(parents=True, exist_ok=True)
    out_yaml.write_text(yaml.safe_dump(cfg, allow_unicode=True, sort_keys=False), encoding="utf-8")


def run_scorer(
    *, omnidocbench_repo: str, config_yaml: str, venv_python: str | None = None
) -> subprocess.CompletedProcess:
    """Run pdf_validation.py --config <cfg> inside the OmniDocBench repo."""
    py = venv_python or DEFAULT_VENV_PYTHON
    cmd = [py, "pdf_validation.py", "--config", str(config_yaml)]
    return subprocess.run(cmd, cwd=omnidocbench_repo, capture_output=True, text=True, check=False)


def parse_run_summary(result_dir: str | Path, save_name: str) -> dict:
    """Read per-task numbers from OmniDocBench's ``run_summary.json``
    (``notebook_metric_summary.metrics`` is the notebook-aligned source of truth).
    ``save_name`` = ``basename(pred_dir) + '_quick_match'``. ``formula_cdm`` is
    ``None`` when the subset has no display-formula pages (CDM did not run)."""
    result_dir = Path(result_dir)
    summary = json.loads((result_dir / f"{save_name}_run_summary.json").read_text(encoding="utf-8"))
    ms = summary["notebook_metric_summary"]["metrics"]

    def raw(key: str) -> float | None:
        return ms.get(key, {}).get("raw")

    text = raw("text_block_Edit_dist")
    cdm = raw("display_formula_CDM")  # None when no formula pages
    teds = raw("table_TEDS")
    order = raw("reading_order_Edit_dist")
    return {
        "overall": overall_score({"text_edit_dist": text, "formula_cdm": cdm, "table_teds": teds}),
        "text_edit_dist": text,
        "formula_cdm": cdm,
        "table_teds": teds,
        "reading_order_edit": order,
    }


def score_directory(
    *,
    gt_json: str,
    pred_dir: str,
    omnidocbench_repo: str | None = None,
    venv_python: str | None = None,
    skip_validation: bool = False,
    strict: bool = True,
) -> dict:
    """Validate (unless skipped) and score a prediction directory end-to-end.

    The OmniDocBench eval config is written into a PRIVATE temp dir (never the
    prediction dir) so repeated scoring passes the strict validator. Runs the
    scorer in the pinned OmniDocBench venv and parses ``run_summary.json``.

    Returns ``{"validation_report": Report|None, "metrics": {...}}``. Raises
    :class:`ScoringError` on validation failure or a non-zero scorer exit — the
    message is user-facing (no raw traceback). Shared by the CLI ``score``
    command and ``scripts/score_predictions.py``.
    """
    from hunyuan_ocr.validation import validate_predictions

    repo = omnidocbench_repo or DEFAULT_OMNIDOCBENCH_REPO
    venv = venv_python or DEFAULT_VENV_PYTHON
    report = None
    if not skip_validation:
        report = validate_predictions(gt_json, pred_dir, strict=strict)
        ok = report.ok_strict if strict else report.ok
        if not ok:
            lines = [
                f"  [{'ERROR' if p.severity == 'error' else 'WARN'}] {p.code}: {p.message}" for p in report.problems
            ]
            raise ScoringError(
                f"predictions invalid ({len(report.errors())} error(s), "
                f"{len(report.warnings())} warning(s)); refusing to score:\n" + "\n".join(lines)
            )
    save_name = f"{Path(pred_dir).name}_quick_match"
    with tempfile.TemporaryDirectory(prefix="hunyuanocr_eval_") as tmpd:
        cfg_path = Path(tmpd) / "_eval_config.yaml"
        write_eval_config(gt_json=gt_json, pred_dir=pred_dir, out_yaml=cfg_path)
        res = run_scorer(omnidocbench_repo=repo, config_yaml=str(cfg_path), venv_python=venv)
        if res.returncode != 0:
            tail = (res.stderr or "")[-2000:]
            raise ScoringError(f"scorer failed (rc={res.returncode})\n{tail}")
        metrics = parse_run_summary(Path(repo) / "result", save_name)
    return {"validation_report": report, "metrics": metrics}


def format_score_table(label: str, metrics: dict) -> str:
    """Render the per-task metrics as the human-readable score table."""

    def fmt(v, pct=False):
        if v is None:
            return "n/a"
        return f"{v * 100:.2f}" if pct else f"{v:.4f}"

    ov = metrics["overall"]
    recomputed = overall_score(
        {
            "text_edit_dist": metrics["text_edit_dist"],
            "formula_cdm": metrics["formula_cdm"],
            "table_teds": metrics["table_teds"],
        }
    )
    return (
        f"\n=== {label} -- OmniDocBench v1.6 ===\n"
        f"  Overall          : {'n/a (CDM missing on this subset)' if ov is None else f'{ov:.2f}'}\n"
        f"  text  EditDist   : {fmt(metrics['text_edit_dist'])}   -> {fmt(metrics['text_edit_dist'], pct=True)}\n"
        f"  formula CDM      : {fmt(metrics['formula_cdm'])}   -> {fmt(metrics['formula_cdm'], pct=True)}\n"
        f"  table  TEDS      : {fmt(metrics['table_teds'])}   -> {fmt(metrics['table_teds'], pct=True)}\n"
        f"  order  EditDist  : {fmt(metrics['reading_order_edit'])}\n"
        f"  (overall recomputed: {'n/a' if recomputed is None else f'{recomputed:.2f}'})\n"
    )
