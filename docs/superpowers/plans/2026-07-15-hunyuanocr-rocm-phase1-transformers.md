# HunyuanOCR-ROCm Phase 1 (transformers oracle) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Stand up the standalone `HunyuanOCR-ROCm` repo and run HunyuanOCR-1.5 via the **transformers** backend to produce the on-machine **OmniDocBench v1.6 BASELINE** (the absolute precision ground-truth for Phases 2–3) plus a frozen canary subset.

**Architecture:** A self-contained Python package `hunyuan_ocr` holds a frozen **decoding contract** + verbatim-ported upstream post-processors + an OmniDocBench eval driver. The transformers backend ports the upstream `inference/transformers/infer_hf_8gpu_hyocr15.py` worker (multi-GPU `multiprocessing.spawn`, greedy decode, `clean_repeated_substrings` + `process_one`). Predictions are one `<stem>.md` per page; OmniDocBench's `pdf_validation.py` (its own 3.11 venv) scores them. No platform adapter / `omnidocbench-amd` coupling (that is Phase 4).

**Tech Stack:** Python 3.12, `transformers==5.13.0`, PyTorch ROCm (gfx1100), Pillow, tqdm, PyYAML; OmniDocBench scorer in its own Python 3.11 venv at `/root/ocr-eval/OmniDocBench/.venv`.

## Global Constraints

(From the approved design spec `docs/superpowers/specs/2026-07-15-hunyuanocr-rocm-design.md`. Every task inherits these.)

- **Platform:** `linux-rocm` / gfx1100 only. Storage plan: code in `/workspace`, weights/datasets/venvs in `/root`.
- **Model:** `tencent/HunyuanOCR` (root = 1.5), loaded as `HunYuanVLForConditionalGeneration` + `AutoProcessor`, `dtype=bfloat16`, `attn_implementation=eager`.
- **Frozen sampling:** `do_sample=False` (greedy, temp=0), `repetition_penalty=1.08`, `max_new_tokens=32768`, `use_cache=True`, tail-repeat early-stop `min_repeats=8`, `skip_special_tokens=True`, `clean_up_tokenization_spaces=False`. **Never change these without re-baselining.**
- **Frozen `doc_parse` prompt:** `提取文档图片中正文的所有信息用markdown格式表示，其中页眉、页脚部分忽略，表格用html格式表达，文档中公式用latex格式表示，按照阅读顺序组织进行解析。`
- **Frozen post-processors** (doc_parse only): `clean_repeated_substrings` then `process_one` — ported **verbatim** from upstream `inference/vllm_0_18_1/hunyuan_utils.py` (= `inference/transformers/hunyuan_utils.py`).
- **Predictions layout:** one UTF-8 `<image_stem>.md` per page in a predictions dir.
- **OmniDocBench v1.6 Overall** = `((1 - text_EditDist) * 100 + formula_CDM * 100 + table_TEDS * 100) / 3` (reading-order reported separately, NOT in Overall).
- **Precision gate (Phases 2–3):** within ±0.3 overall / ±0.5 per-task of this Phase-1 BASELINE. Upstream 94.74 is a sanity reference only.
- **venvs:** model-venv-transformers (Python 3.12, `transformers==5.13.0`); OmniDocBench scorer uses `/root/ocr-eval/OmniDocBench/.venv` (Python 3.11). Never install both stacks in one venv.
- **DRY / YAGNI / TDD / frequent commits.** No platform `adapter/` contract yet (Phase 4).
- **Upstream source of truth (cloned):** `/root/HunyuanOCR-src/` (shallow clone of `github.com/Tencent-Hunyuan/HunyuanOCR`). Key files: `inference/transformers/{infer_hf_8gpu_hyocr15.py,hunyuan_utils.py}`, `inference/vllm_0_18_1/{hunyuan_tasks.py,hunyuan_utils.py}`.

---

## File Structure

| File | Responsibility |
|---|---|
| `pyproject.toml` | Package metadata + deps; `hunyuan_ocr` as `src/` layout. |
| `src/hunyuan_ocr/__init__.py` | Package root; exports `__version__`. |
| `src/hunyuan_ocr/contract.py` | `CONTRACT` — the frozen decoding config (single source of truth). |
| `src/hunyuan_ocr/tasks.py` | `TASK_PROMPTS` + `get_prompt()` (verbatim port). |
| `src/hunyuan_ocr/postprocess.py` | `has_tail_repetition`, `clean_repeated_substrings`, `process_one` + 10 patterns (verbatim port). |
| `src/hunyuan_ocr/omnidocbench.py` | `iter_page_images(gt_json, images_dir)`, `derive_prediction_filename(image_path)`. |
| `src/hunyuan_ocr/scoring.py` | `write_eval_config(...)`, `run_scorer(...)`, `parse_run_summary(...)`, `overall_score(...)`. |
| `src/hunyuan_ocr/backends/__init__.py` | Backend package. |
| `src/hunyuan_ocr/backends/transformers.py` | `load_model_and_processor(model_path)`, `infer_one(model, processor, image_path, prompt) -> str`. |
| `scripts/run_phase1_transformers.py` | Multi-GPU sharded driver: pages → `<stem>.md`. Resumable. |
| `scripts/regression_canary.py` | Run transformers on the 150-page canary subset → canary score. |
| `scripts/score_predictions.py` | Predictions dir → OmniDocBench score table (overall + per-task). |
| `eval/configs/hunyuanocr-1.5_linux-rocm.yaml` | Eval config template (ground-truth / prediction / metrics). |
| `tests/test_*.py` | Unit + characterization tests; `tests/fixtures/` for tiny data. |
| `Makefile` | `demo`, `eval-canary`, `eval-full`, `score`, `oracle-check` targets (stubbed/real). |
| `reports/phase1-transformers.md` | Phase-1 report + BASELINE score table (filled at Task 9). |

---

## Task 1: Package scaffolding

**Files:**
- Create: `pyproject.toml`, `src/hunyuan_ocr/__init__.py`, `src/hunyuan_ocr/backends/__init__.py`, `tests/__init__.py`, `tests/test_smoke.py`, `Makefile`, `eval/configs/.gitkeep`

**Interfaces:**
- Produces: importable package `hunyuan_ocr` with `__version__`.

- [ ] **Step 1: Write `pyproject.toml`**

```toml
[build-system]
requires = ["setuptools>=68"]
build-backend = "setuptools.build_meta"

[project]
name = "hunyuan-ocr"
version = "0.1.0"
description = "Precision-aligned AMD ROCm port of HunyuanOCR-1.5 for OmniDocBench v1.6"
requires-python = ">=3.12"
dependencies = [
  "transformers==5.13.0",
  "torch",          # installed from ROCm wheel, not PyPI — do NOT pin here
  "pillow",
  "tqdm",
  "pyyaml",
  "requests",
]

[project.optional-dependencies]
dev = ["pytest>=8", "pytest-mock"]

[tool.setuptools.packages.find]
where = ["src"]
```

- [ ] **Step 2: Write `src/hunyuan_ocr/__init__.py`**

```python
"""HunyuanOCR-ROCm: precision-aligned AMD ROCm port of HunyuanOCR-1.5."""

__version__ = "0.1.0"
```

- [ ] **Step 3: Write `src/hunyuan_ocr/backends/__init__.py`**

```python
"""Inference backends. Phase 1 ships `transformers`; vLLM/llama.cpp come later."""
```

- [ ] **Step 4: Write the failing test `tests/test_smoke.py`**

```python
def test_package_imports_and_version():
    import hunyuan_ocr

    assert hunyuan_ocr.__version__ == "0.1.0"
```

- [ ] **Step 5: Run test to verify it fails**

Run: `cd /workspace/HunyuanOCR-ROCm && python -m pytest tests/test_smoke.py -v`
Expected: FAIL (`ModuleNotFoundError: No module named 'hunyuan_ocr'`).

- [ ] **Step 6: Install the package editable + run test**

Run:
```bash
cd /workspace/HunyuanOCR-ROCm
python -m pip install -e .
python -m pytest tests/test_smoke.py -v
```
Expected: PASS.

- [ ] **Step 7: Write a minimal `Makefile`**

```makefile
.PHONY: demo eval-canary eval-full score oracle-check
demo:
	@echo "demo target — wired in Task 9"
eval-canary:
	@echo "eval-canary target — wired in Task 9"
eval-full:
	@echo "eval-full target — wired in Task 9"
score:
	@echo "score target — wired in Task 7"
oracle-check:
	@echo "oracle-check target — wired in Task 9"
```

- [ ] **Step 8: Commit**

```bash
cd /workspace/HunyuanOCR-ROCm
git add pyproject.toml src tests Makefile eval/configs/.gitkeep
git commit -m "feat: scaffold hunyuan_ocr package (Phase 1)"
```

---

## Task 2: Frozen decoding contract (`contract.py`)

**Files:**
- Create: `src/hunyuan_ocr/contract.py`
- Test: `tests/test_contract.py`

**Interfaces:**
- Produces: `hunyuan_ocr.contract.CONTRACT` (a `Contract` instance) and `hunyuan_ocr.contract.SAMPLING` (dict).

- [ ] **Step 1: Write the failing test `tests/test_contract.py`**

```python
from hunyuan_ocr.contract import CONTRACT, SAMPLING


def test_sampling_is_frozen_and_greedy():
    assert SAMPLING["do_sample"] is False  # temp=0 -> greedy
    assert SAMPLING["repetition_penalty"] == 1.08
    assert SAMPLING["max_new_tokens"] == 32768
    assert SAMPLING["use_cache"] is True


def test_contract_task_and_postprocessors():
    assert CONTRACT.task_type == "doc_parse"
    assert CONTRACT.prompt == (
        "提取文档图片中正文的所有信息用markdown格式表示，其中页眉、页脚部分忽略，"
        "表格用html格式表达，文档中公式用latex格式表示，按照阅读顺序组织进行解析。"
    )
    assert CONTRACT.postprocessors == ("clean_repeated_substrings", "process_one")


def test_contract_model_load_flags():
    assert CONTRACT.dtype == "bfloat16"
    assert CONTRACT.attn_implementation == "eager"
    assert CONTRACT.repeat_min_repeats == 8


def test_contract_decode_flags():
    assert CONTRACT.skip_special_tokens is True
    assert CONTRACT.clean_up_tokenization_spaces is False
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /workspace/HunyuanOCR-ROCm && python -m pytest tests/test_contract.py -v`
Expected: FAIL (`ModuleNotFoundError` / `ImportError`).

- [ ] **Step 3: Implement `src/hunyuan_ocr/contract.py`**

```python
"""The FROZEN decoding contract — the single shared layer across backends.

Phase 1 (transformers) establishes BASELINE against these values; Phases 2 (vLLM)
and 3 (llama.cpp) MUST match it. Changing any value here re-baselines everything.
All values are copied verbatim from the upstream HunyuanOCR-1.5 inference recipe
(inference/transformers/infer_hf_8gpu_hyocr15.py, aligned with infer_vllm_client.py).
"""

from __future__ import annotations
from dataclasses import dataclass, field


# Sampling kwargs passed straight to model.generate() / mapped onto each backend.
SAMPLING: dict = {
    "do_sample": False,  # temperature=0.0 -> greedy
    "repetition_penalty": 1.08,
    "max_new_tokens": 32768,
    "use_cache": True,
}


@dataclass(frozen=True)
class Contract:
    # Task
    task_type: str = "doc_parse"
    prompt: str = (
        "提取文档图片中正文的所有信息用markdown格式表示，其中页眉、页脚部分忽略，"
        "表格用html格式表达，文档中公式用latex格式表示，按照阅读顺序组织进行解析。"
    )

    # Model loading
    dtype: str = "bfloat16"
    attn_implementation: str = "eager"

    # Decode
    skip_special_tokens: bool = True
    clean_up_tokenization_spaces: bool = False

    # Tail-repetition early-stop (has_tail_repetition min_repeats)
    repeat_min_repeats: int = 8

    # Post-processors applied in order, doc_parse only
    postprocessors: tuple = ("clean_repeated_substrings", "process_one")


CONTRACT: Contract = Contract()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_contract.py -v`
Expected: PASS (4 tests).

- [ ] **Step 5: Commit**

```bash
git add src/hunyuan_ocr/contract.py tests/test_contract.py
git commit -m "feat(contract): frozen decoding contract (sampling/prompt/postproc)"
```

---

## Task 3: Task prompts (`tasks.py`)

**Files:**
- Create: `src/hunyuan_ocr/tasks.py`
- Test: `tests/test_tasks.py`

**Interfaces:**
- Produces: `hunyuan_ocr.tasks.TASK_PROMPTS`, `hunyuan_ocr.tasks.get_prompt(task_type)`.

- [ ] **Step 1: Write the failing test `tests/test_tasks.py`**

```python
import pytest
from hunyuan_ocr.tasks import TASK_PROMPTS, get_prompt, DEFAULT_TASK


def test_doc_parse_prompt_matches_upstream():
    assert get_prompt("doc_parse") == (
        "提取文档图片中正文的所有信息用markdown格式表示，其中页眉、页脚部分忽略，"
        "表格用html格式表达，文档中公式用latex格式表示，按照阅读顺序组织进行解析。"
    )


def test_default_task_is_doc_parse():
    assert DEFAULT_TASK == "doc_parse"


def test_get_prompt_unknown_raises():
    with pytest.raises(KeyError):
        get_prompt("nope")


def test_all_twelve_tasks_present():
    expected = {
        "doc_parse",
        "structured_parse",
        "spotting_json",
        "spotting_hunyuan",
        "layout",
        "layout_parse",
        "chart_parse",
        "formula",
        "table",
        "doc_trans_en2zh",
        "trans_other2en",
        "trans_other2zh",
    }
    assert expected == set(TASK_PROMPTS.keys())
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_tasks.py -v`
Expected: FAIL (`ImportError`).

- [ ] **Step 3: Implement `src/hunyuan_ocr/tasks.py` — verbatim port of upstream `hunyuan_tasks.py`**

Copy `/root/HunyuanOCR-src/inference/vllm_0_18_1/hunyuan_tasks.py` to `src/hunyuan_ocr/tasks.py` **verbatim** (it is the authoritative source). The file defines `TASK_PROMPTS` (12 entries), `TASK_DESCRIPTIONS`, `DEFAULT_TASK = "doc_parse"`, and `get_prompt(task_type)`. Confirm `get_prompt("doc_parse")` equals the contract prompt (it does — both are upstream's official wording).

```bash
cp /root/HunyuanOCR-src/inference/vllm_0_18_1/hunyuan_tasks.py \
   /workspace/HunyuanOCR-ROCm/src/hunyuan_ocr/tasks.py
```

(If `/root/HunyuanOCR-src` is absent in the execution env, fetch from `https://raw.githubusercontent.com/Tencent-Hunyuan/HunyuanOCR/main/inference/vllm_0_18_1/hunyuan_tasks.py`.)

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_tasks.py -v`
Expected: PASS (4 tests).

- [ ] **Step 5: Commit**

```bash
git add src/hunyuan_ocr/tasks.py tests/test_tasks.py
git commit -m "feat(tasks): port upstream TASK_PROMPTS + get_prompt verbatim"
```

---

## Task 4: Post-processors (`postprocess.py`) — verbatim port

**Files:**
- Create: `src/hunyuan_ocr/postprocess.py`
- Test: `tests/test_postprocess.py`

**Interfaces:**
- Produces: `hunyuan_ocr.postprocess.has_tail_repetition`, `clean_repeated_substrings`, `process_one`.

- [ ] **Step 1: Port the upstream file verbatim**

The upstream `hunyuan_utils.py` is **identical** in `inference/transformers/` and `inference/vllm_0_18_1/`. Copy it verbatim:

```bash
cp /root/HunyuanOCR-src/inference/vllm_0_18_1/hunyuan_utils.py \
   /workspace/HunyuanOCR-ROCm/src/hunyuan_ocr/postprocess.py
```

(Fallback URL: `https://raw.githubusercontent.com/Tencent-Hunyuan/HunyuanOCR/main/inference/vllm_0_18_1/hunyuan_utils.py`.)

This file exports, verbatim:
- **Group 1 (tail-repetition):** `has_tail_repetition(text, min_repeats=8, max_unit=256)`, `clean_repeated_substrings(text, min_repeats=10)`, plus `encode_image_as_data_url`, `infer_stream` (streaming helpers — not used by the transformers path, but kept verbatim for parity).
- **Group 2 (doc_parse normalization):** `process_one(text) -> (text, stats)` applying 10 patterns (U strip layout coords, T table caption, E array{l}+\\\\, C multi-row env, C2 array &-cells, D \left/\right balance, A inline→display, W bare arithmetic, F strip eq numbers, V display-block repair). Apply ONLY for doc_parse.

Do **not** edit the ported logic — it is precision-critical; any change diverges from upstream and breaks alignment.

- [ ] **Step 2: Write the characterization test `tests/test_postprocess.py`**

These pin the ported behavior so a future accidental edit is caught.

```python
from hunyuan_ocr.postprocess import (
    has_tail_repetition,
    clean_repeated_substrings,
    process_one,
)


def test_has_tail_repetition_detects_loop():
    assert has_tail_repetition("x" * 200) is True


def test_has_tail_repetition_clean_text():
    assert has_tail_repetition("正常的中文文档内容，没有重复。" * 1) is False


def test_clean_repeated_substrings_trims_long_loop():
    body = "正文内容。" * 5
    loop = "ABCD" * 3000  # >> 2000 chars, repeats > 10x
    out = clean_repeated_substrings(body + loop)
    # upstream keeps ONE surviving copy of the unit: text[: n - length*(count-1)]
    assert out == body + "ABCD"
    assert "ABCDABCD" not in out  # the degenerate loop is collapsed to one copy


def test_clean_repeated_substrings_short_text_untouched():
    assert clean_repeated_substrings("短文本") == "短文本"


def test_process_one_splits_table_caption():
    # Pattern T: <table><caption>X</caption>... -> X\n\n<table>...
    md = "<table><caption>表1 标题</caption><tr><td>a</td></tr></table>"
    out, stats = process_one(md)
    assert out.startswith("表1 标题\n\n<table>")
    assert stats["T_captions"] == 1


def test_process_one_idempotent_on_clean_doc():
    md = "# 标题\n\n这是一段正文。\n\n$$ a^2 + b^2 = c^2 $$\n"
    out, stats = process_one(md)
    assert out == md
    assert all(v == 0 for v in stats.values())
```

- [ ] **Step 3: Run tests to verify they pass**

Run: `python -m pytest tests/test_postprocess.py -v`
Expected: PASS (6 tests).

- [ ] **Step 4: Commit**

```bash
git add src/hunyuan_ocr/postprocess.py tests/test_postprocess.py
git commit -m "feat(postprocess): port has_tail_repetition/clean_repeated_substrings/process_one verbatim"
```

---

## Task 5: OmniDocBench dataset I/O (`omnidocbench.py`)

**Files:**
- Create: `src/hunyuan_ocr/omnidocbench.py`
- Test: `tests/test_omnidocbench.py`, `tests/fixtures/mini_omnidocbench.json`

**Interfaces:**
- Produces: `iter_page_images(gt_json: Path, images_dir: Path) -> Iterator[tuple[str, Path]]` yielding `(image_stem, abs_image_path)`; `derive_prediction_filename(image_path) -> str` → `<stem>.md`.

- [ ] **Step 1: Create fixture `tests/fixtures/mini_omnidocbench.json`**

```json
[
  {"page_info": {"image_path": "page-aaaa-1111.png", "page_no": 1}, "layout_dets": []},
  {"page_info": {"image_path": "PPT_eng_page_002.png", "page_no": 2}, "layout_dets": []}
]
```

Also create the two matching empty image files so path resolution can be tested:
```bash
mkdir -p tests/fixtures/images
touch tests/fixtures/images/page-aaaa-1111.png tests/fixtures/images/PPT_eng_page_002.png
```

- [ ] **Step 2: Write the failing test `tests/test_omnidocbench.py`**

```python
from pathlib import Path
from hunyuan_ocr.omnidocbench import iter_page_images, derive_prediction_filename

FIX = Path(__file__).parent / "fixtures"


def test_derive_prediction_filename():
    assert derive_prediction_filename("images/page-aaaa-1111.png") == "page-aaaa-1111.md"
    assert derive_prediction_filename("/anywhere/PPT_eng_page_002.png") == "PPT_eng_page_002.md"


def test_iter_page_images_resolves_under_images_dir():
    pairs = list(iter_page_images(FIX / "mini_omnidocbench.json", FIX / "images"))
    stems = [s for s, _ in pairs]
    assert stems == ["page-aaaa-1111", "PPT_eng_page_002"]
    for _, p in pairs:
        assert p.exists()
        assert p.parent == (FIX / "images")
```

- [ ] **Step 3: Run test to verify it fails**

Run: `python -m pytest tests/test_omnidocbench.py -v`
Expected: FAIL (`ImportError`).

- [ ] **Step 4: Implement `src/hunyuan_ocr/omnidocbench.py`**

```python
"""OmniDocBench v1.6 dataset iteration + prediction filename mapping.

Ground-truth JSON (e.g. /workspace/OmniDocBench_data/OmniDocBench.json) is a list
of page dicts; each page_info.image_path is a BARE basename resolved under the
dataset's images/ directory. Subsets OmniDocBench_150.json / OmniDocBench_30.json
share the same format.
"""

from __future__ import annotations
import json
from pathlib import Path
from typing import Iterator


def derive_prediction_filename(image_path: str | Path) -> str:
    """Map an image path to its OmniDocBench prediction filename: ``<stem>.md``."""
    return f"{Path(image_path).stem}.md"


def iter_page_images(gt_json: str | Path, images_dir: str | Path) -> Iterator[tuple[str, Path]]:
    """Yield (image_stem, abs_image_path) for every page in the ground-truth JSON.

    ``images_dir`` is the directory holding the page images (e.g. .../images).
    """
    images_dir = Path(images_dir)
    with open(gt_json, encoding="utf-8") as f:
        pages = json.load(f)
    for page in pages:
        rel = page["page_info"]["image_path"]
        abs_path = images_dir / rel
        yield Path(rel).stem, abs_path
```

- [ ] **Step 5: Run test to verify it passes**

Run: `python -m pytest tests/test_omnidocbench.py -v`
Expected: PASS (2 tests).

- [ ] **Step 6: Commit**

```bash
git add src/hunyuan_ocr/omnidocbench.py tests/test_omnidocbench.py tests/fixtures
git commit -m "feat(omnidocbench): dataset iteration + prediction filename mapping"
```

---

## Task 6: Scoring harness (`scoring.py`)

**Files:**
- Create: `src/hunyuan_ocr/scoring.py`
- Create: `eval/configs/hunyuanocr-1.5_linux-rocm.yaml`
- Test: `tests/test_scoring.py`, `tests/fixtures/mini_metric_result.json`, `tests/fixtures/mini_run_summary.json`

**Interfaces:**
- Produces: `write_eval_config(gt_json, pred_dir, out_yaml)`, `run_scorer(omnidocbench_repo, config_yaml, venv_python=None) -> CompletedProcess`, `parse_run_summary(result_dir, save_name) -> dict`, `overall_score(metrics) -> float`.

- [ ] **Step 1: Write the eval config template `eval/configs/hunyuanocr-1.5_linux-rocm.yaml`**

```yaml
end2end_eval:
  metrics:
    text_block: {metric: [Edit_dist]}
    display_formula: {metric: [Edit_dist, CDM], cdm_workers: 13}
    table: {metric: [TEDS, Edit_dist], teds_workers: 13}
    reading_order: {metric: [Edit_dist]}
  dataset:
    dataset_name: end2end_dataset
    ground_truth: {data_path: /workspace/OmniDocBench_data/OmniDocBench.json}
    prediction: {data_path: REPLACE_WITH_PREDICTIONS_DIR}
    match_method: quick_match
    match_workers: 13
    quick_match_truncated_timeout_sec: 300
    match_timeout_sec: 420
    timeout_fallback_max_chunk_span: 10
    timeout_fallback_order_penalty: 0.10
```

- [ ] **Step 2: Create fixtures `tests/fixtures/mini_metric_result.json` and `tests/fixtures/mini_run_summary.json`**

`mini_metric_result.json` (trimmed to the keys `parse_run_summary` reads):
```json
{
  "text_block": {"all": {"Edit_dist": {"ALL_page_avg": 0.04}}},
  "display_formula": {"page": {"CDM": {"ALL": 0.94}}},
  "table": {"page": {"TEDS": {"ALL": 0.93}}},
  "reading_order": {"all": {"Edit_dist": {"ALL_page_avg": 0.13}}}
}
```
`mini_run_summary.json`:
```json
{"notebook_metric_summary": {"overall_notebook": 94.33}}
```

- [ ] **Step 3: Write the failing test `tests/test_scoring.py`**

```python
import json
import subprocess
from pathlib import Path
from unittest.mock import patch

from hunyuan_ocr import scoring

FIX = Path(__file__).parent / "fixtures"


def test_overall_score_formula():
    # v1.6 Overall = ((1-text_edit)*100 + cdm*100 + teds*100) / 3
    metrics = {
        "text_edit_dist": 0.04,
        "formula_cdm": 0.94,
        "table_teds": 0.93,
    }
    assert abs(scoring.overall_score(metrics) - 94.33333333333333) < 1e-6


def test_write_eval_config_substitutes_pred_dir(tmp_path):
    gt = "/workspace/OmniDocBench_data/OmniDocBench.json"
    out = tmp_path / "c.yaml"
    scoring.write_eval_config(gt_json=gt, pred_dir="/tmp/preds", out_yaml=out)
    txt = out.read_text()
    assert "data_path: /tmp/preds" in txt
    assert "data_path: /workspace/OmniDocBench_data/OmniDocBench.json" in txt
    assert "quick_match" in txt


def test_parse_run_summary_reads_overall_and_per_task():
    res = scoring.parse_run_summary(FIX, save_name="mini")
    assert res["overall"] == 94.33
    assert res["text_edit_dist"] == 0.04
    assert res["formula_cdm"] == 0.94
    assert res["table_teds"] == 0.93
    assert res["reading_order_edit"] == 0.13


def test_run_scorer_invokes_pdf_validation_with_venv_python():
    with patch("hunyuan_ocr.scoring.subprocess.run") as mock:
        mock.return_value = subprocess.CompletedProcess([], 0, "", "")
        scoring.run_scorer(
            omnidocbench_repo="/root/ocr-eval/OmniDocBench",
            config_yaml="/tmp/c.yaml",
            venv_python="/root/ocr-eval/OmniDocBench/.venv/bin/python",
        )
        cmd = mock.call_args[0][0]
        assert cmd[0] == "/root/ocr-eval/OmniDocBench/.venv/bin/python"
        assert cmd[1] == "pdf_validation.py"
        assert "--config" in cmd
```

- [ ] **Step 4: Run test to verify it fails**

Run: `python -m pytest tests/test_scoring.py -v`
Expected: FAIL (`ImportError`).

- [ ] **Step 5: Implement `src/hunyuan_ocr/scoring.py`**

```python
"""OmniDocBench v1.6 scoring driver.

Writes an eval config, invokes the OmniDocBench scorer (pdf_validation.py) in its
own 3.11 venv, and parses the resulting metric_result.json / run_summary.json.
Overall = ((1 - text_EditDist)*100 + formula_CDM*100 + table_TEDS*100) / 3
(reading-order EditDist is reported separately, NOT part of Overall).
"""

from __future__ import annotations
import json
import subprocess
import sys
from pathlib import Path

import yaml

DEFAULT_VENV_PYTHON = "/root/ocr-eval/OmniDocBench/.venv/bin/python"
DEFAULT_OMNIDOCBENCH_REPO = "/root/ocr-eval/OmniDocBench"
TEMPLATE_CONFIG = Path(__file__).resolve().parents[2] / "eval" / "configs" / "hunyuanocr-1.5_linux-rocm.yaml"


def overall_score(metrics: dict) -> float:
    """v1.6 Overall from raw metrics (each in [0,1])."""
    text = metrics["text_edit_dist"]
    cdm = metrics["formula_cdm"]
    teds = metrics["table_teds"]
    return ((1.0 - text) * 100.0 + cdm * 100.0 + teds * 100.0) / 3.0


def write_eval_config(*, gt_json: str, pred_dir: str, out_yaml: Path) -> None:
    """Materialize an eval config from the template, substituting GT + pred paths."""
    cfg = yaml.safe_load(TEMPLATE_CONFIG.read_text(encoding="utf-8"))
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
    """Read overall + per-task numbers. save_name = basename(pred_dir) + '_quick_match'."""
    result_dir = Path(result_dir)
    metric = json.loads((result_dir / f"{save_name}_metric_result.json").read_text(encoding="utf-8"))
    summary = json.loads((result_dir / f"{save_name}_run_summary.json").read_text(encoding="utf-8"))
    return {
        "overall": summary["notebook_metric_summary"]["overall_notebook"],
        "text_edit_dist": metric["text_block"]["all"]["Edit_dist"]["ALL_page_avg"],
        "formula_cdm": metric["display_formula"]["page"]["CDM"]["ALL"],
        "table_teds": metric["table"]["page"]["TEDS"]["ALL"],
        "reading_order_edit": metric["reading_order"]["all"]["Edit_dist"]["ALL_page_avg"],
    }
```

- [ ] **Step 6: Run test to verify it passes**

Run: `python -m pytest tests/test_scoring.py -v`
Expected: PASS (4 tests).

- [ ] **Step 7: Commit**

```bash
git add src/hunyuan_ocr/scoring.py eval/configs/hunyuanocr-1.5_linux-rocm.yaml tests/test_scoring.py tests/fixtures/mini_metric_result.json tests/fixtures/mini_run_summary.json
git commit -m "feat(scoring): OmniDocBench config writer + scorer runner + result parser"
```

---

## Task 7: Transformers backend (`backends/transformers.py`)

**Files:**
- Create: `src/hunyuan_ocr/backends/transformers.py`
- Test: `tests/test_transformers_wiring.py` (CPU, mocked); `tests/test_transformers_smoke.py` (GPU, skipped without model)

**Interfaces:**
- Consumes: `hunyuan_ocr.contract.CONTRACT`, `hunyuan_ocr.postprocess.{clean_repeated_substrings, process_one}`, `hunyuan_ocr.tasks.get_prompt`.
- Produces: `load_model_and_processor(model_path) -> (model, processor)`, `build_messages(image_path, prompt)`, `infer_one(model, processor, image_path, prompt) -> str`.

- [ ] **Step 1: Write the CPU wiring test `tests/test_transformers_wiring.py`**

```python
from hunyuan_ocr.backends.transformers import build_messages
from hunyuan_ocr.contract import CONTRACT


def test_build_messages_matches_upstream_shape():
    msgs = build_messages("/x/y/page-1.png", CONTRACT.prompt)
    assert msgs[0] == {"role": "system", "content": ""}
    user = msgs[1]
    assert user["role"] == "user"
    assert {"type": "image", "image": "/x/y/page-1.png"} in user["content"]
    assert {"type": "text", "text": CONTRACT.prompt} in user["content"]
```

- [ ] **Step 2: Write the GPU smoke test `tests/test_transformers_smoke.py`**

```python
import os
import pytest

MODEL_PATH = os.environ.get("HUNYUANOCR_MODEL", "/root/models/HunyuanOCR")
SAMPLE_IMG = os.environ.get("HUNYUANOCR_SAMPLE_IMG", "")


@pytest.mark.skipif(
    not os.path.isdir(MODEL_PATH) or not SAMPLE_IMG,
    reason="needs HUNYUANOCR_MODEL dir + HUNYUANOCR_SAMPLE_IMG on a gfx1100 box",
)
def test_infer_one_returns_markdown():
    from hunyuan_ocr.backends.transformers import load_model_and_processor, infer_one
    from hunyuan_ocr.contract import CONTRACT

    model, processor = load_model_and_processor(MODEL_PATH)
    md = infer_one(model, processor, SAMPLE_IMG, CONTRACT.prompt)
    assert isinstance(md, str) and len(md) > 0
```

- [ ] **Step 3: Run wiring test to verify it fails**

Run: `python -m pytest tests/test_transformers_wiring.py -v`
Expected: FAIL (`ImportError`).

- [ ] **Step 4: Implement `src/hunyuan_ocr/backends/transformers.py` — ported from upstream `infer_hf_8gpu_hyocr15.py`**

```python
"""Transformers backend for HunyuanOCR-1.5 (Phase 1 oracle).

Ported from upstream inference/transformers/infer_hf_8gpu_hyocr15.py:
  HunYuanVLForConditionalGeneration + AutoProcessor, dtype=bfloat16, attn=eager,
  greedy decode (do_sample=False, repetition_penalty=1.08, max_new_tokens=32768),
  tail-repetition StoppingCriteria, clean_repeated_substrings + process_one.
Torch/transformers imported lazily so importing this module needs no GPU.
"""

from __future__ import annotations
import importlib
import sys
from typing import List

from ..contract import CONTRACT
from ..postprocess import clean_repeated_substrings, process_one


def _patch_hunyuan_tokenizer_special_tokens(tokenizer) -> None:
    """Backfill missing special-token attrs on older HunyuanOCR tokenizers."""
    init_kwargs = getattr(tokenizer, "init_kwargs", {}) or {}
    extra_tokens = init_kwargs.get("extra_special_tokens", {}) or {}
    defaults = {
        "image_token": "<｜hy_place▁holder▁no▁102｜>",
        "image_start_token": "<｜hy_place▁holder▁no▁100｜>",
        "image_end_token": "<｜hy_place▁holder▁no▁101｜>",
        "video_token": "<｜hy_place▁holder▁no▁103｜>",
        "video_start_token": "<｜hy_place▁holder▁no▁104｜>",
        "video_end_token": "<｜hy_place▁holder▁no▁105｜>",
    }
    for name, default_value in defaults.items():
        if hasattr(tokenizer, name):
            continue
        value = extra_tokens.get(name)
        if value is None and name == "video_token":
            value = extra_tokens.get("image_token")
        setattr(tokenizer, name, value or default_value)


def _load_processor_with_patch(model_path: str):
    from transformers import AutoImageProcessor, AutoTokenizer

    proc_mod = importlib.import_module("transformers.models.hunyuan_vl.processing_hunyuan_vl")
    HunYuanVLProcessor = proc_mod.HunYuanVLProcessor
    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)
    _patch_hunyuan_tokenizer_special_tokens(tokenizer)
    image_processor = AutoImageProcessor.from_pretrained(model_path)
    video_processor = None
    try:
        from transformers import AutoVideoProcessor

        video_processor = AutoVideoProcessor.from_pretrained(model_path)
    except Exception:
        video_processor = None
    try:
        return HunYuanVLProcessor(image_processor=image_processor, tokenizer=tokenizer, video_processor=video_processor)
    except TypeError:
        return HunYuanVLProcessor(image_processor, tokenizer, video_processor)


def load_model_and_processor(model_path: str, device: str = "cuda:0"):
    import torch
    from transformers import AutoProcessor, HunYuanVLForConditionalGeneration

    dtype = getattr(torch, CONTRACT.dtype)
    try:
        processor = AutoProcessor.from_pretrained(model_path, use_fast=False)
    except AttributeError as e:
        if "video_token" not in str(e):
            raise
        print("[warn] AutoProcessor tokenizer lacks video_token; retrying with patched tokenizer.", file=sys.stderr)
        processor = _load_processor_with_patch(model_path)
    model = HunYuanVLForConditionalGeneration.from_pretrained(
        model_path,
        attn_implementation=CONTRACT.attn_implementation,
        dtype=dtype,
    )
    model = model.to(device)
    model.eval()
    return model, processor


def build_messages(image_path: str, prompt: str) -> List[dict]:
    """Upstream message shape: empty system + user[image, text]."""
    return [
        {"role": "system", "content": ""},
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image_path},
                {"type": "text", "text": prompt},
            ],
        },
    ]


def _build_tail_repetition_stop(processor, prompt_len: int):
    """StoppingCriteria mirroring the vLLM streaming early-stop (per upstream)."""
    from transformers import StoppingCriteria, StoppingCriteriaList
    from ..postprocess import has_tail_repetition

    tokenizer = processor.tokenizer
    min_repeats = CONTRACT.repeat_min_repeats
    check_start_chars, check_step_chars, token_probe_step = 4000, 1000, 64

    class TailRepetitionStop(StoppingCriteria):
        def __init__(self):
            self._next_check_at_chars = check_start_chars
            self._last_probe_tokens = 0
            self._triggered = False

        def __call__(self, input_ids, scores, **kwargs):
            if self._triggered:
                return True
            new_tokens = input_ids[0, prompt_len:]
            n_new = int(new_tokens.numel())
            if n_new - self._last_probe_tokens < token_probe_step:
                return False
            self._last_probe_tokens = n_new
            try:
                text = tokenizer.decode(new_tokens, skip_special_tokens=True)
            except Exception:
                return False
            acc_len = len(text)
            if acc_len < self._next_check_at_chars:
                return False
            self._next_check_at_chars = acc_len + check_step_chars
            if has_tail_repetition(text[-8000:], min_repeats=min_repeats):
                self._triggered = True
                return True
            return False

    return StoppingCriteriaList([TailRepetitionStop()])


def infer_one(model, processor, image_path: str, prompt: str, device: str = "cuda:0") -> str:
    """Run one image through the model with the frozen contract; return markdown."""
    import torch
    from PIL import Image

    tokenizer = processor.tokenizer
    eos_token_id = getattr(tokenizer, "eos_token_id", None)
    pad_token_id = getattr(tokenizer, "pad_token_id", None) or eos_token_id

    with Image.open(image_path) as raw:
        image = raw.convert("RGB")

    text = processor.apply_chat_template(build_messages(image_path, prompt), tokenize=False, add_generation_prompt=True)
    inputs = processor(text=[text], images=image, padding=True, return_tensors="pt").to(device)
    input_ids = inputs["input_ids"] if "input_ids" in inputs else inputs["inputs"]
    prompt_len = int(input_ids.shape[1])

    stopping_criteria = _build_tail_repetition_stop(processor, prompt_len=prompt_len)
    gen_kwargs = dict(
        max_new_tokens=32768,
        do_sample=False,
        repetition_penalty=1.08,
        use_cache=True,
        stopping_criteria=stopping_criteria,
    )
    if eos_token_id is not None:
        gen_kwargs["eos_token_id"] = eos_token_id
    if pad_token_id is not None:
        gen_kwargs["pad_token_id"] = pad_token_id

    with torch.inference_mode():
        generated_ids = model.generate(**inputs, **gen_kwargs)
    trimmed = [out_ids[len(in_ids) :] for in_ids, out_ids in zip(input_ids, generated_ids)]
    decoded = processor.batch_decode(trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)
    out_text = decoded[0] if decoded else ""
    out_text = clean_repeated_substrings(out_text)
    try:
        out_text, _ = process_one(out_text)  # doc_parse normalization (frozen postproc)
    except Exception:
        pass
    return out_text
```

Note: the literal `1.08` and `32768` in `infer_one` mirror upstream's frozen recipe (also in `CONTRACT`); keep them inline here exactly as upstream does so the generate call is unambiguous.

- [ ] **Step 5: Run wiring test to verify it passes**

Run: `python -m pytest tests/test_transformers_wiring.py -v`
Expected: PASS (1 test). (Smoke test stays skipped until the model is present.)

- [ ] **Step 6: Commit**

```bash
git add src/hunyuan_ocr/backends/transformers.py tests/test_transformers_wiring.py tests/test_transformers_smoke.py
git commit -m "feat(backend): transformers backend ported from upstream infer_hf"
```

---

## Task 8: Phase-1 multi-GPU driver (`scripts/run_phase1_transformers.py`)

**Files:**
- Create: `scripts/run_phase1_transformers.py`

**Interfaces:**
- Consumes: `hunyuan_ocr.backends.transformers.{load_model_and_processor, infer_one}`, `hunyuan_ocr.omnidocbench.iter_page_images`, `hunyuan_ocr.contract.CONTRACT`.
- Produces: `<pred_dir>/<stem>.md` per page (resumable, sharded across GPUs).

- [ ] **Step 1: Write `scripts/run_phase1_transformers.py`** (port of the upstream spawn driver, adapted to write `.md`)

```python
#!/usr/bin/env python3
"""Phase-1 driver: run HunyuanOCR-1.5 (transformers) over OmniDocBench pages.

Spawns one worker process per GPU, shards the page list across them, and writes
one <stem>.md prediction per page. Resumable (skips pages whose .md exists).
Usage:
  python scripts/run_phase1_transformers.py \
      --gt-json /workspace/OmniDocBench_data/OmniDocBench.json \
      --images-dir /workspace/OmniDocBench_data/images \
      --pred-dir /root/hunyuanocr-results/phase1-transformers/preds \
      --model /root/models/HunyuanOCR \
      --gpu-ids 0,1,2 \
      [--limit N]   # quick smoke run on first N pages (single GPU recommended)
"""

from __future__ import annotations
import argparse
import json
import multiprocessing as mp
import os
import sys
import time
from pathlib import Path


def load_page_list(gt_json: str, images_dir: str, limit: int | None = None):
    pages = json.load(open(gt_json, encoding="utf-8"))
    if limit:
        pages = pages[:limit]
    return [
        (Path(p["page_info"]["image_path"]).stem, os.path.join(images_dir, p["page_info"]["image_path"])) for p in pages
    ]


def shard(items, n):
    k = -(-len(items) // n)
    return [items[i : i + k] for i in range(0, len(items), k)]


def worker(gpu_id: int, chunk, args_dict: dict):
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    # Import heavy deps inside the worker (child pinned to one GPU).
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
    from hunyuan_ocr.backends.transformers import load_model_and_processor, infer_one
    from hunyuan_ocr.contract import CONTRACT

    a = argparse.Namespace(**args_dict)
    device = "cuda:0"
    print(f"[GPU {gpu_id}] loading model ...", flush=True)
    t0 = time.time()
    model, processor = load_model_and_processor(a.model, device=device)
    print(f"[GPU {gpu_id}] model ready in {time.time() - t0:.1f}s", flush=True)

    os.makedirs(a.pred_dir, exist_ok=True)
    todo = []
    for stem, img in chunk:
        if (Path(a.pred_dir) / f"{stem}.md").exists():
            continue
        todo.append((stem, img))
    print(f"[GPU {gpu_id}] {len(todo)} to do ({len(chunk) - len(todo)} resumed)", flush=True)

    for i, (stem, img) in enumerate(todo):
        try:
            md = infer_one(model, processor, img, CONTRACT.prompt, device=device)
            status = "ok"
        except Exception as e:
            md = f"ERROR: {type(e).__name__}: {e}"
            status = "failed"
        (Path(a.pred_dir) / f"{stem}.md").write_text(md, encoding="utf-8")
        if (i + 1) % 10 == 0:
            print(f"[GPU {gpu_id}] {i + 1}/{len(todo)} ({status})", flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--gt-json", required=True)
    p.add_argument("--images-dir", required=True)
    p.add_argument("--pred-dir", required=True)
    p.add_argument("--model", default="/root/models/HunyuanOCR")
    p.add_argument("--gpu-ids", default="0,1,2")
    p.add_argument("--limit", type=int, default=None)
    args = p.parse_args()

    pages = load_page_list(args.gt_json, args.images_dir, args.limit)
    gpu_ids = [int(x) for x in args.gpu_ids.split(",") if x.strip()]
    chunks = shard(pages, len(gpu_ids))
    print(f"[info] {len(pages)} pages across GPUs {gpu_ids}: {[len(c) for c in chunks]}", flush=True)

    ctx = mp.get_context("spawn")
    procs = [
        ctx.Process(target=worker, args=(gid, chunks[i], vars(args)), daemon=False) for i, gid in enumerate(gpu_ids)
    ]
    for pr in procs:
        pr.start()
    for pr in procs:
        pr.join()
    print("[done] all workers finished", flush=True)


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Smoke-check the arg parser (no GPU needed)**

Run: `cd /workspace/HunyuanOCR-ROCm && python scripts/run_phase1_transformers.py --help`
Expected: prints usage with all flags, exit 0.

- [ ] **Step 3: Commit**

```bash
git add scripts/run_phase1_transformers.py
git commit -m "feat(scripts): Phase-1 multi-GPU transformers driver (sharded, resumable)"
```

---

## Task 9: Canary + score scripts, run Phase 1, record BASELINE

**Files:**
- Create: `scripts/regression_canary.py`, `scripts/score_predictions.py`
- Create: `reports/phase1-transformers.md` (filled with real numbers)
- Modify: `Makefile` (wire targets)

**Interfaces:**
- Consumes: `hunyuan_ocr.scoring.{write_eval_config, run_scorer, parse_run_summary, overall_score}`, `hunyuan_ocr.omnidocbench`.
- Produces: the **BASELINE** record (`reports/phase1-transformers.md` + a frozen canary score) — the ground truth Phases 2–3 must match.

- [ ] **Step 1: Write `scripts/score_predictions.py`**

```python
#!/usr/bin/env python3
"""Score a predictions dir against OmniDocBench v1.6 and print the score table.

Usage:
  python scripts/score_predictions.py \
      --pred-dir /root/hunyuanocr-results/phase1-transformers/preds \
      --gt-json /workspace/OmniDocBench_data/OmniDocBench.json \
      [--label transformers] [--no-cdm]
"""

from __future__ import annotations
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from hunyuan_ocr import scoring


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--pred-dir", required=True)
    p.add_argument("--gt-json", required=True)
    p.add_argument("--label", default="backend")
    p.add_argument("--omnidocbench-repo", default=scoring.DEFAULT_OMNIDOCBENCH_REPO)
    p.add_argument("--venv-python", default=scoring.DEFAULT_VENV_PYTHON)
    args = p.parse_args()

    cfg_path = Path(args.pred_dir) / "_eval_config.yaml"
    scoring.write_eval_config(gt_json=args.gt_json, pred_dir=args.pred_dir, out_yaml=cfg_path)
    res = scoring.run_scorer(
        omnidocbench_repo=args.omnidocbench_repo, config_yaml=str(cfg_path), venv_python=args.venv_python
    )
    if res.returncode != 0:
        print(res.stdout[-4000:])
        print(res.stderr[-4000:], file=sys.stderr)
        sys.exit(f"[error] scorer failed (rc={res.returncode})")

    save_name = f"{Path(args.pred_dir).name}_quick_match"
    s = scoring.parse_run_summary(Path(args.omnidocbench_repo) / "result", save_name)
    print(f"\n=== {args.label} — OmniDocBench v1.6 ===")
    print(f"  Overall          : {s['overall']:.2f}")
    print(f"  text  EditDist   : {s['text_edit_dist']:.4f}   -> {(1 - s['text_edit_dist']) * 100:.2f}")
    print(f"  formula CDM      : {s['formula_cdm']:.4f}   -> {s['formula_cdm'] * 100:.2f}")
    print(f"  table  TEDS      : {s['table_teds']:.4f}   -> {s['table_teds'] * 100:.2f}")
    print(f"  order  EditDist  : {s['reading_order_edit']:.4f}")
    recomputed = scoring.overall_score(
        {"text_edit_dist": s["text_edit_dist"], "formula_cdm": s["formula_cdm"], "table_teds": s["table_teds"]}
    )
    print(f"  (overall recomputed: {recomputed:.2f})")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Write `scripts/regression_canary.py`**

```python
#!/usr/bin/env python3
"""Run the transformers backend on the 150-page canary, then score it.

The canary is the project's minute-level regression oracle (Absorb-C):
later phases compare their canary score against this Phase-1 canary score.
Usage:
  python scripts/regression_canary.py --model /root/models/HunyuanOCR --gpu-ids 0,1,2
"""

from __future__ import annotations
import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = Path("/workspace/OmniDocBench_data")
CANARY_GT = DATA / "OmniDocBench_150.json"
PRED = Path("/root/hunyuanocr-results/canary-transformers/preds")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="/root/models/HunyuanOCR")
    p.add_argument("--gpu-ids", default="0,1,2")
    args = p.parse_args()

    subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "run_phase1_transformers.py"),
            "--gt-json",
            str(CANARY_GT),
            "--images-dir",
            str(DATA / "images"),
            "--pred-dir",
            str(PRED),
            "--model",
            args.model,
            "--gpu-ids",
            args.gpu_ids,
        ],
        check=True,
    )

    subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "score_predictions.py"),
            "--pred-dir",
            str(PRED),
            "--gt-json",
            str(CANARY_GT),
            "--label",
            "transformers-canary-150",
        ],
        check=True,
    )


if __name__ == "__main__":
    main()
```

- [ ] **Step 3: Wire the `Makefile` targets**

Replace the stub bodies of `eval-canary`, `eval-full`, `score`, `oracle-check`, `demo`:

```makefile
MODEL ?= /root/models/HunyuanOCR
GPUS ?= 0,1,2
DATA ?= /workspace/OmniDocBench_data
GT_FULL ?= $(DATA)/OmniDocBench.json
GT_CANARY ?= $(DATA)/OmniDocBench_150.json
PRED_FULL ?= /root/hunyuanocr-results/phase1-transformers/preds
PRED_CANARY ?= /root/hunyuanocr-results/canary-transformers/preds

.PHONY: demo eval-canary eval-full score score-canary oracle-check
demo:
	python scripts/run_phase1_transformers.py --gt-json $(DATA)/OmniDocBench_30.json --images-dir $(DATA)/images --pred-dir /root/hunyuanocr-results/demo/preds --model $(MODEL) --gpu-ids 0 --limit 3
eval-canary:
	python scripts/run_phase1_transformers.py --gt-json $(GT_CANARY) --images-dir $(DATA)/images --pred-dir $(PRED_CANARY) --model $(MODEL) --gpu-ids $(GPUS)
	python scripts/score_predictions.py --pred-dir $(PRED_CANARY) --gt-json $(GT_CANARY) --label transformers-canary-150
eval-full:
	python scripts/run_phase1_transformers.py --gt-json $(GT_FULL) --images-dir $(DATA)/images --pred-dir $(PRED_FULL) --model $(MODEL) --gpu-ids $(GPUS)
score:
	python scripts/score_predictions.py --pred-dir $(PRED_FULL) --gt-json $(GT_FULL) --label transformers
score-canary:
	python scripts/score_predictions.py --pred-dir $(PRED_CANARY) --gt-json $(GT_CANARY) --label transformers-canary-150
oracle-check:
	@echo "oracle = transformers canary (150). Re-run: make score-canary"
```

- [ ] **Step 4: Commit the scripts + Makefile**

```bash
git add scripts/score_predictions.py scripts/regression_canary.py Makefile
git commit -m "feat(scripts): canary driver + scorer + Makefile targets"
```

- [ ] **Step 5: Provision the model-venv-transformers (Python 3.12)**

Run:
```bash
python3.12 -m venv /root/hunyuanocr-venvs/transformers
/root/hunyuanocr-venvs/transformers/bin/pip install -e /workspace/HunyuanOCR-ROCm
# torch: install the ROCm 6.x gfx1100 wheel per the vLLM/Unlimited-OCR precedent already on this box
/root/hunyuanocr-venvs/transformers/bin/pip install "transformers==5.13.0" pillow tqdm pyyaml requests pytest
```
Expected: `transformers` import works; `HunYuanVLForConditionalGeneration` resolves (`python -c "from transformers import HunYuanVLForConditionalGeneration"`).

- [ ] **Step 6: Download the weights to `/root/models/HunyuanOCR`**

Run:
```bash
/root/hunyuanocr-venvs/transformers/bin/pip install huggingface_hub
/root/hunyuanocr-venvs/transformers/bin/python -c "
from huggingface_hub import snapshot_download
snapshot_download('tencent/HunyuanOCR', local_dir='/root/models/HunyuanOCR')
"
```
Expected: `/root/models/HunyuanOCR` populated with `config.json`, `*.safetensors`, `preprocessor_config.json`, `chat_template.jinja`, `tokenizer*`.

- [ ] **Step 7: Smoke run on 3 pages (`make demo`)**

Run: `cd /workspace/HunyuanOCR-ROCm && PATH=/root/hunyuanocr-venvs/transformers/bin:$PATH make demo`
Expected: 3 `.md` files written under `/root/hunyuanocr-results/demo/preds`, each containing real OCR markdown (no `ERROR:`). If `ERROR:` appears, debug before proceeding.

- [ ] **Step 8: Run the canary (150 pages) and score it**

Run: `cd /workspace/HunyuanOCR-ROCm && PATH=/root/hunyuanocr-venvs/transformers/bin:$PATH make eval-canary`
Expected: 150 `.md` files; a printed score table. Record the canary Overall → this is the frozen regression oracle.

- [ ] **Step 9: Run the FULL set (1651 pages) and score it → BASELINE**

Run: `cd /workspace/HunyuanOCR-ROCm && PATH=/root/hunyuanocr-venvs/transformers/bin:$PATH make eval-full && make score`
Expected: 1651 `.md` files; a printed score table. The **Overall** is the **BASELINE**. Sanity: it should land ≈ 94.74 ± ~1.0; a larger gap flags a protocol/revision mismatch to investigate before trusting BASELINE.

- [ ] **Step 10: Write the Phase-1 report `reports/phase1-transformers.md`**

Fill in the measured numbers (replace the placeholders with real values from Step 9):

```markdown
# Phase 1 — transformers oracle (BASELINE)

**Date:** <fill>
**Backend:** transformers 5.13.0, HunYuanVLForConditionalGeneration, bf16, attn=eager, greedy
**Hardware:** 3× gfx1100 (RDNA3)
**Dataset:** OmniDocBench v1.6 (1651 pages)

## BASELINE (the precision ground-truth for Phases 2–3)

| Metric | Value |
|---|---|
| **Overall** | **<fill>** |
| text EditDist | <fill> |
| formula CDM | <fill> |
| table TEDS | <fill> |
| reading-order EditDist | <fill> |

Sanity vs upstream 94.74: <within ±1.0 / INVESTIGATE>

## Frozen canary oracle (150 pages)

Overall (canary) = <fill>   — the regression target for Phases 2–3 (`make score-canary`).

## Gate for Phases 2–3

A backend passes when its full-set score is within **±0.3 Overall / ±0.5 per-task** of the BASELINE above.

## Frozen contract

See `src/hunyuan_ocr/contract.py` + `tasks.py` + `postprocess.py`. Changing any of these re-baselines.
```

- [ ] **Step 11: Commit the report + artifacts index**

```bash
git add reports/phase1-transformers.md
git commit -m "docs(phase1): record transformers BASELINE + frozen canary oracle"
```

(Prediction `.md` files are gitignored; commit only the report. If reproducibility artifacts are desired, copy the `_run_stats`/score JSONs into `results/omnidocbench/v16/linux-rocm/transformers/` and commit those.)

---

## Self-Review (completed during planning)

**Spec coverage:**
- §4 standalone repo structure → Tasks 1–9 create `src/hunyuan_ocr/` + `scripts/` + `eval/` + `reports/`. `adapter/` (platform contract) is correctly deferred to Phase 4 (no task creates it). ✓
- §5 frozen contract → Task 2 (`contract.py`) + Task 3 (`tasks.py`) + Task 4 (`postprocess.py`). ✓
- §6 Phase 1 transformers (oracle) → Tasks 7–9 (backend, driver, run). ✓
- §6 canary + regression oracle (Absorb-C) → Task 9 Step 8 + `regression_canary.py`. Canary reuses the existing `OmniDocBench_150.json` subset. ✓
- §7 eval harness + gate → Task 6 (`scoring.py`) + Task 9 BASELINE record. ✓
- §8 per-page failure / resume / runaway guard → driver `try/except` per page (Task 8), resume via existing-`.md` skip (Task 8), tail-repetition early-stop + `clean_repeated_substrings` (Task 7). ✓
- §1.2 gate definition → Task 9 Step 10 report states ±0.3/±0.5. ✓

**Placeholder scan:** Report template (Task 9 Step 10) intentionally has `<fill>` fields for *measured* numbers produced at run time — these are runtime values, not plan placeholders; the code/tasks themselves have none. ✓

**Type consistency:** `iter_page_images` → `(stem, abs_path)` consumed identically by the driver (Task 8) and canary (Task 9). `infer_one(model, processor, image_path, prompt, device)` signature consistent across Task 7 (def) and Task 8 (call). `scoring.run_scorer/parse_run_summary/overall_score/write_eval_config` signatures consistent across Task 6 (def) and Tasks 8/9 (use). ✓

**Scope:** This plan delivers working, testable software (the transformers oracle + BASELINE) on its own. Phases 2 (vLLM) and 3 (llama.cpp) are explicitly separate plans. ✓
