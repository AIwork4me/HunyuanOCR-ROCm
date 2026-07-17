# Benchmark methodology

> How every formal number in this repo is produced, scored, and what it is and is
> **not** comparable to. Read this before quoting any result.

This project is an **evaluation-backed** AMD ROCm port — not a *precision-aligned*
port. That distinction is the whole point of this document.

## 1. What "precision-aligned" would require

We do **not** claim precision alignment against the upstream CUDA implementation.
Such a claim would require, on a **single identical page set**, all of:

1. The **same page set** (same GT, same image files).
2. The **same ground truth** and the **same prompt**.
3. The **same post-processing** (`clean_repeated_substrings` + `process_one`, the
   frozen `src/hunyuan_ocr/contract.py`).
4. The **same resolution policy** (uncapped, or the same pixel cap).
5. The **same scorer** at the **same commit** (OmniDocBench, pinned in
   `reproducibility.lock.yaml`).
6. The **same metric config** (OmniDocBench eval YAML).
7. A **pre-declared tolerance**, e.g. Overall ±0.3.

The missing piece is a **CUDA control run on the same 148-page canary**. Without
it, this project's numbers and the upstream's numbers are *evaluations of
different runs on (possibly) different page sets*, not a precision comparison:

> **Precision alignment against the upstream CUDA implementation has not yet been
> established on the same page set.**

Two further, independent reasons a same-number comparison is not apples-to-apples:

- The **official HunyuanOCR OmniDocBench headline (94.10) is measured with
  TensorRT**, not vLLM/transformers/llama.cpp. Tencent's own README states these
  metrics "may slightly differ from the inference methods using Transformers or
  vLLM." (Source: [official repo](https://github.com/Tencent-Hunyuan/HunyuanOCR)
  benchmark table.)
- The **official table does not label its OmniDocBench dataset version**; this
  project evaluates on **OmniDocBench v1.6**. A version mismatch cannot be ruled
  out, so even the 94.10 reference is approximate.

## 2. The two result tables (never mixed)

This repo reports on **two different page sets**. They are never placed in the
same column, because their numbers are not comparable.

- **Canary (148 pages)** — `OmniDocBench_150.json`, a project-defined subset. All
  three backends ran the *same* 148 pages. The upstream CUDA backend was **not**
  evaluated on these 148 pages. (`eval/canary_148.manifest.json` lists the pages
  and the source-GT SHA256; `scripts/create_canary_manifest.py` regenerates it;
  `hunyuan-ocr canary materialize` rebuilds the subset from the full GT.)
- **Full (1651 pages)** — the complete OmniDocBench v1.6 set. Only the llama.cpp
  backend completed a valid full-set run. The vLLM full-set run is **invalid**
  (server crashes → ~780 ERROR pages → a false 46.31 that is excluded).

### Table A — same 148-page canary, three ROCm backends

| Backend | Overall | text EditDist↓ | formula CDM↑ | table TEDS↑ | order EditDist↓ | Resolution | Status |
|---|---|---|---|---|---|---|---|
| vLLM 0.16.1 (Flash-Attn ViT) | **94.81** | 0.0514 | 0.9648 | 0.9308 | 0.1135 | capped 3.4M | 148/148 complete |
| transformers 5.13.0 (SDPA ViT) | 94.11 | 0.0437 | 0.9425 | 0.9246 | 0.1184 | capped 3.4M | 148/148 complete |
| llama.cpp (C++ GGML, BF16 GGUF) | 93.33 | 0.0512 | 0.9083 | 0.9429 | 0.1270 | uncapped | 148/148 complete |
| upstream CUDA | _Not evaluated on this canary_ | — | — | — | — | — | — |

Source: `reports/canary-baseline.md` + `reports/HANDOFF.md` (canary-148 section).
Sub-metrics are raw scorer outputs (CDM/TEDS in 0–1; multiply by 100 for a
percentage). Overall = `((1−text)·100 + CDM·100 + TEDS·100) / 3`
(`src/hunyuan_ocr/scoring.py::overall_score`).

### Table B — full 1651-page set (OmniDocBench v1.6)

| Backend / source | Overall | text EditDist↓ | formula CDM↑ | table TEDS↑ | order EditDist↓ | Inference engine | Notes |
|---|---|---|---|---|---|---|---|
| **llama.cpp** (this repo, gfx1100/ROCm) | **92.09** | 0.0467 | 0.8964 | 0.9130 | 0.1375 | llama.cpp C++ GGML (HIP) | 1651/1651, 0 errors; verified twice |
| Official HunyuanOCR (OmniDocBench) | **94.10** | 0.042 | 0.9473 | 0.9181 | _not reported_ | **TensorRT** | [official repo](https://github.com/Tencent-Hunyuan/HunyuanOCR) benchmark table |
| Δ (ours − official) | **−2.01** | +0.0047 | −0.0509 | −0.0051 | — | different engine | not a precision comparison |

> A **94.74** figure circulates in third-party summaries attributed to
> "OmniDocBench v1.6"; it is **not present in the official GitHub benchmark
> table** and could not be verified against the technical report
> ([arXiv:2511.19575](https://arxiv.org/abs/2511.19575)). It is therefore
> `not_verified` and is not used as the official anchor. The 94.10 figure above
> is taken directly from the official repository's published table.

The **−2.01 Overall gap** is **not** attributable to ROCm vs CUDA: the official
number uses a third inference engine (TensorRT) and an unlabeled dataset version,
while ours uses llama.cpp on v1.6. Treat it as "how this port's full-set run
compares to a published reference", not as a measured ROCm-vs-CUDA delta.

## 3. Scoring pipeline (fully reproducible)

1. **Predict** — `run_phase2_vllm.py` (OpenAI-compatible) or
   `run_phase1_transformers.py`, writing one `<stem>.md` per page atomically.
2. **Validate** — `hunyuan-ocr validate` / `scripts/validate_predictions.py`
   blocks scoring on any missing / empty / `ERROR:` / `.partial` / unresolved
   `_errors/<stem>.json` page (strict mode: warnings are also fatal).
3. **Score** — `hunyuan-ocr score` / `scripts/score_predictions.py` materializes
   an OmniDocBench eval config **into a private temp dir** (never polluting the
   prediction directory), runs `pdf_validation.py` in the pinned OmniDocBench
   venv, and parses `run_summary.json`.

`Overall = ((1 − text_EditDist)·100 + formula_CDM·100 + table_TEDS·100) / 3`.
`reading_order_EditDist` is reported **separately** and is **not** part of
Overall. When a subset has no display-formula pages, CDM is `null` and Overall is
undefined (reported `n/a`).

## 4. Performance numbers (and what they are not)

Throughput in this repo is **diagnostic**, not a ranked benchmark. We deliberately
do **not** collapse "speed" into one column, because the measurements mix:

- single-request latency vs. multi-server aggregate throughput,
- warm vs. cold,
- wall-clock over a fixed page set vs. tokens/s.

Recorded observations (gfx1100, ROCm 7.2, bf16) — treat as rough, not comparable
across backends without restating the measurement protocol:

| Metric | vLLM (compiled) | transformers (SDPA) | llama.cpp (HIP) |
|---|---|---|---|
| warm single-page latency | ~6 s/page | ~180 s/page | ~1.4 s/page |
| decode rate | ~150 tok/s/server | ~5.5 tok/s | ~22 tok/s/slot (×4 slots) |
| full-set feasibility | servers crash under sustained load | ~40 h, impractical | ~hours, stable |

These are **diagnostic**: latency is single-request warm; decode rate is
per-server/per-slot; no p50/p95 were recorded from raw timing data (do **not**
invent percentiles). Any cross-backend ranking from this table is invalid unless
the measurement protocol is identical.

## 5. Provenance of each formal number

| Number | Source | Reproducible via |
|---|---|---|
| vLLM canary 94.81 (and sub-metrics) | `reports/canary-baseline.md`, prediction dir | `scripts/reproduce_llamacpp_canary.sh` (swap server to vLLM) |
| transformers canary 94.11 | `reports/canary-baseline.md`, prediction dir | `make eval-canary` (transformers) |
| llama.cpp canary 93.33 | `reports/HANDOFF.md` §3.1 | `scripts/reproduce_llamacpp_canary.sh` |
| llama.cpp full 92.09 | `reports/HANDOFF.md` §3.2 (scored twice) | `scripts/reproduce_llamacpp_full.sh` |
| Official 94.10 / 0.042 / 0.9473 / 0.9181 | official GitHub README benchmark table | [link](https://github.com/Tencent-Hunyuan/HunyuanOCR) |
| 94.74 | third-party summaries only | **not_verified** — not used as anchor |

Everything else (the >14k ViT isolation, throughput tuning, formula-CDM ablation)
is **diagnostic** and labeled as such in `reports/`.
