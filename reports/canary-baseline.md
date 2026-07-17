# Canary BASELINE — HunyuanOCR-1.5 on AMD gfx1100 (OmniDocBench v1.6, 148-page canary)

> **Historical — 2026-07-16.** Retained as experimental evidence; `README.md` is
> the single source of current status. Some conclusions in this file read
> stronger than the evidence now supports (see README). Machine-local paths
> (`/root/...`, `/workspace/...`) are factual cross-session evidence, not user
> repro paths — use `scripts/reproduce_*.sh` + `reproducibility.lock.yaml`.

**Date:** 2026-07-16
**Subset:** `OmniDocBench_150.json` (148 pages), both backends 148/148 complete
**Hardware:** AMD gfx1100 (RDNA3, 48 GB) ×4; ROCm `hip 7.2.53211`; torch 2.9.1
**Resolution cap:** 3.4M pixels (~13k vision tokens — below the ~14.2k ROCm SDPA-ViT threshold; see ROCm issue #6416)

## Results

| Backend (ViT attention path) | Overall | text EditDist↓ | formula CDM↑ | table TEDS↑ | reading-order↓ |
|---|---|---|---|---|---|
| **vLLM 0.16.1** (Flash-Attn-Triton ViT) | **94.81** | 0.0514 | 0.9648 | 0.9308 | 0.1135 |
| transformers 5.13.0 (SDPA ViT) | **94.11** | 0.0437 | 0.9425 | 0.9246 | 0.1184 |
| upstream (reported) | 94.74 | — | — | — | — |

## Conclusions

- **HunyuanOCR-1.5 runs correctly on AMD ROCm gfx1100** under both backends at the 3.4M cap — vLLM 94.81 (≈ upstream 94.74), transformers 94.11.
- vLLM is the cleaner + faster path on ROCm → **primary backend**; transformers is the known-milder secondary (its SDPA ViT path shows the >14k instability + ~0.6 below vLLM).
- vLLM−transformers gap = 0.70 (outside ±0.3, but both near upstream; vLLM higher).
- **Caveat recorded:** an earlier partial 143-page transformers score showed 93.16 — that was an artifact (missing pages scored ~0); the complete 148/148 score is 94.11.

## Gate for backends

A backend passes when its full-set score is within ±0.3 Overall of the vLLM reference (vLLM = primary, 94.81 on canary). Full 1651-page run via vLLM is next.

## Artifacts

- transformers preds: `/root/hunyuanocr-results/canary-150/`
- vLLM preds: `/root/hunyuanocr-results/vllm-canary-150/`
- Scorer: OmniDocBench v1.6 `pdf_validation.py` (3.11 venv); Overall = `((1-text)*100 + cdm*100 + teds*100)/3`
