**Update — HunyuanOCR-1.5 evaluates cleanly on AMD ROCm at the capped resolution; two-backend canary results + a localization datapoint for the high-res instability**

We finished the OmniDocBench v1.6 canary (`OmniDocBench_150.json` = 148 pages) under **two backends on the same AMD gfx1100 box, identical weights, identical 3.4M-pixel cap**, both **148/148 complete**:

| Backend (ViT attention path) | Overall | text EditDist ↓ | formula CDM ↑ | table TEDS ↑ | reading-order EditDist ↓ |
|---|---|---|---|---|---|
| **vLLM 0.16.1** — Flash-Attention (Triton) ViT | **94.81** | 0.0514 | 0.9648 | 0.9308 | 0.1135 |
| transformers 5.13.0 — SDPA ViT | **94.11** | 0.0437 | 0.9425 | 0.9246 | 0.1184 |
| upstream (reported) | 94.74 | — | — | — | — |

**What this tells us**

1. **The model itself is healthy on ROCm at this resolution.** vLLM reaches **94.81** (≈ your reported 94.74); transformers reaches **94.11**. So the 3.4M-pixel cap costs very little, and HunyuanOCR-1.5 produces correct, high-quality OCR on AMD gfx1100 when the vision-token count is kept under the threshold.
2. **The two backends agree to within 0.70 Overall** on identical model/cap/hardware — vLLM's Flash-Attention ViT path lands marginally higher (and slightly above upstream). This is consistent with the high-resolution instability we reported (original post) being concentrated in the **SDPA/eager ViT path on ROCm** and only surfacing at **extreme (>~14k) vision-token counts**; at the capped resolution both paths are stable and accurate.
3. We cap at **3.4M pixels (~13k vision tokens)** precisely to stay below the ~14.2k threshold where the SDPA ViT path becomes non-deterministic + NaN on our ROCm stack (filed with AMD: [#6416](https://github.com/ROCm/ROCm/issues/6416)).

So the picture is reassuring rather than alarming: this looks like an **extreme-resolution edge case in one attention path on ROCm**, not a model defect — but we still can't rule out a model-side contribution without a CUDA comparison, which is why **Q2 from the original post (is full-resolution deterministic on CUDA?)** remains the single most useful datapoint for us.

**Environment:** AMD gfx1100 (RDNA3, 48 GB); ROCm `hip 7.2.53211`; torch 2.9.1; transformers 5.13.0 / vLLM 0.16.1; bf16; greedy decode; 3.4M-pixel cap; 148-page OmniDocBench v1.6 canary. Full 1651-page run via vLLM is in progress — happy to share final numbers, per-page outputs, or the scoring artifacts on request.

Thanks again for any guidance, and for open-sourcing the model! 🙏
