**Update — two-backend results on the OmniDocBench v1.6 canary (148 pages): the model is healthy on AMD ROCm at the capped resolution; the high-res instability is an extreme-resolution edge case**

We completed the canary (the `OmniDocBench_150.json` subset = 148 pages) under **two backends on the same AMD gfx1100 box, identical weights, identical 3.4M-pixel resolution cap** (the workaround from the original post). Both finished 148/148:

| Backend (ViT attention path) | Overall | text EditDist | formula CDM | table TEDS |
|---|---|---|---|---|
| **vLLM 0.16.1** — Flash-Attention (Triton) ViT | **94.81** | 0.0514 | 0.9648 | 0.9308 |
| transformers 5.13.0 — SDPA ViT | 94.11 | 0.0437 | 0.9425 | 0.9246 |
| upstream (reported) | 94.74 | — | — | — |

Takeaways:

1. **HunyuanOCR-1.5 runs correctly on AMD ROCm gfx1100** under both backends at the 3.4M-pixel cap — vLLM reaches **94.81** (≈ your reported 94.74), transformers **94.11**. So the model itself is healthy at this resolution; the cap costs little.
2. The two backends agree to within **0.70 Overall** on the identical model/cap/hardware. vLLM's Flash-Attention ViT path lands marginally higher (and slightly above upstream); transformers' SDPA path lands ~0.6 below. This is consistent with the high-resolution instability we reported being concentrated in the SDPA/eager ViT path on ROCm and only manifesting at extreme (>~14k) vision-token counts — at the capped resolution both paths are stable and accurate.
3. We cap at 3.4M pixels (~13k vision tokens) specifically to stay below the ~14.2k threshold where the SDPA ViT path becomes non-deterministic/NaN on ROCm.

So our open questions from the original post (Q1–Q3) still stand, but the picture is reassuring: the model evaluates cleanly on ROCm as long as the vision-token count is kept under the threshold. A CUDA determinism check at full resolution (Q2) would still help us confirm whether the >14k instability is ROCm-specific.

**Environment:** AMD gfx1100 (RDNA3, 48 GB); ROCm `hip 7.2.53211`; torch 2.9.1; transformers 5.13.0 / vLLM 0.16.1; bf16; greedy decode; 3.4M-pixel cap; 148-page OmniDocBench v1.6 canary. (Full 1651-page run via vLLM in progress; happy to share final numbers + per-page outputs / scoring artifacts.)
