# HunyuanOCR-ROCm — Project Stage Summary

**As of:** 2026-07-16 · **Branch:** `feat/phase1-transformers` (off `main`) · **Repo:** `/workspace/HunyuanOCR-ROCm`

## Goal (recap)

A standalone, eval-backed project running Tencent **HunyuanOCR-1.5** (~1B VLM) on **AMD gfx1100 (RDNA3, ROCm)**, precision-aligned with the original on **OmniDocBench v1.6**, across three backends: transformers → vLLM → llama.cpp. Standalone first; integration into OmniDocBench-AMD is Phase 4 (post-eval).

## Status at a glance

- ✅ **Phase-1 code** (transformers backend): 9 tasks, all reviewed clean, ~14 commits.
- ✅ **Phase-2 code** (vLLM backend): adapter + serve + driver + plan.
- ✅ **Canary BASELINE** (OmniDocBench_150 = 148 pages, **both backends 148/148 complete**).
- ✅ **Two upstream issues filed** (ROCm, Tencent) — both OPEN with evidence.
- 🔄 **Full 1651-page vLLM run in progress** (~5h, background; auto-scores on completion).

## Canary results (the headline)

Same 148-page OmniDocBench v1.6 subset, same weights, same 3.4M-pixel cap, gfx1100:

| Backend (ViT attention) | Overall | text EditDist | formula CDM | table TEDS |
|---|---|---|---|---|
| **vLLM 0.16.1** (Flash-Attn-Triton ViT) | **94.81** | 0.0514 | 0.9648 | 0.9308 |
| transformers 5.13.0 (SDPA ViT) | 94.11 | 0.0437 | 0.9425 | 0.9246 |
| upstream (reported) | 94.74 | — | — | — |

**Conclusion:** HunyuanOCR-1.5 runs correctly on AMD ROCm gfx1100 — vLLM reaches 94.81 (≈ upstream 94.74), transformers 94.11. **Precision-alignment with the original is essentially achieved (via vLLM).**

## Key technical findings

1. **A real ROCm ViT instability (root-caused + worked around).** The Hunyuan-ViT forward becomes **non-deterministic + NaN above a sharp ~14.2k–14.7k vision-token threshold** — in the **transformers SDPA/eager ViT path** only (vLLM's Flash-Attention ViT avoids it). Sharp threshold; isolated to the full ViT forward (single ops don't reproduce); LLM unaffected; fp32 doesn't NaN. **Workaround:** cap image to 3.4M pixels (~13k tokens) → deterministic + correct. Filed with AMD.
2. **vLLM is the better/faster backend on ROCm.** It reaches upstream (94.81) where transformers (94.11) is slightly lower — consistent with the SDPA-ViT degradation. And it's the only path fast enough for the full set.
3. **A `max-model-len` footgun:** the contract's `max_tokens=32768` requires `--max-model-len ≥ ~49k`; setting it to 32768 silently 400-rejects every request.
4. **Throughput:** transformers-native ~5.5 tok/s (full set ~40h, impractical). vLLM eager decode ~2 tok/s (slow, unfused kernels). vLLM **torch.compile** fuses decode kernels → ~28× single-request speedup (compiles in ~140s, no stall with the capped dir). Batched throughput settles around ~5–30 pages/min depending on warmup.

## Issues filed (both OPEN, by AIwork4me)

- **AMD ROCm #6416** — the bf16 ViT forward non-determinism + NaN above ~14.3k tokens on gfx1100; full repro + threshold table + isolation. https://github.com/ROCm/ROCm/issues/6416
- **Tencent HunyuanOCR #114** — asks the team for the intended vision-token budget / CUDA determinism check / recommended cap; reports the two-backend canary data localizing the issue to the transformers/ROCm ViT path; points to free AMD GPUs (Radeon Cloud). https://github.com/Tencent-Hunyuan/HunyuanOCR/issues/114

## gfx1100 adaptations (in code)

- `backends/transformers.py`: ViT pixel cap (`GFX1100_VIT_MAX_PIXELS=3.4M`, env-overridable) + sdpa attention; ROCm-issue #6416 referenced.
- `scripts/serve_vllm.sh`: capped model dir, `max-model-len 65536`, `torch.compile` by default (enforce-eager fallback).
- Frozen decoding contract (prompt/sampling/post-processors) shared across backends.

## In progress

- **Full 1651-page OmniDocBench v1.6 via vLLM** (4 compiled servers, one/GPU) → background; auto-scores on completion. Expected: Overall ≈ 94.x (confirming the canary at full scale).

## Next steps (after the full number lands)

1. Record the full-set vLLM number + post a follow-up to Tencent #114.
2. **Phase 3 — llama.cpp** (HIP on gfx1100): it has its own C++ ViT (different from both transformers and vLLM) — the >14k bug may not apply; and it's the user's headline goal ("runs on AMD via llama.cpp"). Official ggml-org GGUF + `hunyuan-vl` arch already exist.
3. Resume the transformers full-set (optional; it's the slow oracle) or accept canary as the transformers reference.
4. **Phase 4** — integrate into OmniDocBench-AMD (adapter contract + registry + badge) once the three backends are aligned.

## Risks / open

- Full-set throughput is adequate (~hours) but not maximally efficient (client↔server interaction on this vLLM build caps batched throughput); an async driver could speed it ~6× if needed later.
- ROCm-vs-CUDA attribution of the ViT instability is unconfirmed (no NVIDIA box) — that's exactly what Tencent #114 asks.
