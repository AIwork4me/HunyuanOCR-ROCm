# HunyuanOCR-ROCm

A precision-aligned AMD ROCm port of Tencent's **HunyuanOCR-1.5** (~1B multimodal OCR model), benchmarked on **OmniDocBench v1.6** (full 1651-page set) on **gfx1100 (RDNA3)**.

**Status:** design phase. The full design is at
[`docs/superpowers/specs/2026-07-15-hunyuanocr-rocm-design.md`](docs/superpowers/specs/2026-07-15-hunyuanocr-rocm-design.md).

## Goal

Run HunyuanOCR-1.5 on AMD gfx1100 across three inference backends, precision-aligned with the original on OmniDocBench v1.6:

1. **transformers** (native) — the on-machine oracle / absolute baseline
2. **vLLM** (native `HunYuanVL`, ROCm serving)
3. **llama.cpp** (HIP on gfx1100) — the headline "runs on AMD GPU via llama.cpp" path

A frozen **decoding contract** is shared by all three backends, so the backend is the only
variable. The transformers phase's full-set score is the ground-truth baseline; vLLM and
llama.cpp must each match it within ±0.3 overall / ±0.5 per-task.

## Standalone, then integrated

This is an **independent, self-contained project**. It runs its own OmniDocBench evaluation
directly. Integration into the [OmniDocBench-AMD](https://github.com/AIwork4me/OmniDocBench-AMD)
platform (adapter contract, registry, conformance, badge) is a **final step (Phase 4)**,
performed only after adaptation + evaluation succeed.

## Weights & license

Model weights are `tencent/HunyuanOCR` (Hugging Face), under the **Tencent Hunyuan Community
License** (source-available, not OSI-open). Derivatives must carry the **"Powered by Tencent
Hunyuan"** notice. Formal licensing posture for this project is resolved after evaluation
(per the design spec, §1.1).
