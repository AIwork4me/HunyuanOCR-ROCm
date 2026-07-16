**Update 2 — Complete three-backend evaluation on AMD ROCm gfx1100: llama.cpp full-set (1651 pages) + canary comparison across all backends**

We've now completed a full three-backend evaluation of HunyuanOCR-1.5 on AMD gfx1100 (RDNA3, ROCm 7.2, bf16). Here are the complete results.

## Canary (148 pages) — all three backends, same weights

| Backend | ViT attention path | Overall | text EditDist ↓ | formula CDM ↑ | table TEDS ↑ | resolution |
|---|---|---|---|---|---|---|
| **vLLM 0.16.1** | Flash-Attn (Triton) | **94.81** | 0.0514 | 0.9648 | 0.9308 | capped 3.4M |
| transformers 5.13.0 | SDPA | 94.11 | 0.0437 | 0.9425 | 0.9246 | capped 3.4M |
| **llama.cpp** (ggml-org BF16 GGUF) | C++ GGML | 93.33 | 0.0512 | 0.9083 | 0.9429 | **uncapped (full res)** |

## Full set (1651 pages) — llama.cpp, BF16, uncapped

| Metric | Value |
|---|---|
| **Overall** | **92.09** |
| text EditDist | 0.0467 (95.3%) |
| formula CDM | 0.8964 (89.6%) |
| table TEDS | 0.9130 (91.3%) |
| reading-order EditDist | 0.1375 |

1651/1651 pages, 0 errors. llama-server built from `ggml-org/llama.cpp` master with HIP (`-DGGML_HIP=ON -DGPU_TARGETS=gfx1100`), BF16 GGUF from `ggml-org/HunyuanOCR-GGUF`, `-c 65536 -ngl 999`.

## Key findings

**1. llama.cpp's C++ ViT is stable at full resolution — no >14k instability.**
Running the same determinism check (3× identical forward on a ~15k vision-token page) that fails on the transformers SDPA path (NaN + non-deterministic above ~14.2k tokens), **llama.cpp's C++ GGML ViT produces identical output across runs — fully deterministic, no NaN, no resolution cap needed.** This is the third independent code path (after vLLM's Flash-Attn) that avoids the issue, reinforcing that the instability is specific to the PyTorch/ROCm SDPA ViT kernel path.

**2. The accuracy gap is concentrated in formula CDM.**
llama.cpp's text accuracy (95.3%) and table TEDS (91.3%) are competitive with the other backends. The gap to upstream (~2.65 Overall) is driven primarily by **formula CDM (89.6% vs vLLM's 96.5%)** — consistent with your "accuracy not yet aligned" note. The likely cause is a subtle preprocessing difference between llama.cpp's C++ image processor and the HF `HunYuanVLImageProcessor` affecting fine-grained LaTeX formula features. Text and table (which are less sensitive to pixel-level precision) are much closer to parity.

**3. llama.cpp is the only backend that runs uncapped on ROCm.**
Both vLLM and transformers require a 3.4M-pixel cap (to stay under the ~14.2k ViT threshold). llama.cpp runs at **full native resolution** with no cap — and is also the **fastest** (~1.4 s/page for a warm single request vs ~6 s vLLM compiled / ~180 s transformers) and **most stable** (single server, deterministic, no multi-server management issues).

## What would help us close the formula CDM gap

If the team has any guidance on:
- **Image preprocessing parity**: is there a reference for the exact patch-n-pack / normalization / tiling the `HunYuanVLImageProcessor` applies, so we can diff it against llama.cpp's C++ preprocessor?
- **Known good llama.cpp settings**: has anyone internally tested HunyuanOCR with llama.cpp at specific settings that achieve closer parity?

we'd be happy to investigate further. Meanwhile, the full per-page outputs and scoring artifacts are available on request.

**Environment:** AMD gfx1100 (RDNA3, 48 GB) ×4; ROCm `hip 7.2.53211`; torch 2.9.1; llama.cpp master (ggml 0.16.0, commit a320cbf); BF16 GGUF from `ggml-org/HunyuanOCR-GGUF`; OmniDocBench v1.6 (1651 pages, greedy decode, `doc_parse` task).

Thanks again! 🙏
