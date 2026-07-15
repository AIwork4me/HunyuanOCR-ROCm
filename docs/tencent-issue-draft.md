# HunyuanOCR-1.5 vision-tower numerical instability (non-deterministic output + NaN) at high resolution (>~14k vision tokens) — confirmation + recommended max resolution?

## Summary

When running HunyuanOCR-1.5 (transformers 5.13.0, `HunYuanVLForConditionalGeneration` + `AutoProcessor`) on **AMD ROCm (gfx1100)**, the vision tower (`Hunyuan-ViT`) forward becomes **non-deterministic and produces `NaN`** once the per-image vision-token/patch count exceeds a sharp threshold of **~14,200–14,688 tokens**. Below the threshold, the same forward is bit-for-bit deterministic and the model produces correct OCR. With the default processor setting (`max_pixels`/`size.longest_edge = 16,777,216`), a full-resolution document page (~1653×2339 → 15,184 patches) lands above the threshold and yields degenerate output.

We're filing this to (a) check whether this is a **known characteristic of the model at high resolution** versus a backend issue, and (b) ask for the **recommended maximum resolution / vision-token count for stable inference**.

## Environment

- GPU: AMD Radeon gfx1100 (RDNA3), 48 GB
- ROCm: `hip 7.2.53211`; torch `2.9.1+gitff65f5b` (ROCm/HIP build)
- transformers `5.13.0` (native `hunyuan_vl`); model `tencent/HunyuanOCR`
- Linux x86_64, Python 3.12

(Your `inference/transformers/requirements.txt` validates only NVIDIA/CUDA; we're running on AMD ROCm, hence the question about expected behaviour.)

## Observations

1. **Non-determinism + NaN above ~14k tokens.** Running `model.model.get_image_features(...)` 3× on the **same** image, bf16, SDPA attention:
   - ≤ ~14,200 patches → `max|Δ| = 0.0` across runs (deterministic), no NaN, correct OCR.
   - ~14,688 patches (image longest side ≥ ~2300 px) → `max|Δ|` up to ~9312 across identical runs, and intermittent `NaN`.
2. **End-to-end greedy** (`do_sample=False`) on a full-resolution page gives three different outputs on three identical runs, e.g. `"I"`, `"(1,0),(1000,999)"`, and a coherent-but-wrong paragraph.
3. **Isolated to the vision tower forward.** The LLM text-only forward is deterministic at 4k tokens; a bare bf16 matmul, standalone `scaled_dot_product_attention`, `nn.LayerNorm`, and a single ViT block (random input) are all deterministic at 14,688 — so the instability only emerges in the full 27-layer ViT forward with the model's real weights/activations.
4. A single **fp32** forward at full resolution does **not** NaN (stays finite), suggesting the failure is specific to the bf16 long-sequence path.

## What we did (workaround)

Capping the image-processor pixel area so the patch count stays below the threshold restores determinism and correctness:

```python
processor.image_processor.size.longest_edge = 3_400_000   # -> ~13k patches, under the threshold
```

With this cap, a 30-page smoke subset scores **text EditDist 0.0029 (99.7%)** and **table TEDS 0.985** — i.e. the model works correctly when kept under the threshold. (A 150-page run is in progress; happy to share full numbers.)

We also filed the non-determinism with AMD: https://github.com/ROCm/ROCm/issues/6416 — but we could **not** compare against NVIDIA/CUDA locally, so we can't yet tell whether the instability is ROCm-specific or also present (just unnoticed) on CUDA at this resolution.

## Questions for the team

1. **Attribution:** Is numerical instability at high vision-token count (>~14k) a known property of the Hunyuan-ViT, or is it expected to be stable at that scale on CUDA (i.e. likely ROCm-specific)? Have you observed anything similar internally at high resolution?
2. **Recommended cap:** Is there a recommended maximum image resolution / vision-token count for stable inference? The default `max_pixels`/`size.longest_edge` (16,777,216 ≈ 4096², i.e. up to ~16k patches for typical pages) appears to sit right at/above where we see instability — is full 2k–4k resolution expected to be numerically safe end-to-end?
3. **ROCm:** Any guidance or planned support for AMD ROCm (the inference docs currently list NVIDIA only)? Is `vLLM` the recommended serving path for throughput (transformers-native is quite slow on our hardware)?

Happy to provide a minimal repro script and the full per-resolution threshold table on request. Thanks!
