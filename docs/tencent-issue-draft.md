# Recommended max image resolution / vision-token budget for HunyuanOCR-1.5 stable inference? (high-resolution ViT instability observed on ROCm)

Hi HunyuanOCR team 👋 — thanks for open-sourcing HunyuanOCR-1.5. While running it on AMD ROCm for OmniDocBench v1.6, we noticed a sharp resolution-dependent instability in the vision tower. We're not sure whether it's specific to our ROCm stack or relevant to the model's intended vision-token budget, and we'd appreciate your guidance on the intended operating range.

## TL;DR

- On AMD ROCm, HunyuanOCR-1.5's vision tower exhibits **bf16 instability at high vision-token counts**.
- **< 14.2k vision tokens:** deterministic (`max|Δ| = 0` across identical runs), correct OCR.
- **~14.7k vision tokens:** non-deterministic output (different results across identical greedy runs), intermittent `NaN`.
- **Capping image pixel area so tokens stay < ~13k restores stability and correct output.**

We cannot yet determine whether this originates from a **ROCm kernel issue**, the **model's resolution envelope**, or **bf16 long-sequence numerics**. To help localize it, we'd value confirmation of three things:

1. Was **OmniDocBench v1.6 evaluated at full native resolution**?
2. Is that **resolution range deterministic on CUDA**?
3. What is the **recommended maximum vision-token budget**?

## Questions for the HunyuanOCR team

**Q1 — Was OmniDocBench v1.6 (94.74) evaluated at full native resolution?**
Were pages processed at full native resolution (up to ~2k×2.4k, i.e. ~16k vision tokens), or was any pixel/patch cap applied? This would directly tell us the intended operating range.

**Q2 — Is full-resolution inference deterministic on CUDA?**
If possible, could you help confirm whether this resolution range (longest side ~2300 px, ~14.7k vision tokens) is deterministic and correct on CUDA — e.g. identical greedy output across repeated runs? We can't compare to CUDA locally, so this would be the single most useful data point for attribution.

**Q3 — Is there a recommended maximum vision-token count?**
The processor default `size.longest_edge = 16,777,216` (≈ 4096² pixel area, i.e. up to ~16k vision tokens for typical pages) sits right at/above where we observe trouble on ROCm. Is this within the model's expected safe range, or is a lower cap intended for stable inference?

## Workaround that works for us

Capping the image-processor pixel area so patches stay under the threshold restores determinism and correctness:

```python
processor.image_processor.size.longest_edge = 3_400_000  # -> ~13k patches, under the threshold
```

On a 30-page smoke subset, the capped configuration achieves **text EditDist 0.0029 (99.7%)** and **table TEDS 0.985**. (A 150-page run is in progress; we can share full numbers.) This suggests the model behaves correctly when kept under the threshold — we'd like to confirm whether such a cap reflects the intended usage or whether full resolution is expected to be safe end-to-end.

## Detailed observation (on AMD ROCm gfx1100)

**Environment:** AMD gfx1100 (RDNA3, 48 GB); ROCm `hip 7.2.53211`; torch `2.9.1` (ROCm build); transformers `5.13.0`; model `tencent/HunyuanOCR`; bf16, SDPA attention.

**Threshold** (vision-tower forward `model.model.get_image_features(...)`, 3 identical runs):

| Vision patches (seq len) | Image longest side | 3-run `max\|Δ\|` | NaN | Output |
|---|---|---|---|---|
| ≤ ~14,200 | ≲ ~2270 px | **0.0** (deterministic) | No | correct OCR |
| ~14,688 | ≳ ~2300 px | **up to ~9312** | intermittent | degenerate |

At ~14.7k patches, end-to-end greedy (`do_sample=False`) on the same image gives different outputs across identical runs — e.g. `"I"`, `"(1,0),(1000,999)"`, and a coherent-but-wrong paragraph.

**Isolated to the full ViT forward:** the LLM text-only forward is deterministic at 4k tokens; a bare bf16 matmul, standalone `scaled_dot_product_attention`, `nn.LayerNorm`, and a single ViT block (random input) are all deterministic at 14,688 — so the instability only emerges in the full 27-layer ViT forward with the model's real weights/activations. A single **fp32** forward at full resolution does **not** NaN, suggesting the bf16 long-sequence path is involved.

We filed the non-determinism with AMD for completeness: https://github.com/ROCm/ROCm/issues/6416 — but again, we cannot yet tell whether the root cause is a ROCm kernel issue, the model's resolution envelope, or bf16 long-sequence numerics, which is why the CUDA check (Q2) would be so helpful.

## Reproducing (free AMD GPU available)

If anyone would like to reproduce on AMD hardware, **free ROCm GPU instances are available on Radeon Cloud: https://radeon.anruicloud.com/** — happy to help set up if useful.

Minimal check (any ROCm box with the model downloaded):

```python
import torch
from transformers import AutoProcessor, HunYuanVLForConditionalGeneration

MODEL = "tencent/HunyuanOCR"
proc = AutoProcessor.from_pretrained(MODEL, use_fast=False)
model = (
    HunYuanVLForConditionalGeneration.from_pretrained(MODEL, attn_implementation="sdpa", dtype=torch.bfloat16)
    .to("cuda:0")
    .eval()
)  # ROCm PyTorch also uses the "cuda" device naming
from PIL import Image

img = Image.new("RGB", (2400, 2400))  # any image; ~2400px longest side -> ~14.7k patches (over threshold)
inp = proc(images=img, return_tensors="pt").to("cuda:0")
feats = [
    model.model.get_image_features(inp["pixel_values"], inp["image_grid_thw"], return_dict=True)["last_hidden_state"]
    for _ in range(3)
]
print(
    "max|Δ| across 3 runs:",
    max(torch.abs(feats[0] - feats[1]).max().item(), torch.abs(feats[0] - feats[2]).max().item()),
    "NaN:",
    any(torch.isnan(f).any().item() for f in feats),
)
# Under threshold (cap): proc.image_processor.size.longest_edge = 3_400_000  ->  max|Δ| == 0, no NaN
```

## Additional ROCm deployment question

Separately from the stability question above: your `inference/` docs validate NVIDIA/CUDA only — are there any plans or known-good settings for AMD ROCm? And is **vLLM** the recommended serving path for throughput? (transformers-native inference runs at ~5–6 tok/s on our hardware.)

---

Thanks very much for any guidance on the intended vision-token budget — happy to provide more isolation data or a narrower reproduction on request.
