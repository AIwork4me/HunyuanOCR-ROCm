# Recommended max image resolution / vision-token count for stable inference? (high-resolution instability observed on AMD ROCm)

Hi! We're running **HunyuanOCR-1.5** (transformers 5.13.0, `HunYuanVLForConditionalGeneration` + `AutoProcessor`) on **AMD ROCm (gfx1100)** to evaluate on OmniDocBench v1.6, and we hit a sharp high-resolution instability in the vision tower. We can't tell from here whether it's specific to our ROCm stack or also relevant to the model's intended resolution envelope — so we'd value the team's guidance. Four quick questions below; the ROCm observation is provided as context.

## Questions for the team

1. **What image resolution did you use for the reported OmniDocBench v1.6 numbers (94.74)?** Did you process pages at full native resolution (up to ~16k vision tokens for a ~2k×2.4k page) or apply a pixel/patch cap? (This would directly tell us the intended operating range.)
2. **Can you confirm full-resolution inference is deterministic + correct on your CUDA setup?** i.e. for a page whose longest side is ~2300 px (~14.7k vision tokens), is greedy output identical across repeated runs? (A 30-second check for you; we can't compare to CUDA locally.)
3. **Is there a recommended maximum vision-token count for numerically stable inference?** The default processor setting (`size.longest_edge` = 16,777,216 ≈ 4096², i.e. up to ~16k patches for typical pages) sits right at/above where we see trouble on ROCm — is full 2k–4k resolution expected to be safe end-to-end?
4. **ROCm guidance?** Your `inference/` docs validate NVIDIA/CUDA only — are there plans or known-good settings for AMD ROCm? And is **vLLM** the recommended serving path for throughput? (transformers-native inference is quite slow on our hardware, ~5–6 tok/s.)

## What we observe (on AMD ROCm gfx1100) — context

- Env: AMD gfx1100 (RDNA3, 48 GB); ROCm `hip 7.2.53211`; torch `2.9.1` (ROCm build); transformers `5.13.0`; model `tencent/HunyuanOCR`; bf16, SDPA attention.
- The vision-tower forward (`model.model.get_image_features(...)`) becomes **non-deterministic and emits `NaN`** once the per-image patch count exceeds a sharp threshold of **~14,200–14,688 tokens**. Below it, the same forward is bit-for-bit deterministic (`max|Δ| = 0`) and produces correct OCR.
  - ≤ ~14,200 patches (image longest side ≲ ~2270 px) → deterministic, correct.
  - ~14,688 patches (longest side ≳ ~2300 px) → `max|Δ|` up to ~9312 across 3 identical runs, intermittent NaN; end-to-end greedy gives different outputs on identical runs (e.g. `"I"`, `"(1,0),(1000,999)"`, a coherent-but-wrong paragraph).
- **Isolated to the full ViT forward:** the LLM text-only forward is deterministic at 4k tokens; a bare bf16 matmul, standalone `scaled_dot_product_attention`, `nn.LayerNorm`, and a single ViT block (random input) are all deterministic at 14,688 — so the instability only emerges in the full 27-layer ViT forward with the model's real weights/activations. A single **fp32** forward at full res does **not** NaN, suggesting it's specific to the bf16 long-sequence path.
- We filed the non-determinism with AMD: https://github.com/ROCm/ROCm/issues/6416 — but couldn't compare against CUDA locally, which is why question #2 matters to us.

## Workaround that works for us

Capping the image-processor pixel area so patches stay under the threshold restores determinism and correctness:

```python
processor.image_processor.size.longest_edge = 3_400_000   # -> ~13k patches, under the threshold
```

With this cap, a 30-page smoke subset scores **text EditDist 0.0029 (99.7%)**, **table TEDS 0.985**. (A 150-page run is in progress; we can share full numbers.) So the model is clearly correct when kept under the threshold — we just want to confirm whether a cap is the *intended* usage or whether full-res is expected to be safe (i.e. the trouble is ours/ROCm's to fix).

## Reproducing (free AMD GPU available)

If anyone on the team would like to reproduce on AMD hardware, **free ROCm GPU instances are available on Radeon Cloud: https://radeon.anruicloud.com/** — happy to help set up if useful.

Minimal check (any ROCm box with the model downloaded):

```python
import torch
from transformers import AutoProcessor, HunYuanVLForConditionalGeneration
MODEL = "tencent/HunyuanOCR"
proc = AutoProcessor.from_pretrained(MODEL, use_fast=False)
model = HunYuanVLForConditionalGeneration.from_pretrained(MODEL, attn_implementation="sdpa", dtype=torch.bfloat16).to("cuda:0").eval()
from PIL import Image
img = Image.new("RGB", (2400, 2400))   # any image; ~2400px longest side -> ~14.7k patches (over threshold)
inp = proc(images=img, return_tensors="pt").to("cuda:0")
feats = [model.model.get_image_features(inp["pixel_values"], inp["image_grid_thw"], return_dict=True)["last_hidden_state"] for _ in range(3)]
print("max|Δ| across 3 runs:", max(torch.abs(feats[0]-feats[1]).max().item(), torch.abs(feats[0]-feats[2]).max().item()), "NaN:", any(torch.isnan(f).any().item() for f in feats))
# Under threshold (cap): proc.image_processor.size.longest_edge = 3_400_000 -> max|Δ| == 0, no NaN
```

Thanks very much!
