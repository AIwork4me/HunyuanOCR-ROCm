# [gfx1100] bf16 ViT forward becomes non-deterministic and emits NaN above a sharp ~14.3k-token sequence-length threshold

## Summary

On **gfx1100 (RDNA3)** with the ROCm **torch 2.9.1** wheel (`hip 7.2.53211`) and **transformers 5.13.0**, the vision transformer (a standard pre-norm ViT using PyTorch SDPA attention) produces **non-deterministic output and `NaN`** when the per-image token/patch sequence length exceeds a **sharp threshold of ~14,200–14,688 tokens**. Below the threshold the same forward is **bit-for-bit deterministic** and the model produces correct output. The non-determinism is observed for a **deterministic-by-spec** forward (fixed weights, fixed input, `do_sample=False` / pure `argmax`).

This breaks end-to-end inference for high-resolution inputs: at full page resolution (~15k vision tokens) the model emits degenerate output (`"I"`, `"(1,0),(1000,999)"`, or a coherent-but-wrong paragraph on different identical runs) instead of correct OCR.

## Environment

- **GPU:** AMD Radeon gfx1100 (RDNA3), 48 GB (verified `rocminfo`: `gfx1100`)
- **ROCm:** userland `hip 7.2.53211-e1a6bc5663`; ROCk kernel module `6.14.14`
- **torch:** `2.9.1+gitff65f5b` (ROCm/HIP build), `torch.version.hip == 7.2.53211-e1a6bc5663`
- **transformers:** `5.13.0`
- **Model:** `tencent/HunyuanOCR` (HunyuanOCR-1.5, ~1 B params; native `hunyuan_vl` arch, standard pre-norm ViT vision tower with SDPA attention, `patch_size=16`, no QK-norm in the vision attention)
- OS: Linux x86_64; Python 3.12

## Reproducibility

Reproduces **every run** above the threshold; deterministic below it. The instability is in the **vision tower forward** (`model.model.get_image_features`), isolated as follows (all on the same GPU, bf16):

| Isolation experiment @ seq≈14,688 | Result |
|---|---|
| Full model ViT `get_image_features`, 3× identical | **non-deterministic** (`max|Δ|` up to 9312) and intermittently **NaN** |
| Bare `bf16` batched matmul `Q@Kᵀ` (heads×seq×hd) | **deterministic** (`max|Δ| = 0`), no NaN |
| Standalone `F.scaled_dot_product_attention` (bf16) | **deterministic**, no NaN |
| `nn.LayerNorm(1152)` on `[1, seq, 1152]` | **deterministic**, no NaN |
| Single real ViT block (`layers[9]`) with random input | **deterministic**, no NaN |
| LLM text-only forward @ 4,000 tokens | **deterministic**, no NaN |

So no single op reproduces it; the instability **emerges only in the full 27-layer ViT forward with the model's real weights/activations** at long sequence. The LLM is unaffected.

### Sequence-length threshold (full ViT `get_image_features`, 3× identical bf16 forwards)

| Image max-dim | ViT patches (seq len) | `max|Δ|` across 3 runs | NaN |
|---|---|---|---|
| ≤ 2272 | ≤ 14,200 | **0.000** | No (deterministic) |
| 2304 | 14,688 | **9312 / NaN** | **Yes** |
| full (2339) | 15,184 | **NaN** | **Yes** |

The transition is sharp: **deterministic at 14,200 tokens, non-deterministic + NaN at 14,688 tokens.** Just below the threshold (e.g. 13,056 tokens) end-to-end greedy decoding produces **correct OCR**; just above it produces garbage.

### End-to-end greedy (deterministic-by-spec) at full resolution

Three **identical** `do_sample=False` generations on the same full-resolution image yield three different outputs:

```
run0: 'I'
run1: '(1,0),(1000,999)'
run2: '# 2023年大学英语六级词汇选择与翻译\n\n## 词汇选择\n1. 首先'
```

## Minimal repro

```python
# Requires: pip install torch (ROCm 7.x wheel) "transformers==5.13.0" pillow
# Weights:  huggingface-cli download tencent/HunyuanOCR   (~2.2 GB)
import torch
from PIL import Image
from transformers import AutoProcessor, HunYuanVLForConditionalGeneration

MODEL = "tencent/HunyuanOCR"  # or local path
proc = AutoProcessor.from_pretrained(MODEL, use_fast=False)
model = (
    HunYuanVLForConditionalGeneration.from_pretrained(MODEL, attn_implementation="sdpa", dtype=torch.bfloat16)
    .to("cuda:0")
    .eval()
)

# any RGB image; thumbnail controls the ViT sequence length
img = Image.new("RGB", (3000, 3000))  # or a real page image


def vit_run(maxdim):
    im = img.copy()
    im.thumbnail((maxdim, maxdim))
    inp = proc(images=im, return_tensors="pt").to("cuda:0")  # pixel_values + image_grid_thw
    feats = []
    with torch.inference_mode():
        for _ in range(3):
            f = (
                model.model.get_image_features(inp["pixel_values"], inp["image_grid_thw"], return_dict=True)[
                    "last_hidden_state"
                ]
                .float()
                .cpu()
            )
            feats.append(f)
    d = max(torch.abs(feats[0] - feats[1]).max().item(), torch.abs(feats[0] - feats[2]).max().item())
    nan = any(torch.isnan(x).any().item() for x in feats)
    return inp["pixel_values"].shape[0], d, nan


for md in (2208, 2304):  # 2208 -> ~13.5k tokens (safe); 2304 -> ~14.7k tokens (triggers)
    n, d, nan = vit_run(md)
    print(f"maxdim={md} tokens={n} 3x-max-diff={d} NaN={nan}")
# Expected:
#   maxdim=2208 tokens=~13524 3x-max-diff=0.0   NaN=False
#   maxdim=2304 tokens=~14688 3x-max-diff=NaN(or large) NaN=True
```

## Notes / what I could not confirm

- **No NVIDIA GPU available** in this environment, so I could **not directly compare against CUDA** to prove ROCm-specificity. A run of the above on an NVIDIA GPU (where upstream reports the model works correctly at this resolution) would confirm whether this is a ROCm numerical issue vs a model-level instability.
- A single `fp32` forward at full resolution on gfx1100 does **not** NaN (it stays finite), which suggests the failure is specifically in the **bf16** long-sequence path. The **run-to-run non-determinism of a deterministic forward**, however, is a GPU/kernel behavior regardless of precision.
- The threshold (~14,300 tokens) does not correspond to an obvious power-of-two boundary; it may align with an internal tile/sequence limit in a HIP attention/GEMM kernel.

## Workaround

Cap the image resolution so the ViT sequence length stays below ~14,000 tokens (≈ `max_pixels ≈ 3.5e6`, i.e. longest side ≲ 1870 px). Below the threshold inference is deterministic and produces correct output.

## Ask

Could the ROCm team:
1. Reproduce the above on gfx1100 (RDNA3) and confirm whether the bf16 long-sequence ViT forward non-determinism + NaN is a HIP kernel issue;
2. If possible, compare against CUDA to confirm ROCm-specificity;
3. Advise whether a kernel/environment setting (e.g. forcing an fp32-accumulation attention path, or a newer ROCm/torch build) restores determinism and correctness at long sequence.

Happy to provide more isolation data or a narrower repro on request.
