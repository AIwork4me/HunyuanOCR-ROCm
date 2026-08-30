#!/usr/bin/env python3
"""HF-only control: same workload through plain transformers 5.13.0 (sdpa, greedy).

Reference output for judging the vLLM Transformers-backend path: if this
produces the faithful markdown while vLLM main + fixes does not, the remaining
garble is in vLLM's plumbing, not the HF model/processor.
"""

import os

os.environ.setdefault("HF_HUB_OFFLINE", "1")

def _banner(model_desc):
    """Print the exact vLLM SHA and package versions before doing anything."""
    import subprocess
    import sys

    import torch
    import transformers

    print(f"[env] model: {model_desc}", flush=True)
    try:
        import vllm

        repo = os.path.dirname(vllm.__file__)
        sha = subprocess.run(
            ["git", "-C", repo, "rev-parse", "HEAD"], capture_output=True, text=True
        ).stdout.strip()
        print(f"[env] vllm {vllm.__version__} @ {sha} ({vllm.__file__})", flush=True)
    except Exception as e:
        print(f"[env] vllm import failed: {e!r}", flush=True)
    print(
        f"[env] python {sys.version.split()[0]} | torch {torch.__version__} "
        f"| transformers {transformers.__version__}",
        flush=True,
    )


import torch
from PIL import Image, ImageDraw, ImageFont

W, H = 1024, 960
im = Image.new("RGB", (W, H), "white")
d = ImageDraw.Draw(im)
bf = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
rf = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
d.text((60, 60), "Quarterly Financial Summary", font=ImageFont.truetype(bf, 56), fill="black")
for i, ln in enumerate(["Revenue: 12,480,000 USD", "Cost of Goods Sold: 4,210,000 USD", "Gross Profit: 8,270,000 USD"]):
    d.text((60, 170 + i * 70), ln, font=ImageFont.truetype(rf, 36), fill="black")
im.save("/tmp/v53615_page_small.png")

PROMPT = (
    "提取文档图片中正文的所有信息用markdown格式表示，其中页眉、页脚部分忽略，"
    "表格用html格式表达，文档中公式用latex格式表示，按照阅读顺序组织进行解析。"
)

from transformers import AutoModelForImageTextToText, AutoProcessor

MODEL = os.environ.get(
    "MODEL_PATH",
    "/root/.cache/huggingface/hub/models--tencent--HunyuanOCR/snapshots/"
    "de8f10ad2f00a0cefd790b526de8a65dcfdb3205",
)
_banner(MODEL)

processor = AutoProcessor.from_pretrained(MODEL, trust_remote_code=True)
model = AutoModelForImageTextToText.from_pretrained(
    MODEL,
    torch_dtype=torch.bfloat16,
    attn_implementation="sdpa",
    device_map="cuda",
    trust_remote_code=True,
)
model.eval()

text = processor.apply_chat_template(
    [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": PROMPT}]}],
    tokenize=False,
    add_generation_prompt=True,
)
inputs = processor(images=[Image.open("/tmp/v53615_page_small.png").convert("RGB")], text=text, return_tensors="pt").to("cuda")

print("input_ids:", tuple(inputs["input_ids"].shape), "grid:", inputs["image_grid_thw"].tolist())
with torch.inference_mode():
    out = model.generate(
        **inputs,
        do_sample=False,
        temperature=None,
        top_p=None,
        top_k=None,
        max_new_tokens=128,
    )
new_tokens = out[0][inputs["input_ids"].shape[1]:]
print("=== OUTPUT ===")
print(processor.tokenizer.decode(new_tokens, skip_special_tokens=True).strip())
print("=== TOKEN IDS ===")
print(new_tokens.tolist())
