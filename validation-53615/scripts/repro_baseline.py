#!/usr/bin/env python3
"""Minimal HunyuanOCR reproducer for the #53615 Transformers-backend regression.

One synthetic document image -> processor -> model runner -> language model ->
generation. Greedy, TP=1, enforce_eager, conservative max_model_len. On pristine
upstream/main (b5707bf994) this is expected to fail with a 3-vs-4 multimodal
RoPE position-channel mismatch.
"""

import os

os.environ.setdefault("HF_HUB_OFFLINE", "1")

MODEL_ID = os.environ.get("MODEL_ID", "tencent/HunyuanOCR")

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


_banner(MODEL_ID)

from PIL import Image, ImageDraw, ImageFont

W, H = 1024, 960
im = Image.new("RGB", (W, H), "white")
d = ImageDraw.Draw(im)
bf = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
rf = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
d.text((60, 60), "Quarterly Financial Summary", font=ImageFont.truetype(bf, 56), fill="black")
for i, ln in enumerate(
    [
        "Revenue: 12,480,000 USD",
        "Cost of Goods Sold: 4,210,000 USD",
        "Gross Profit: 8,270,000 USD",
    ]
):
    d.text((60, 170 + i * 70), ln, font=ImageFont.truetype(rf, 36), fill="black")
im.save("/tmp/v53615_page_small.png")

PROMPT = (
    "提取文档图片中正文的所有信息用markdown格式表示，其中页眉、页脚部分忽略，"
    "表格用html格式表达，文档中公式用latex格式表示，按照阅读顺序组织进行解析。"
)

from vllm import LLM, SamplingParams

llm = LLM(
    model=MODEL_ID,
    dtype="bfloat16",
    max_model_len=8192,
    **({"max_num_batched_tokens": int(os.environ["V53615_MAX_BATCHED_TOKENS"])} if os.environ.get("V53615_MAX_BATCHED_TOKENS") else {}),
    gpu_memory_utilization=0.90,
    enforce_eager=True,
    limit_mm_per_prompt={"image": 1},
    trust_remote_code=True,
)
sp = SamplingParams(temperature=0.0, top_p=1.0, top_k=-1, max_tokens=128)
prompt = llm.get_tokenizer().apply_chat_template(
    [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": PROMPT}]}],
    tokenize=False,
    add_generation_prompt=True,
)
img = Image.open("/tmp/v53615_page_small.png").convert("RGB")
out = llm.generate(
    {"prompt": prompt, "multi_modal_data": {"image": img}}, sampling_params=sp, use_tqdm=False
)[0]
print("=== OUTPUT ===")
print(out.outputs[0].text.strip())
print("=== TOKEN IDS ===")
print(list(out.outputs[0].token_ids))
