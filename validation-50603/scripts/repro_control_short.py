#!/usr/bin/env python3
"""Repro for vllm-project/vllm#50603 — control: symptom A is NOT length-gated.

Same as repro_determinism.py but a SMALL page (1024x960 = 3840 ViT patches, well below
~14k). If run 0 still differs from runs 1 & 2, the first-call non-determinism is not a
long-sequence-threshold effect.
"""

import hashlib

from PIL import Image, ImageDraw, ImageFont
from vllm import LLM, SamplingParams

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
        "Operating Expense: 3,150,000 USD",
        "Net Income: 5,120,000 USD",
    ]
):
    d.text((60, 170 + i * 70), ln, font=ImageFont.truetype(rf, 36), fill="black")
im.save("/tmp/page_small.png")

PROMPT = (
    "提取文档图片中正文的所有信息用markdown格式表示，其中页眉、页脚部分忽略，"
    "表格用html格式表达，文档中公式用latex格式表示，按照阅读顺序组织进行解析。"
)

llm = LLM(
    model="tencent/HunyuanOCR",
    dtype="bfloat16",
    max_model_len=32768,
    gpu_memory_utilization=0.90,
    enforce_eager=True,
    limit_mm_per_prompt={"image": 1},
    trust_remote_code=True,
)
sp = SamplingParams(temperature=0.0, top_p=1.0, top_k=-1, max_tokens=512, repetition_penalty=1.08)
prompt = llm.get_tokenizer().apply_chat_template(
    [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": PROMPT}]}],
    tokenize=False,
    add_generation_prompt=True,
)
img = Image.open("/tmp/page_small.png").convert("RGB")

outs = []
for i in range(3):
    t = (
        llm.generate({"prompt": prompt, "multi_modal_data": {"image": img}}, sampling_params=sp, use_tqdm=False)[0]
        .outputs[0]
        .text.strip()
    )
    outs.append(t)
    print(f"run{i}: sha8={hashlib.sha256(t.encode()).hexdigest()[:8]} chars={len(t)}")
print(
    "3x identical (deterministic):",
    outs[0] == outs[1] == outs[2],
    "(expect False — first-call effect at short seq too)",
)

# --- validation-50603 additive evidence dump: the reproducer code above is
# --- verbatim from the issue gist; this block only records what was printed.
import json
import os

if os.environ.get("V50603_EVIDENCE"):
    with open(os.environ["V50603_EVIDENCE"], "w") as f:
        json.dump(
            {
                "script": os.path.basename(__file__),
                "sha8s": [hashlib.sha256(o.encode()).hexdigest()[:8] for o in outs],
                "texts": outs,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )
