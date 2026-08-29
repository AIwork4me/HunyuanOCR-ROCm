#!/usr/bin/env python3
"""Repro for vllm-project/vllm#50603 — warmup test: fixes determinism, not correctness.

Full-res page (15360 patches). One WARMUP generate() (discarded), then three measured
identical calls. Expect the three measured runs to be byte-identical (warmup fixes
symptom A) but still garbled (symptom B remains).
"""

import hashlib

from PIL import Image, ImageDraw, ImageFont
from vllm import LLM, SamplingParams

W, H = 2048, 1920
im = Image.new("RGB", (W, H), "white")
d = ImageDraw.Draw(im)
bf = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
rf = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
d.text((120, 120), "Quarterly Financial Summary", font=ImageFont.truetype(bf, 96), fill="black")
for i, ln in enumerate(
    [
        "Revenue: 12,480,000 USD",
        "Cost of Goods Sold: 4,210,000 USD",
        "Gross Profit: 8,270,000 USD",
        "Operating Expense: 3,150,000 USD",
        "Net Income: 5,120,000 USD",
        "Earnings Per Share: 2.84 USD",
        "Fiscal Year: 2026 Q2 Report",
        "Prepared by Finance Committee",
    ]
):
    d.text((120, 320 + i * 130), ln, font=ImageFont.truetype(rf, 64), fill="black")
im.save("/tmp/page.png")

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
sp = SamplingParams(temperature=0.0, top_p=1.0, top_k=-1, max_tokens=1024, repetition_penalty=1.08)
prompt = llm.get_tokenizer().apply_chat_template(
    [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": PROMPT}]}],
    tokenize=False,
    add_generation_prompt=True,
)
inputs = {"prompt": prompt, "multi_modal_data": {"image": Image.open("/tmp/page.png").convert("RGB")}}

w = llm.generate(inputs, sampling_params=sp, use_tqdm=False)[0].outputs[0].text.strip()
print(f"[warmup] sha8={hashlib.sha256(w.encode()).hexdigest()[:8]} (discarded)")

outs = []
for i in range(3):
    t = llm.generate(inputs, sampling_params=sp, use_tqdm=False)[0].outputs[0].text.strip()
    outs.append(t)
    print(f"measured run{i}: sha8={hashlib.sha256(t.encode()).hexdigest()[:8]} :: {t[:80]!r}")
print("measured 3x identical (warmup fixed determinism):", outs[0] == outs[1] == outs[2])
print(
    "GT title present:",
    any("quarterly financial summary" in o.lower() for o in outs),
    "(expect False — symptom B / long-seq corruption remains)",
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
                "warmup_sha8": hashlib.sha256(w.encode()).hexdigest()[:8],
                "texts": outs,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )
