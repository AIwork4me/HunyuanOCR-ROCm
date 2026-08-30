#!/usr/bin/env python3
"""Boundary A probe: call HF HunYuanVL.get_rope_index directly (meta model, no weights).

Shows what the Transformers backend's get_mrope_input_positions would receive as
prefill positions for the regression-test workload: a (axes, 1, seq) tensor whose
first dim is len(mrope_section) == 4 for tencent/HunyuanOCR.
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

MODEL = os.environ.get(
    "MODEL_PATH",
    "/root/.cache/huggingface/hub/models--tencent--HunyuanOCR/snapshots/"
    "de8f10ad2f00a0cefd790b526de8a65dcfdb3205",
)
_banner(MODEL)

W, H = 1024, 960
im = Image.new("RGB", (W, H), "white")
d = ImageDraw.Draw(im)
bf = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
rf = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
d.text((60, 60), "Quarterly Financial Summary", font=ImageFont.truetype(bf, 56), fill="black")
for i, ln in enumerate(["Revenue: 12,480,000 USD", "Cost of Goods Sold: 4,210,000 USD", "Gross Profit: 8,270,000 USD"]):
    d.text((60, 170 + i * 70), ln, font=ImageFont.truetype(rf, 36), fill="black")
im.save("/tmp/v53615_page_small.png")

from transformers import AutoConfig, AutoProcessor

cfg = AutoConfig.from_pretrained(MODEL, trust_remote_code=True)
proc = AutoProcessor.from_pretrained(MODEL, trust_remote_code=True)

PROMPT = (
    "提取文档图片中正文的所有信息用markdown格式表示，其中页眉、页脚部分忽略，"
    "表格用html格式表达，文档中公式用latex格式表示，按照阅读顺序组织进行解析。"
)
text = proc.apply_chat_template(
    [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": PROMPT}]}],
    tokenize=False,
    add_generation_prompt=True,
)
inputs = proc(images=[Image.open("/tmp/v53615_page_small.png").convert("RGB")], text=text, return_tensors="pt")
input_ids = inputs["input_ids"]
grid = inputs["image_grid_thw"]
print("input_ids:", tuple(input_ids.shape), "image_grid_thw:", grid.tolist())

mm_token_type_ids = (input_ids == cfg.image_token_id).to(torch.int)
# contiguous spans only mark the first token of each image group in some
# processors; hunyuan marks every vision token
print("vision-token count:", int(mm_token_type_ids.sum()))

from transformers.models.hunyuan_vl.modeling_hunyuan_vl import HunYuanVLModel

with torch.device("meta"):
    model = HunYuanVLModel._from_config(cfg)

rope_parameters = model.config.text_config.rope_parameters
mrope_section = rope_parameters.get("mrope_section")
print("text_config.rope_parameters.mrope_section:", mrope_section)
print("sections sum:", sum(mrope_section), "| head_dim // 2:", model.config.text_config.head_dim // 2)

pos, delta = model.get_rope_index(
    input_ids=input_ids,
    image_grid_thw=grid,
    mm_token_type_ids=mm_token_type_ids,
)
print("get_rope_index output shape:", tuple(pos.shape), "delta:", delta.tolist())
pos0 = pos[:, 0]
for ax in range(pos.shape[0]):
    row = pos0[ax]
    vis = row[mm_token_type_ids[0].bool()]
    txt = row[~mm_token_type_ids[0].bool()]
    print(
        f"axis {ax}: min={int(row.min())} max={int(row.max())} "
        f"text-range=({int(txt.min())}..{int(txt.max())}) vis-range=({int(vis.min())}..{int(vis.max())})"
    )
