#!/usr/bin/env python3
"""Instrumented HunyuanOCR reproducer — boundary evidence for the 3-vs-4 m-RoPE bug.

Runs the exact workload of repro_baseline.py with monkeypatched logging (NO
production-file edits) at the boundaries:
  A. HF get_rope_index output (via TransformersMultiModal get_mrope_input_positions)
  B. vLLM allocated RoPE position buffer (RopeState.__init__)
  C. per-request prefill position staging (RopeState.init_prefill_positions)
  D/E. positions handed into the HF language model (captured at forward).
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


import torch
_banner(MODEL_ID)

from PIL import Image, ImageDraw, ImageFont


def log(tag, msg):
    print(f"[V53615 {tag}] {msg}", flush=True)


# --- instrumentation: patch BEFORE engine construction -----------------------
import vllm.v1.worker.gpu.mm.rope as rope_mod
import vllm.model_executor.models.transformers.multimodal as tf_mm

_orig_get_rope_state = rope_mod.get_rope_state
_orig_rope_init = rope_mod.RopeState.__init__
_orig_init_prefill = rope_mod.RopeState.init_prefill_positions
_orig_get_mrope = tf_mm.MultiModalMixin.get_mrope_input_positions


def patched_get_rope_state(model_config, model, *a, **kw):
    state = _orig_get_rope_state(model_config, model, *a, **kw)
    log(
        "get_rope_state",
        f"uses_mrope={model_config.uses_mrope} uses_xdrope_dim={model_config.uses_xdrope_dim} "
        f"-> RopeState num_dims={state.num_dims if state else None} has_delta={state.has_delta if state else None}",
    )
    return state


def patched_rope_init(self, num_dims, has_delta, *a, **kw):
    _orig_rope_init(self, num_dims, has_delta, *a, **kw)
    log("RopeState.__init__", f"num_dims={num_dims} positions_buffer={tuple(self.positions.shape)}")


def patched_init_prefill(self, req_idx, model, prefill_token_ids, mm_features):
    if self.has_delta:
        pos, delta = model.get_mrope_input_positions(prefill_token_ids, mm_features)
        log(
            "init_prefill_positions",
            f"model returned {pos.shape[0]} axes x {pos.shape[1]} tokens (delta={delta}); "
            f"staging loop keeps only num_dims={self.num_dims}",
        )
        import types

        bound = types.MethodType(_orig_init_prefill, self)
        return bound(req_idx, model, prefill_token_ids, mm_features)
    return _orig_init_prefill(self, req_idx, model, prefill_token_ids, mm_features)


def patched_get_mrope(self, input_tokens, mm_features):
    out = _orig_get_mrope(self, input_tokens, mm_features)
    pos = out[0]
    log("get_mrope_input_positions", f"returned tensor {tuple(pos.shape)} (axes, tokens) delta={out[1]}")
    return out


rope_mod.get_rope_state = patched_get_rope_state
rope_mod.RopeState.__init__ = patched_rope_init
rope_mod.RopeState.init_prefill_positions = patched_init_prefill
tf_mm.MultiModalMixin.get_mrope_input_positions = patched_get_mrope

# Boundary D (positions handed into the HF language model) is evidenced by the
# ValueError itself. Boundary E: what the HunYuan rotary embedding receives.
import transformers.models.hunyuan_vl.modeling_hunyuan_vl as hf_hy

_orig_rotary_forward = hf_hy.HunYuanVLRotaryEmbedding.forward


def patched_rotary_forward(self, x, position_ids=None, *a, **kw):
    if position_ids is not None:
        log(
            "HF HunYuanVLRotaryEmbedding.forward",
            f"position_ids {tuple(position_ids.shape)} x {tuple(x.shape[-2:])} "
            f"mrope_section={getattr(self, 'mrope_section', None)}",
        )
    return _orig_rotary_forward(self, x, position_ids, *a, **kw)


hf_hy.HunYuanVLRotaryEmbedding.forward = patched_rotary_forward

# --- workload (identical to repro_baseline.py) -------------------------------
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
