# SPDX-License-Identifier: NOASSERTION
# Copyright (c) Tencent. All rights reserved.
# Copyright 2026 AIwork4me (modifications for the ROCm port)
#
# This file derives from Tencent HunyuanOCR (https://github.com/Tencent-Hunyuan/HunyuanOCR),
# licensed under the Tencent Hunyuan Community License Agreement. Upstream-derived
# portions retain that license; see LICENSES/Tencent-Hunyuan-Community-License.txt.
# The "Powered by Tencent Hunyuan" mark is encouraged (license §3c), not required.
"""Transformers backend for HunyuanOCR-1.5 (Phase 1 oracle).

Ported from upstream inference/transformers/infer_hf_8gpu_hyocr15.py:
  HunYuanVLForConditionalGeneration + AutoProcessor, dtype=bfloat16, attn=eager,
  greedy decode (do_sample=False, repetition_penalty=1.08, max_new_tokens=32768),
  tail-repetition StoppingCriteria, clean_repeated_substrings + process_one.
Torch/transformers imported lazily so importing this module needs no GPU.
"""
from __future__ import annotations
import importlib
import os
import sys
from typing import List

from ..contract import CONTRACT
from ..postprocess import clean_repeated_substrings, process_one


# --- gfx1100 / ROCm ViT resolution cap (workaround) -------------------------
# On AMD gfx1100 + torch 2.9.1 ROCm, the Hunyuan-ViT bf16 forward becomes
# non-deterministic and emits NaN above ~14.2k-14.7k vision tokens (sharp
# threshold). Capping the input pixel area keeps the patch count below the
# threshold so inference is deterministic and correct. Override or disable
# (set to 0) via HUNYUANOCR_VIT_MAX_PIXELS. See ROCm issue #6416:
# https://github.com/ROCm/ROCm/issues/6416
GFX1100_VIT_MAX_PIXELS = int(os.environ.get("HUNYUANOCR_VIT_MAX_PIXELS", "3400000"))
# gfx1100 attention impl. With the ViT cap above, sdpa is deterministic, correct,
# and ~1.4x faster than eager on RDNA3 (eager is the upstream default, tuned for
# H100). Override via HUNYUANOCR_ATTN.
GFX1100_ATTN_IMPLEMENTATION = os.environ.get("HUNYUANOCR_ATTN", "sdpa")


def _apply_vit_resolution_cap(processor):
    """Cap HunyuanVL image-processor pixel area to stay under the ROCm ViT
    threshold. No-op when the cap is 0 or the processor lacks the ``size`` knob."""
    if GFX1100_VIT_MAX_PIXELS:
        try:
            processor.image_processor.size.longest_edge = GFX1100_VIT_MAX_PIXELS
        except Exception:
            pass
    return processor


def _patch_hunyuan_tokenizer_special_tokens(tokenizer) -> None:
    """Backfill missing special-token attrs on older HunyuanOCR tokenizers."""
    init_kwargs = getattr(tokenizer, "init_kwargs", {}) or {}
    extra_tokens = init_kwargs.get("extra_special_tokens", {}) or {}
    defaults = {
        "image_token": "<｜hy_place▁holder▁no▁102｜>",
        "image_start_token": "<｜hy_place▁holder▁no▁100｜>",
        "image_end_token": "<｜hy_place▁holder▁no▁101｜>",
        "video_token": "<｜hy_place▁holder▁no▁103｜>",
        "video_start_token": "<｜hy_place▁holder▁no▁104｜>",
        "video_end_token": "<｜hy_place▁holder▁no▁105｜>",
    }
    for name, default_value in defaults.items():
        if hasattr(tokenizer, name):
            continue
        value = extra_tokens.get(name)
        if value is None and name == "video_token":
            value = extra_tokens.get("image_token")
        setattr(tokenizer, name, value or default_value)


def _load_processor_with_patch(model_path: str):
    from transformers import AutoImageProcessor, AutoTokenizer
    proc_mod = importlib.import_module("transformers.models.hunyuan_vl.processing_hunyuan_vl")
    HunYuanVLProcessor = proc_mod.HunYuanVLProcessor
    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)
    _patch_hunyuan_tokenizer_special_tokens(tokenizer)
    image_processor = AutoImageProcessor.from_pretrained(model_path)
    video_processor = None
    try:
        from transformers import AutoVideoProcessor
        video_processor = AutoVideoProcessor.from_pretrained(model_path)
    except Exception:
        video_processor = None
    try:
        return HunYuanVLProcessor(image_processor=image_processor, tokenizer=tokenizer, video_processor=video_processor)
    except TypeError:
        return HunYuanVLProcessor(image_processor, tokenizer, video_processor)


def load_model_and_processor(model_path: str, device: str = "cuda:0"):
    import torch
    from transformers import AutoProcessor, HunYuanVLForConditionalGeneration
    dtype = getattr(torch, CONTRACT.dtype)
    try:
        processor = AutoProcessor.from_pretrained(model_path, use_fast=False)
    except AttributeError as e:
        if "video_token" not in str(e):
            raise
        print("[warn] AutoProcessor tokenizer lacks video_token; retrying with patched tokenizer.", file=sys.stderr)
        processor = _load_processor_with_patch(model_path)
    processor = _apply_vit_resolution_cap(processor)
    model = HunYuanVLForConditionalGeneration.from_pretrained(
        model_path, attn_implementation=GFX1100_ATTN_IMPLEMENTATION, dtype=dtype,
    )
    model = model.to(device)
    model.eval()
    return model, processor


def build_messages(image_path: str, prompt: str) -> List[dict]:
    """Upstream message shape: empty system + user[image, text]."""
    return [
        {"role": "system", "content": ""},
        {"role": "user", "content": [
            {"type": "image", "image": image_path},
            {"type": "text", "text": prompt},
        ]},
    ]


def _build_tail_repetition_stop(processor, prompt_len: int):
    """StoppingCriteria mirroring the vLLM streaming early-stop (per upstream)."""
    from transformers import StoppingCriteria, StoppingCriteriaList
    from ..postprocess import has_tail_repetition
    tokenizer = processor.tokenizer
    min_repeats = CONTRACT.repeat_min_repeats
    check_start_chars, check_step_chars, token_probe_step = 4000, 1000, 64

    class TailRepetitionStop(StoppingCriteria):
        def __init__(self):
            self._next_check_at_chars = check_start_chars
            self._last_probe_tokens = 0
            self._triggered = False

        def __call__(self, input_ids, scores, **kwargs):
            if self._triggered:
                return True
            new_tokens = input_ids[0, prompt_len:]
            n_new = int(new_tokens.numel())
            if n_new - self._last_probe_tokens < token_probe_step:
                return False
            self._last_probe_tokens = n_new
            try:
                text = tokenizer.decode(new_tokens, skip_special_tokens=True)
            except Exception:
                return False
            acc_len = len(text)
            if acc_len < self._next_check_at_chars:
                return False
            self._next_check_at_chars = acc_len + check_step_chars
            if has_tail_repetition(text[-8000:], min_repeats=min_repeats):
                self._triggered = True
                return True
            return False

    return StoppingCriteriaList([TailRepetitionStop()])


def infer_one(model, processor, image_path: str, prompt: str, device: str = "cuda:0") -> str:
    """Run one image through the model with the frozen contract; return markdown."""
    import torch
    from PIL import Image
    tokenizer = processor.tokenizer
    eos_token_id = getattr(tokenizer, "eos_token_id", None)
    pad_token_id = getattr(tokenizer, "pad_token_id", None) or eos_token_id

    with Image.open(image_path) as raw:
        image = raw.convert("RGB")

    text = processor.apply_chat_template(build_messages(image_path, prompt), tokenize=False, add_generation_prompt=True)
    inputs = processor(text=[text], images=image, padding=True, return_tensors="pt").to(device)
    input_ids = inputs["input_ids"] if "input_ids" in inputs else inputs["inputs"]
    prompt_len = int(input_ids.shape[1])

    stopping_criteria = _build_tail_repetition_stop(processor, prompt_len=prompt_len)
    gen_kwargs = dict(
        max_new_tokens=32768, do_sample=False, repetition_penalty=1.08, use_cache=True,
        stopping_criteria=stopping_criteria,
    )
    if eos_token_id is not None:
        gen_kwargs["eos_token_id"] = eos_token_id
    if pad_token_id is not None:
        gen_kwargs["pad_token_id"] = pad_token_id

    with torch.inference_mode():
        generated_ids = model.generate(**inputs, **gen_kwargs)
    trimmed = [out_ids[len(in_ids):] for in_ids, out_ids in zip(input_ids, generated_ids)]
    decoded = processor.batch_decode(trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)
    out_text = decoded[0] if decoded else ""
    out_text = clean_repeated_substrings(out_text)
    try:
        out_text, _ = process_one(out_text)      # doc_parse normalization (frozen postproc)
    except Exception:
        pass
    return out_text
