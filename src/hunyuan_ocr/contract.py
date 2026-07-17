# SPDX-License-Identifier: LicenseRef-Tencent-Hunyuan-Community-License
# Copyright (c) Tencent. All rights reserved.
# Copyright 2026 AIwork4me (modifications for the ROCm port)
#
# This file derives from Tencent HunyuanOCR (https://github.com/Tencent-Hunyuan/HunyuanOCR),
# licensed under the Tencent Hunyuan Community License Agreement. Upstream-derived
# portions retain that license; see LICENSES/LicenseRef-Tencent-Hunyuan-Community-License.txt.
# The "Powered by Tencent Hunyuan" mark is encouraged (license §3c), not required.
"""The FROZEN decoding contract — the single shared layer across backends.

Phase 1 (transformers) establishes BASELINE against these values; Phases 2 (vLLM)
and 3 (llama.cpp) MUST match it. Changing any value here re-baselines everything.
All values are copied verbatim from the upstream HunyuanOCR-1.5 inference recipe
(inference/transformers/infer_hf_8gpu_hyocr15.py, aligned with infer_vllm_client.py).
"""
from __future__ import annotations
from dataclasses import dataclass


# Sampling kwargs passed straight to model.generate() / mapped onto each backend.
SAMPLING: dict = {
    "do_sample": False,            # temperature=0.0 -> greedy
    "repetition_penalty": 1.08,
    "max_new_tokens": 32768,
    "use_cache": True,
}


@dataclass(frozen=True)
class Contract:
    # Task
    task_type: str = "doc_parse"
    prompt: str = (
        "提取文档图片中正文的所有信息用markdown格式表示，其中页眉、页脚部分忽略，"
        "表格用html格式表达，文档中公式用latex格式表示，按照阅读顺序组织进行解析。"
    )

    # Model loading
    dtype: str = "bfloat16"
    attn_implementation: str = "eager"

    # Decode
    skip_special_tokens: bool = True
    clean_up_tokenization_spaces: bool = False

    # Tail-repetition early-stop (has_tail_repetition min_repeats)
    repeat_min_repeats: int = 8

    # Post-processors applied in order, doc_parse only
    postprocessors: tuple = ("clean_repeated_substrings", "process_one")


CONTRACT: Contract = Contract()
