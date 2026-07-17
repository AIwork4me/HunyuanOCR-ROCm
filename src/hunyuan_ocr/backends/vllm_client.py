# SPDX-License-Identifier: LicenseRef-Tencent-Hunyuan-Community-License
# Copyright (c) Tencent. All rights reserved.
# Copyright 2026 AIwork4me (modifications for the ROCm port)
#
# This file derives from Tencent HunyuanOCR (https://github.com/Tencent-Hunyuan/HunyuanOCR),
# licensed under the Tencent Hunyuan Community License Agreement. Upstream-derived
# portions retain that license; see LICENSES/LicenseRef-Tencent-Hunyuan-Community-License.txt.
# The "Powered by Tencent Hunyuan" mark is encouraged (license §3c), not required.
"""vLLM backend client (Phase 2).

Talks to an OpenAI-compatible vLLM server serving ``tencent/HunyuanOCR`` and
mirrors the transformers backend's output via the shared decoding contract
(same prompt, sampling, streaming tail-repetition early-stop, and the
``clean_repeated_substrings`` + ``process_one`` post-processors) so the two
backends are directly comparable on OmniDocBench.

The server is started separately with ``scripts/serve_vllm.sh`` (the box's
``/opt/venv`` vLLM 0.16.1 ROCm build, which ships a native ``HunYuanVL``).
"""

from __future__ import annotations
import base64
import mimetypes

from ..contract import CONTRACT
from ..postprocess import clean_repeated_substrings, infer_stream, process_one


def encode_image_as_data_url(path: str) -> str:
    """image -> base64 data URL (mime inferred from extension; vLLM ignores the
    declared mime for base64 payloads). Mirrors upstream ``hunyuan_utils``."""
    mime = mimetypes.guess_type(path)[0] or "image/png"
    with open(path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode("utf-8")
    return f"data:{mime};base64,{b64}"


def _cap_cache_dir() -> str:
    """Cache dir for client-side capped image copies (never written next to the
    source). Override with HUNYUANOCR_CAP_CACHE. Lifetime: persists across runs
    and grows unbounded — clear it manually (``rm -rf $HUNYUANOCR_CAP_CACHE``)
    or point it at a tmpfs."""
    import os
    import tempfile

    d = os.environ.get("HUNYUANOCR_CAP_CACHE") or os.path.join(tempfile.gettempdir(), "hunyuanocr-caps")
    os.makedirs(d, exist_ok=True)
    return d


def _maybe_cap_image(path: str, max_pixels: int | None) -> str:
    """If ``max_pixels`` is set and the image exceeds it, return a path to a
    longest-edge-thumbnail copy that stays under the ROCm ViT threshold; else
    return the original path untouched.

    The copy lives in a content-hash-keyed cache (see :func:`_cap_cache_dir`),
    NEVER next to the source image, so read-only dataset directories are
    supported and there is no ``image.png.cap*.png`` pollution. Concurrent workers
    writing the same key race only on an atomic ``os.replace`` (last writer wins;
    content is identical so the result is correct).
    """
    if not max_pixels:
        return path
    import hashlib
    import os
    import threading

    from PIL import Image

    with Image.open(path) as im:
        w, h = im.size
        if w * h <= max_pixels:
            return path
        digest = hashlib.sha256(open(path, "rb").read()).hexdigest()[:16]
        cached = os.path.join(_cap_cache_dir(), f"{digest}_cap{max_pixels}.png")
        if os.path.isfile(cached) and os.path.getsize(cached) > 0:
            return cached
        scale = (max_pixels / (w * h)) ** 0.5
        thumb = im.convert("RGB")
        thumb.thumbnail((max(int(w * scale), 1), max(int(h * scale), 1)))
        # unique staging name so concurrent writers don't clobber each other
        staging = f"{cached}.{os.getpid()}.{threading.get_ident()}.tmp"
        thumb.save(staging, format="PNG")
        os.replace(staging, cached)  # atomic
        return cached


def infer_one(
    client,
    image_path: str,
    prompt: str | None = None,
    *,
    model: str = "tencent/HunyuanOCR",
    max_tokens: int = 32768,
    repetition_penalty: float = 1.08,
    repeat_min_repeats: int = 8,
    max_pixels: int | None = None,
) -> str:
    """Run one image through the vLLM server; return markdown (contract post-processed).

    Mirrors upstream ``inference/vllm_0_18_1/batch_infer.run_one``: image-first
    chat, greedy (``temperature=0``), ``repetition_penalty`` via ``extra_body``,
    the streaming tail-repetition early-stop, then ``clean_repeated_substrings``
    + ``process_one`` (doc_parse). ``max_pixels`` optionally caps the image
    client-side (set only if the vLLM ViT path also exhibits the >14k-token
    instability — to be determined by the Phase-2 determinism check)."""
    prompt = prompt or CONTRACT.prompt
    use_path = _maybe_cap_image(image_path, max_pixels)
    messages = [
        {"role": "system", "content": ""},
        {
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": encode_image_as_data_url(use_path)}},
                {"type": "text", "text": prompt},
            ],
        },
    ]
    common = dict(
        model=model,
        messages=messages,
        max_tokens=max_tokens,
        temperature=0.0,
        top_p=1.0,
        extra_body={"top_k": -1, "repetition_penalty": repetition_penalty, "skip_special_tokens": True},
    )
    text, _early = infer_stream(client, common, repeat_min_repeats)
    text = clean_repeated_substrings(text)
    try:
        text, _ = process_one(text)
    except Exception:
        pass
    return text
