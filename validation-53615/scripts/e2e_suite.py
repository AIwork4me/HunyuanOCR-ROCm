#!/usr/bin/env python3
"""validation-53615 Phase 8 E2E matrix on the W7900 (gfx1100).

Tests (greedy, TP=1, transformers 5.15.1):
  1. simple image       — single-word image, expect the word in output
  2. document/OCR image — synthetic financial page, expect faithful markdown
  3. repeatability      — same document request x3 in-process, token-identical
  4. alternate aspect   — portrait 768x1280 page, expect faithful markdown

Mode variants are driven by env knobs (separate processes):
  V53615_EAGER=0|1 (default 1), V53615_NO_PREFIX_CACHE=1, plus the runner
  selection env (VLLM_USE_V2_MODEL_RUNNER) set by the driver.
"""

import hashlib
import json
import os
import sys

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

BF = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
RF = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"

PROMPT = (
    "提取文档图片中正文的所有信息用markdown格式表示，其中页眉、页脚部分忽略，"
    "表格用html格式表达，文档中公式用latex格式表示，按照阅读顺序组织进行解析。"
)


def text_page(w, h, title, lines, title_size=56, line_size=36):
    im = Image.new("RGB", (w, h), "white")
    d = ImageDraw.Draw(im)
    d.text((60, 60), title, font=ImageFont.truetype(BF, title_size), fill="black")
    for i, ln in enumerate(lines):
        d.text((60, 170 + i * (line_size + 34)), ln, font=ImageFont.truetype(RF, line_size), fill="black")
    return im


def images():
    simple = Image.new("RGB", (640, 400), "white")
    d = ImageDraw.Draw(simple)
    d.text((80, 160), "W7900-OK", font=ImageFont.truetype(BF, 72), fill="black")
    doc = text_page(1024, 960, "Quarterly Financial Summary",
                    ["Revenue: 12,480,000 USD", "Cost of Goods Sold: 4,210,000 USD", "Gross Profit: 8,270,000 USD"])
    portrait = text_page(768, 1280, "Maintenance Log",
                         ["Pump A: pressure nominal", "Valve B: replaced 2026-08-01", "Motor C: vibration high"], 48, 30)
    return {"simple": simple, "doc": doc, "portrait": portrait}


def sha(tokens):
    return hashlib.sha256(json.dumps(tokens).encode()).hexdigest()[:16]


def main():
    eager = os.environ.get("V53615_EAGER", "1") == "1"
    no_pc = os.environ.get("V53615_NO_PREFIX_CACHE", "0") == "1"
    mode = ("eager" if eager else "default") + ("+nopc" if no_pc else "")
    out_path = sys.argv[1]

    from vllm import LLM, SamplingParams

    kwargs = dict(
        model=MODEL_ID,
        dtype="bfloat16",
        max_model_len=8192,
        gpu_memory_utilization=0.90,
        enforce_eager=eager,
        limit_mm_per_prompt={"image": 1},
        trust_remote_code=True,
    )
    if no_pc:
        kwargs["enable_prefix_caching"] = False
    if os.environ.get("V53615_CUDAGRAPHS") == "piecewise":
        kwargs["compilation_config"] = {"cudagraph_mode": "PIECEWISE"}
    llm = LLM(**kwargs)
    sp = SamplingParams(temperature=0.0, top_p=1.0, top_k=-1, max_tokens=128)

    def run(img):
        prompt = llm.get_tokenizer().apply_chat_template(
            [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": PROMPT}]}],
            tokenize=False,
            add_generation_prompt=True,
        )
        o = llm.generate({"prompt": prompt, "multi_modal_data": {"image": img}}, sampling_params=sp, use_tqdm=False)[0]
        return o.outputs[0].text.strip(), list(o.outputs[0].token_ids)

    imgs = images()
    results = {"mode": mode, "tests": {}}

    t, ids = run(imgs["simple"])
    results["tests"]["1_simple"] = {"expect_in": "W7900", "ok": "W7900" in t.upper(), "text": t, "sha": sha(ids)}

    t, ids = run(imgs["doc"])
    ok = all(s in t for s in ("Quarterly", "12,480,000", "4,210,000", "8,270,000"))
    results["tests"]["2_document"] = {"ok": ok, "text": t, "sha": sha(ids)}

    rep = [run(imgs["doc"]) for _ in range(3)]
    results["tests"]["3_repeatability_x3"] = {
        "deterministic": len({sha(ids) for _, ids in rep}) == 1,
        "shas": [sha(ids) for _, ids in rep],
        "texts_equal": len({t for t, _ in rep}) == 1,
    }

    t, ids = run(imgs["portrait"])
    ok = all(s in t for s in ("Maintenance", "Pump A", "Valve B", "Motor C"))
    results["tests"]["4_portrait"] = {"ok": ok, "text": t, "sha": sha(ids)}

    results["all_ok"] = (
        results["tests"]["1_simple"]["ok"]
        and results["tests"]["2_document"]["ok"]
        and results["tests"]["3_repeatability_x3"]["deterministic"]
        and results["tests"]["4_portrait"]["ok"]
    )
    with open(out_path, "w") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(json.dumps(results, ensure_ascii=False, indent=2))
    sys.exit(0 if results["all_ok"] else 1)


if __name__ == "__main__":
    main()
