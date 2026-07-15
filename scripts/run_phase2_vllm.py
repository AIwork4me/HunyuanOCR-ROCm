#!/usr/bin/env python3
"""Phase-2 driver: run HunyuanOCR-1.5 via vLLM OpenAI server(s) over OmniDocBench pages.

Writes one ``<stem>.md`` per page (resumable: skips pages whose .md exists) and
distributes work across one or more server endpoints (``--ports``) for
throughput. Uses the shared decoding contract via ``vllm_client.infer_one``.

Usage:
  # start one server per GPU first:
  GPU=0 PORT=8000 bash scripts/serve_vllm.sh
  GPU=1 PORT=8001 bash scripts/serve_vllm.sh
  GPU=2 PORT=8002 bash scripts/serve_vllm.sh
  # then:
  python scripts/run_phase2_vllm.py \
      --gt-json /workspace/OmniDocBench_data/OmniDocBench_150.json \
      --images-dir /workspace/OmniDocBench_data/images \
      --pred-dir /root/hunyuanocr-results/vllm-canary-150 \
      --ports 8000,8001,8002 [--max-pixels 3400000]
"""
from __future__ import annotations
import argparse
import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from openai import OpenAI  # noqa: E402

from hunyuan_ocr.backends.vllm_client import infer_one  # noqa: E402
from hunyuan_ocr.contract import CONTRACT  # noqa: E402


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--gt-json", required=True)
    p.add_argument("--images-dir", required=True)
    p.add_argument("--pred-dir", required=True)
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--ports", default="8000", help="comma-separated server ports")
    p.add_argument("--model", default="tencent/HunyuanOCR")
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--concurrency", type=int, default=24)
    p.add_argument("--max-pixels", type=int, default=0,
                   help="client-side ViT cap (0 = uncapped; set 3400000 if vLLM also needs the cap)")
    args = p.parse_args()

    pages = json.load(open(args.gt_json, encoding="utf-8"))
    if args.limit:
        pages = pages[:args.limit]
    os.makedirs(args.pred_dir, exist_ok=True)
    ports = [int(x) for x in args.ports.split(",") if x.strip()]
    clients = [OpenAI(api_key="EMPTY", base_url=f"http://{args.host}:{pt}/v1", timeout=3600.0)
               for pt in ports]
    max_pixels = args.max_pixels or None

    todo = []
    for pg in pages:
        stem = Path(pg["page_info"]["image_path"]).stem
        out = Path(args.pred_dir) / f"{stem}.md"
        if out.exists():
            continue
        img = os.path.join(args.images_dir, pg["page_info"]["image_path"])
        todo.append((stem, img, out))
    print(f"[info] {len(todo)} to do ({len(pages) - len(todo)} resumed) across ports {ports}",
          flush=True)

    def work(item):
        i, (stem, img, out) = item
        try:
            md = infer_one(clients[i % len(clients)], img, CONTRACT.prompt,
                           model=args.model, max_pixels=max_pixels)
        except Exception as e:
            md = f"ERROR: {type(e).__name__}: {e}"
        out.write_text(md, encoding="utf-8")
        return stem

    done = 0
    with ThreadPoolExecutor(max_workers=args.concurrency) as ex:
        futs = [ex.submit(work, it) for it in enumerate(todo)]
        for f in as_completed(futs):
            done += 1
            if done % 20 == 0:
                print(f"[info] {done}/{len(todo)}", flush=True)
    print(f"[done] {done} pages -> {args.pred_dir}", flush=True)


if __name__ == "__main__":
    main()
