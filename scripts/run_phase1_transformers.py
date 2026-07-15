#!/usr/bin/env python3
"""Phase-1 driver: run HunyuanOCR-1.5 (transformers) over OmniDocBench pages.

Spawns one worker process per GPU, shards the page list across them, and writes
one <stem>.md prediction per page. Resumable (skips pages whose .md exists).
Usage:
  python scripts/run_phase1_transformers.py \
      --gt-json /workspace/OmniDocBench_data/OmniDocBench.json \
      --images-dir /workspace/OmniDocBench_data/images \
      --pred-dir /root/hunyuanocr-results/phase1-transformers/preds \
      --model /root/models/HunyuanOCR \
      --gpu-ids 0,1,2 \
      [--limit N]   # quick smoke run on first N pages (single GPU recommended)
"""
from __future__ import annotations
import argparse
import json
import multiprocessing as mp
import os
import sys
import time
from pathlib import Path


def load_page_list(gt_json: str, images_dir: str, limit: int | None = None):
    pages = json.load(open(gt_json, encoding="utf-8"))
    if limit:
        pages = pages[:limit]
    return [(Path(p["page_info"]["image_path"]).stem,
             os.path.join(images_dir, p["page_info"]["image_path"])) for p in pages]


def shard(items, n):
    k = -(-len(items) // n)
    return [items[i:i + k] for i in range(0, len(items), k)]


def worker(gpu_id: int, chunk, args_dict: dict):
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    # Import heavy deps inside the worker (child pinned to one GPU).
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
    from hunyuan_ocr.backends.transformers import load_model_and_processor, infer_one
    from hunyuan_ocr.contract import CONTRACT

    a = argparse.Namespace(**args_dict)
    device = "cuda:0"
    print(f"[GPU {gpu_id}] loading model ...", flush=True)
    t0 = time.time()
    model, processor = load_model_and_processor(a.model, device=device)
    print(f"[GPU {gpu_id}] model ready in {time.time() - t0:.1f}s", flush=True)

    os.makedirs(a.pred_dir, exist_ok=True)
    todo = []
    for stem, img in chunk:
        if (Path(a.pred_dir) / f"{stem}.md").exists():
            continue
        todo.append((stem, img))
    print(f"[GPU {gpu_id}] {len(todo)} to do ({len(chunk) - len(todo)} resumed)", flush=True)

    for i, (stem, img) in enumerate(todo):
        try:
            md = infer_one(model, processor, img, CONTRACT.prompt, device=device)
            status = "ok"
        except Exception as e:
            md = f"ERROR: {type(e).__name__}: {e}"
            status = "failed"
        (Path(a.pred_dir) / f"{stem}.md").write_text(md, encoding="utf-8")
        if (i + 1) % 10 == 0:
            print(f"[GPU {gpu_id}] {i + 1}/{len(todo)} ({status})", flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--gt-json", required=True)
    p.add_argument("--images-dir", required=True)
    p.add_argument("--pred-dir", required=True)
    p.add_argument("--model", default="/root/models/HunyuanOCR")
    p.add_argument("--gpu-ids", default="0,1,2")
    p.add_argument("--limit", type=int, default=None)
    args = p.parse_args()

    pages = load_page_list(args.gt_json, args.images_dir, args.limit)
    gpu_ids = [int(x) for x in args.gpu_ids.split(",") if x.strip()]
    chunks = shard(pages, len(gpu_ids))
    print(f"[info] {len(pages)} pages across GPUs {gpu_ids}: {[len(c) for c in chunks]}", flush=True)

    ctx = mp.get_context("spawn")
    procs = [ctx.Process(target=worker, args=(gid, chunks[i], vars(args)), daemon=False)
             for i, gid in enumerate(gpu_ids)]
    for pr in procs:
        pr.start()
    for pr in procs:
        pr.join()
    print("[done] all workers finished", flush=True)


if __name__ == "__main__":
    main()
