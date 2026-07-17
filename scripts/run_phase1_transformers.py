#!/usr/bin/env python3
"""Phase-1 driver: run HunyuanOCR-1.5 (transformers) over OmniDocBench pages.

One spawned worker process per GPU, sharded. One <stem>.md per page, written
atomically; errors to _errors/<stem>.json. Resumable. Exits non-zero on any
worker crash, model-load failure, or page that ends up FAILED/PENDING.

Usage:
  python scripts/run_phase1_transformers.py --gt-json GT.json --images-dir images \
      --pred-dir ./predictions --model /path/to/HunyuanOCR --gpu-ids 0,1,2 [--limit N]
"""
from __future__ import annotations
import argparse
import json
import multiprocessing as mp
import os
import queue as _queue
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from hunyuan_ocr import runner  # noqa: E402


def _load_page_list(gt_json, images_dir, limit=None):
    pages = json.load(open(gt_json, encoding="utf-8"))
    if limit:
        pages = pages[:limit]
    return [(Path(p["page_info"]["image_path"]).stem,
             os.path.join(images_dir, p["page_info"]["image_path"])) for p in pages]


def _shard(items, n):
    k = -(-len(items) // n)
    return [items[i:i + k] for i in range(0, len(items), k)]


def _worker(gpu_id, chunk, args_dict, out_q):
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
    try:
        from hunyuan_ocr.backends.transformers import load_model_and_processor, infer_one
        from hunyuan_ocr.contract import CONTRACT
    except Exception as e:  # import failure (e.g. missing torch)
        out_q.put({"gpu": gpu_id, "kind": "worker_error", "msg": f"import failed: {e}"})
        return
    a = argparse.Namespace(**args_dict)
    try:
        print(f"[GPU {gpu_id}] loading model ...", flush=True)
        t0 = time.time()
        model, processor = load_model_and_processor(a.model, device="cuda:0")
        print(f"[GPU {gpu_id}] model ready in {time.time()-t0:.1f}s", flush=True)
    except Exception as e:
        out_q.put({"gpu": gpu_id, "kind": "worker_error",
                   "msg": f"model load failed: {type(e).__name__}: {e}"})
        return

    os.makedirs(a.pred_dir, exist_ok=True)
    todo, skipped = runner.select_todo(chunk, a.pred_dir,
                                       overwrite=a.overwrite, retry_failed=a.retry_failed)
    for _ in range(skipped):
        out_q.put({"gpu": gpu_id, "kind": "skip"})

    for stem, img in todo:
        last_exc, attempt = None, 0
        for attempt in range(1, a.max_retries + 1):
            try:
                md = infer_one(model, processor, img, CONTRACT.prompt, device="cuda:0")
                runner.commit_success(a.pred_dir, stem, md)
                out_q.put({"gpu": gpu_id, "kind": "complete", "stem": stem})
                break
            except Exception as e:
                last_exc = e
                if attempt < a.max_retries:
                    time.sleep(a.retry_backoff * (2 ** (attempt - 1)))
        else:
            runner.record_error(a.pred_dir, stem, image_path=img, backend="transformers",
                                endpoint=f"gpu{gpu_id}", exc=last_exc, attempt=attempt)
            out_q.put({"gpu": gpu_id, "kind": "failed", "stem": stem})
    out_q.put({"gpu": gpu_id, "kind": "worker_done"})


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--gt-json", required=True)
    p.add_argument("--images-dir", required=True)
    p.add_argument("--pred-dir", required=True)
    p.add_argument("--model", default="/root/models/HunyuanOCR")
    p.add_argument("--gpu-ids", default="0,1,2")
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--max-retries", type=int, default=2)
    p.add_argument("--retry-backoff", type=float, default=2.0)
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--retry-failed", action="store_true")
    args = p.parse_args()

    pages = _load_page_list(args.gt_json, args.images_dir, args.limit)
    conflicts = runner.detect_stem_conflicts([img for _, img in pages])
    if conflicts:
        for stem, srcs in conflicts:
            print(f"[conflict] stem '{stem}' from {len(srcs)} images: {srcs}", file=sys.stderr)
        sys.exit("[fatal] output filename conflict(s)")

    os.makedirs(args.pred_dir, exist_ok=True)
    gpu_ids = [int(x) for x in args.gpu_ids.split(",") if x.strip()]
    chunks = _shard(pages, len(gpu_ids))
    print(f"[info] {len(pages)} pages across GPUs {gpu_ids}: {[len(c) for c in chunks]}", flush=True)

    ctx = mp.get_context("spawn")
    out_q = ctx.Queue()
    procs = [ctx.Process(target=_worker, args=(gid, chunks[i], vars(args), out_q), daemon=False)
             for i, gid in enumerate(gpu_ids)]
    for pr in procs:
        pr.start()

    n_done = 0
    worker_errors = []
    counts = {"complete": 0, "failed": 0, "skip": 0}

    def _handle(msg):
        nonlocal n_done
        k = msg.get("kind")
        if k == "worker_done":
            n_done += 1
        elif k in counts:
            counts[k] += 1
        elif k == "worker_error":
            worker_errors.append(msg)

    # drain while workers run; stop when all reported done OR all dead (crash)
    while n_done < len(procs) and any(pr.is_alive() for pr in procs):
        try:
            _handle(out_q.get(timeout=0.5))
        except _queue.Empty:
            continue
    # best-effort final drain
    try:
        while True:
            _handle(out_q.get_nowait())
    except _queue.Empty:
        pass
    for pr in procs:
        pr.join()
    crashed = [pr for pr in procs if pr.exitcode not in (0, None)]

    runner.aggregate_errors(args.pred_dir)

    final_complete = sum(1 for s, _ in pages if runner.page_status(args.pred_dir, s) == "complete")
    final_failed = sum(1 for s, _ in pages if runner.page_status(args.pred_dir, s) == "failed")
    final_pending = len(pages) - final_complete - final_failed
    status = runner.decide_run_status(final_failed, final_pending,
                                      worker_errors=len(worker_errors), crashed=len(crashed))
    runner.write_run_manifest(args.pred_dir, backend="transformers", model=args.model,
                              counts={"expected": len(pages), "succeeded": final_complete,
                                      "failed": final_failed, "skipped": counts["skip"]},
                              gpu_ids=gpu_ids, status=status)
    print(f"[summary] expected={len(pages)} complete={final_complete} failed={final_failed} "
          f"pending={final_pending} worker_errors={len(worker_errors)} crashed={len(crashed)}",
          flush=True)
    for e in worker_errors:
        print(f"[worker_error] GPU {e['gpu']}: {e['msg']}", file=sys.stderr)
    if status != "ok":
        sys.exit(f"[error] run failed: {final_failed} failed, {final_pending} pending, "
                 f"{len(worker_errors)} worker errors, {len(crashed)} crashed")
    print("[done] all pages complete", flush=True)


if __name__ == "__main__":
    main()
