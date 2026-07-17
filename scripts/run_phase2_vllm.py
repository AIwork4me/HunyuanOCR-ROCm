#!/usr/bin/env python3
"""Phase-2 driver: run HunyuanOCR-1.5 via an OpenAI-compatible server over OmniDocBench.

One <stem>.md per page, written atomically; errors recorded to _errors/<stem>.json.
Resumable (skips only COMPLETE pages; FAILED/PENDING are retried). Exits non-zero
on any page that ends up FAILED or PENDING, or on any unhandled worker exception.

Usage:
  # start servers first (one/GPU), then:
  python scripts/run_phase2_vllm.py --gt-json GT.json --images-dir images \
      --pred-dir ./predictions --ports 8081,8082,8083,8084 --model HYVL --concurrency 16
"""
from __future__ import annotations
import argparse
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from openai import OpenAI  # noqa: E402

from hunyuan_ocr import runner  # noqa: E402
from hunyuan_ocr.backends.vllm_client import infer_one  # noqa: E402
from hunyuan_ocr.contract import CONTRACT  # noqa: E402


def _load_pages(gt_json, images_dir, limit=0):
    pages = json.load(open(gt_json, encoding="utf-8"))
    if limit:
        pages = pages[:limit]
    out = []
    for pg in pages:
        rel = pg["page_info"]["image_path"]
        out.append((Path(rel).stem, os.path.join(images_dir, rel)))
    return out


def main_with_args(argv):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--gt-json", required=True)
    p.add_argument("--images-dir", required=True)
    p.add_argument("--pred-dir", required=True)
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--ports", default="8000", help="comma-separated server ports")
    p.add_argument("--model", default="tencent/HunyuanOCR")
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--concurrency", type=int, default=24)
    p.add_argument("--max-pixels", type=int, default=0,
                   help="client-side ViT cap (0 = uncapped)")
    p.add_argument("--max-retries", type=int, default=2)
    p.add_argument("--retry-backoff", type=float, default=2.0)
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--retry-failed", action="store_true",
                   help="scope this run to FAILED pages only")
    args = p.parse_args(argv)

    pages = _load_pages(args.gt_json, args.images_dir, args.limit)
    os.makedirs(args.pred_dir, exist_ok=True)

    conflicts = runner.detect_stem_conflicts([img for _, img in pages])
    if conflicts:
        for stem, srcs in conflicts:
            print(f"[conflict] stem '{stem}' from {len(srcs)} images: {srcs}", file=sys.stderr)
        sys.exit("[fatal] output filename conflict(s); refusing to overwrite")

    ports = [int(x) for x in args.ports.split(",") if x.strip()]
    clients = [OpenAI(api_key="EMPTY", base_url=f"http://{args.host}:{pt}/v1", timeout=3600.0)
               for pt in ports]
    max_pixels = args.max_pixels or None

    todo, skipped = runner.select_todo(pages, args.pred_dir,
                                       overwrite=args.overwrite,
                                       retry_failed=args.retry_failed)
    print(f"[info] {len(todo)} to do ({skipped} skipped) across ports {ports}", flush=True)

    def work(item):
        idx, (stem, img) = item
        last_exc, ep = None, f"{args.host}:{ports[idx % len(ports)]}"
        attempt = 0
        for attempt in range(1, args.max_retries + 1):
            client = clients[(idx + attempt - 1) % len(clients)]
            ep = f"{args.host}:{ports[(idx + attempt - 1) % len(ports)]}"
            try:
                md = infer_one(client, img, CONTRACT.prompt,
                               model=args.model, max_pixels=max_pixels)
                runner.commit_success(args.pred_dir, stem, md)
                return {"stem": stem, "status": "complete"}
            except Exception as e:  # bounded retry; recorded if exhausted
                last_exc = e
                if attempt < args.max_retries:
                    time.sleep(args.retry_backoff * (2 ** (attempt - 1)))
        runner.record_error(args.pred_dir, stem, image_path=img, backend="vllm",
                            endpoint=ep, exc=last_exc, attempt=attempt)
        return {"stem": stem, "status": "failed", "error": str(last_exc)}

    results = []
    with ThreadPoolExecutor(max_workers=max(1, args.concurrency)) as ex:
        futs = [ex.submit(work, it) for it in enumerate(todo)]
        for f in as_completed(futs):
            res = f.result()  # propagates any UNEXPECTED exception to main -> abort
            results.append(res)
            if len(results) % 20 == 0:
                print(f"[info] {len(results)}/{len(todo)}", flush=True)

    runner.aggregate_errors(args.pred_dir)

    final_complete = sum(1 for s, _ in pages if runner.page_status(args.pred_dir, s) == "complete")
    final_failed = sum(1 for s, _ in pages if runner.page_status(args.pred_dir, s) == "failed")
    final_pending = len(pages) - final_complete - final_failed
    status = runner.decide_run_status(final_failed, final_pending)

    runner.write_run_manifest(args.pred_dir, backend="vllm", model=args.model,
                              counts={"expected": len(pages), "succeeded": final_complete,
                                      "failed": final_failed, "skipped": skipped},
                              ports=ports, max_pixels=args.max_pixels,
                              max_tokens=32768, status=status)
    print(f"[summary] expected={len(pages)} complete={final_complete} failed={final_failed} "
          f"pending={final_pending} skipped={skipped} -> {args.pred_dir}", flush=True)
    if status != "ok":
        sys.exit(f"[error] {final_failed} page(s) failed, {final_pending} pending; see _errors/")
    print("[done] all pages complete", flush=True)


def main():
    main_with_args(sys.argv[1:])


if __name__ == "__main__":
    main()
