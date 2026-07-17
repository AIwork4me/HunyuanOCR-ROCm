#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 AIwork4me
"""OpenAI-compatible driver: run HunyuanOCR-1.5 over OmniDocBench via a server.

Serves vLLM, llama.cpp (llama-server), or any OpenAI-compatible backend. One
``<stem>.md`` per page, written atomically; per-page errors to
``_errors/<stem>.json``. Resumable (skips only COMPLETE pages; FAILED/PENDING are
retried). Uses a circuit-breaking endpoint pool, pre-flight input validation,
and an exclusive prediction-directory lock. Exits non-zero on any page that ends
up FAILED/PENDING.

``run_openai_compatible.py`` is the same program under its generic name; this
file keeps the historical ``run_phase2_vllm`` name for existing users/Makefile.

Usage:
  # start servers first (one/GPU), then:
  python scripts/run_phase2_vllm.py --backend-name llamacpp \\
      --gt-json GT.json --images-dir images --pred-dir ./predictions \\
      --ports 8081,8082,8083,8084 --model HYVL --concurrency 16
"""

from __future__ import annotations
import argparse
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from openai import OpenAI  # noqa: E402

from hunyuan_ocr import preflight, runner  # noqa: E402
from hunyuan_ocr.backends.vllm_client import infer_one  # noqa: E402
from hunyuan_ocr.contract import CONTRACT  # noqa: E402
from hunyuan_ocr.endpoint_pool import EndpointPool, NoHealthyEndpoint  # noqa: E402

# Backends this driver knows how to label. ``openai`` allows any other
# OpenAI-compatible server not explicitly named.
SUPPORTED_BACKENDS = {"vllm", "llamacpp", "openai"}


def health_check(base_url: str) -> bool:
    """Probe an OpenAI-compatible server's ``/v1/models``. Monkeypatch in tests."""
    import requests

    try:
        return requests.get(f"{base_url}/models", timeout=10).status_code == 200
    except Exception:
        return False


def _endpoints(host: str, ports: list[int]) -> list[tuple[str, str]]:
    return [(f"port-{p}", f"http://{host}:{p}/v1") for p in ports]


def _provenance(args, endpoints) -> dict:
    return {
        "backend": args.backend_name,
        "protocol": args.protocol,
        "base_urls": [url for _, url in endpoints],  # no credentials
        "server_alias": args.server_alias,
        "host": args.host,
        "model": args.model,
        "decoding": {
            "max_tokens": 32768,
            "temperature": 0.0,
            "top_p": 1.0,
            "top_k": -1,
            "repetition_penalty": 1.08,
            "skip_special_tokens": True,
        },
    }


def main_with_args(argv):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--gt-json", required=True)
    p.add_argument("--images-dir", required=True)
    p.add_argument("--pred-dir", required=True)
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--ports", default="8000", help="comma-separated server ports")
    p.add_argument("--model", default="tencent/HunyuanOCR")
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--concurrency", type=int, default=24)
    p.add_argument("--max-pixels", type=int, default=0, help="client-side ViT cap (0 = uncapped)")
    p.add_argument("--max-retries", type=int, default=2)
    p.add_argument("--retry-backoff", type=float, default=2.0)
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--retry-failed", action="store_true", help="scope this run to FAILED pages only")
    p.add_argument(
        "--backend-name",
        default="vllm",
        choices=sorted(SUPPORTED_BACKENDS),
        help="label recorded in the manifest (vllm|llamacpp|openai)",
    )
    p.add_argument("--server-alias", default=None, help="free-form server label for the manifest (e.g. HYVL)")
    p.add_argument("--protocol", default="openai-chat-v1")
    p.add_argument(
        "--failure-threshold", type=int, default=3, help="consecutive failures before an endpoint's circuit opens"
    )
    p.add_argument(
        "--cooldown", type=float, default=30.0, help="seconds before an open circuit allows a half-open probe"
    )
    args = p.parse_args(argv)

    # --- pre-flight: fail before creating clients or loading models ------------
    problems = preflight.check_prediction_inputs(
        gt_json=args.gt_json,
        images_dir=args.images_dir,
        ports=args.ports,
        gpu_ids=None,
        concurrency=args.concurrency,
        max_retries=args.max_retries,
        retry_backoff=args.retry_backoff,
        max_pixels=args.max_pixels,
        model=args.model,
        pred_dir=args.pred_dir,
        backend_name=args.backend_name,
        allowed_backends=SUPPORTED_BACKENDS,
    )
    pages = preflight.pages_with_images(args.gt_json, args.images_dir)  # raises PreflightError
    preflight.assert_ok(problems)
    if args.limit:
        pages = pages[: args.limit]

    conflicts = runner.detect_stem_conflicts([img for _, img in pages])
    if conflicts:
        for stem, srcs in conflicts:
            print(f"[conflict] stem '{stem}' from {len(srcs)} images: {srcs}", file=sys.stderr)
        sys.exit("[fatal] output filename conflict(s); refusing to overwrite")

    ports = [int(x) for x in args.ports.split(",") if x.strip()]
    endpoints = _endpoints(args.host, ports)
    clients = {url: OpenAI(api_key="EMPTY", base_url=url, timeout=3600.0) for _, url in endpoints}
    pool = EndpointPool(
        endpoints, check=lambda url: health_check(url), failure_threshold=args.failure_threshold, cooldown=args.cooldown
    )
    pool.probe_initial()
    ep_summary = [(s["alias"], s["state"], s["initial_health"]) for s in pool.snapshot()]
    print(f"[info] endpoints: {ep_summary}", flush=True)

    if not pool.has_healthy():
        runner.write_run_manifest(
            args.pred_dir,
            backend=args.backend_name,
            model=args.model,
            run_counts={"attempted": 0, "succeeded": 0, "failed": 0, "skipped": 0},
            final_state={"expected": len(pages), "complete": 0, "failed": 0, "pending": len(pages)},
            backend_provenance=_provenance(args, endpoints),
            status="failed",
            extra={"endpoints": pool.snapshot()},
        )
        sys.exit(f"[fatal] no healthy endpoint among ports {ports}; aborting before dispatch (see manifest)")

    max_pixels = args.max_pixels or None
    todo, skipped = runner.select_todo(pages, args.pred_dir, overwrite=args.overwrite, retry_failed=args.retry_failed)
    print(f"[info] {len(todo)} to do ({skipped} skipped) across ports {ports}", flush=True)

    with runner.acquire_run_lock(args.pred_dir, command=["run_phase2", args.backend_name, str(args.pred_dir)]):

        def work(item):
            idx, (stem, img) = item
            last_exc, ep_url = None, None
            attempt = 0
            for attempt in range(1, args.max_retries + 1):
                try:
                    ep = pool.acquire()
                except NoHealthyEndpoint as e:
                    last_exc = e
                    break
                ep_url = ep.base_url
                try:
                    md = infer_one(clients[ep_url], img, CONTRACT.prompt, model=args.model, max_pixels=max_pixels)
                    pool.report(ep_url, True)
                    runner.commit_success(args.pred_dir, stem, md)
                    return {"stem": stem, "status": "complete"}
                except Exception as e:  # bounded retry; recorded if exhausted
                    pool.report(ep_url, False)
                    last_exc = e
                    if attempt < args.max_retries:
                        time.sleep(args.retry_backoff * (2 ** (attempt - 1)))
            runner.record_error(
                args.pred_dir,
                stem,
                image_path=img,
                backend=args.backend_name,
                endpoint=ep_url,
                exc=last_exc,
                attempt=attempt,
            )
            return {"stem": stem, "status": "failed", "error": str(last_exc)}

        results = []
        with ThreadPoolExecutor(max_workers=max(1, args.concurrency)) as ex:
            futs = [ex.submit(work, it) for it in enumerate(todo)]
            for f in as_completed(futs):
                res = f.result()  # propagates UNEXPECTED exception -> abort
                results.append(res)
                if len(results) % 20 == 0:
                    print(f"[info] {len(results)}/{len(todo)}", flush=True)

        runner.aggregate_errors(args.pred_dir)

        run_succeeded = sum(1 for r in results if r["status"] == "complete")
        run_failed = sum(1 for r in results if r["status"] == "failed")
        final_complete = sum(1 for s, _ in pages if runner.page_status(args.pred_dir, s) == "complete")
        final_failed = sum(1 for s, _ in pages if runner.page_status(args.pred_dir, s) == "failed")
        final_pending = len(pages) - final_complete - final_failed
        status = runner.decide_run_status(final_failed, final_pending)

        runner.write_run_manifest(
            args.pred_dir,
            backend=args.backend_name,
            model=args.model,
            run_counts={"attempted": len(todo), "succeeded": run_succeeded, "failed": run_failed, "skipped": skipped},
            final_state={
                "expected": len(pages),
                "complete": final_complete,
                "failed": final_failed,
                "pending": final_pending,
            },
            ports=ports,
            max_pixels=args.max_pixels,
            max_tokens=32768,
            status=status,
            backend_provenance=_provenance(args, endpoints),
            extra={"endpoints": pool.snapshot()},
        )
        print(
            f"[summary] expected={len(pages)} complete={final_complete} "
            f"failed={final_failed} pending={final_pending} skipped={skipped} "
            f"(this run: attempted={len(todo)} succeeded={run_succeeded} "
            f"failed={run_failed}) -> {args.pred_dir}",
            flush=True,
        )
        if status != "ok":
            sys.exit(f"[error] {final_failed} page(s) failed, {final_pending} pending; see _errors/")
    print("[done] all pages complete", flush=True)


def main():
    main_with_args(sys.argv[1:])


if __name__ == "__main__":
    main()
