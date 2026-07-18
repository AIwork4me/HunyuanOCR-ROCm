# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 AIwork4me
"""OpenAI-compatible prediction driver (package-resident, wheel-runnable).

Runs HunyuanOCR-1.5 over OmniDocBench via one or more OpenAI-compatible servers
(vLLM, llama.cpp ``llama-server``, or any OAI server). Writes one ``<stem>.md``
per page atomically, records per-page errors to ``_errors/<stem>.json``, is
resumable (skips only COMPLETE pages; FAILED/PENDING are retried), and uses a
circuit-breaking endpoint pool plus an exclusive prediction-directory lock.

Crash-safe: an unexpected worker exception, ``KeyboardInterrupt``, endpoint-pool
fatal, or executor error is captured (never escapes) and the run manifest is
**always** written with a terminal status of ``ok`` / ``failed`` / ``crashed`` /
``interrupted``. The normal success path is unchanged.

The health check, the OpenAI client factory, and the per-image inference
callable are **injected**, so the whole driver is unit-testable on a CPU with
fakes (no network, no GPU, no torch). ``scripts/run_phase2_vllm.py`` is a thin
wrapper that supplies the real callables; ``hunyuan-ocr predict`` calls this
module directly so the CLI works from a wheel install.
"""

from __future__ import annotations

import argparse
import sys
import time
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed

from hunyuan_ocr import preflight, runner
from hunyuan_ocr.contract import CONTRACT
from hunyuan_ocr.endpoint_pool import EndpointPool, NoHealthyEndpoint

# Backends this driver knows how to label. ``openai`` allows any other
# OpenAI-compatible server not explicitly named.
SUPPORTED_BACKENDS = {"vllm", "llamacpp", "openai"}

MAX_TOKENS = 32768


def health_check(base_url: str) -> bool:
    """Probe an OpenAI-compatible server's ``/v1/models``. Override in tests."""
    import requests

    try:
        return requests.get(f"{base_url}/models", timeout=10).status_code == 200
    except Exception:  # noqa: BLE001
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
            "max_tokens": MAX_TOKENS,
            "temperature": 0.0,
            "top_p": 1.0,
            "top_k": -1,
            "repetition_penalty": 1.08,
            "skip_special_tokens": True,
        },
    }


def parse_args(argv) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="run_inference",
        description="OpenAI-compatible driver: run HunyuanOCR-1.5 over OmniDocBench via a server.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
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
    return p.parse_args(argv)


def _crash_record(exc, kind: str) -> dict:
    """Structured record of a terminal exception. No secrets (args are redacted
    elsewhere; exception text is truncated). ``kind`` is ``crashed`` or
    ``interrupted``."""
    tb = traceback.format_exception(type(exc), exc, exc.__traceback__)
    return {
        "kind": kind,
        "exception_type": type(exc).__name__,
        "exception_message": str(exc)[:1000],
        # tail only — enough to diagnose, not enough to leak volumes of context
        "traceback_tail": "".join(tb)[-2000:],
    }


def run_workers(todo, concurrency: int, work) -> tuple[list[dict], dict | None]:
    """Submit ``todo`` to a thread pool calling ``work(indexed_item)`` and collect
    results. Any unexpected exception (worker bug, pool fatal, executor error) or
    ``KeyboardInterrupt`` is captured into a crash record and returned — it never
    propagates, so the caller can always finalize the manifest. Returns
    ``(partial_results, crash_record_or_None)``.
    """
    results: list[dict] = []
    crash: dict | None = None
    try:
        with ThreadPoolExecutor(max_workers=max(1, concurrency)) as ex:
            futs = [ex.submit(work, it) for it in enumerate(todo)]
            for f in as_completed(futs):
                results.append(f.result())  # propagates only UNEXPECTED exceptions
                if len(results) % 20 == 0:
                    print(f"[info] {len(results)}/{len(todo)}", flush=True)
    except KeyboardInterrupt as exc:
        crash = _crash_record(exc, "interrupted")
    except Exception as exc:  # noqa: BLE001 — endpoint-pool fatal / executor / worker-bug
        crash = _crash_record(exc, "crashed")
    return results, crash


def finalize(args, pool, endpoints, pages, todo, skipped, results, crash, ports) -> str:
    """Always-callable finalization: aggregate errors, recount from disk, and write
    the run manifest with a terminal status. Returns the status string."""
    runner.aggregate_errors(args.pred_dir)
    run_succeeded = sum(1 for r in results if r.get("status") == "complete")
    run_failed = sum(1 for r in results if r.get("status") == "failed")
    # pages dispatched this run whose outcome is unresolved (only > 0 on a crash)
    interrupted = max(0, len(todo) - run_succeeded - run_failed)
    final_complete = sum(1 for s, _ in pages if runner.page_status(args.pred_dir, s) == "complete")
    final_failed = sum(1 for s, _ in pages if runner.page_status(args.pred_dir, s) == "failed")
    final_pending = len(pages) - final_complete - final_failed
    if crash:
        status = crash["kind"]  # "crashed" | "interrupted"
    else:
        status = runner.decide_run_status(final_failed, final_pending)

    extensions = {"endpoints": pool.snapshot()}
    if crash:
        extensions["crash"] = crash
    runner.write_run_manifest(
        args.pred_dir,
        backend=args.backend_name,
        model=args.model,
        run_counts={
            "attempted": len(todo),
            "succeeded": run_succeeded,
            "failed": run_failed,
            "skipped": skipped,
            "interrupted": interrupted,
        },
        final_state={
            "expected": len(pages),
            "complete": final_complete,
            "failed": final_failed,
            "pending": final_pending,
        },
        ports=ports,
        max_pixels=args.max_pixels,
        max_tokens=MAX_TOKENS,
        status=status,
        backend_provenance=_provenance(args, endpoints),
        extra=extensions,
    )
    print(
        f"[summary] expected={len(pages)} complete={final_complete} "
        f"failed={final_failed} pending={final_pending} skipped={skipped} "
        f"(this run: attempted={len(todo)} succeeded={run_succeeded} "
        f"failed={run_failed} interrupted={interrupted}) -> {args.pred_dir}",
        flush=True,
    )
    return status


def dispatch(args, *, infer_one, client_factory, health_check_fn=health_check) -> int:
    """Run predict -> (the manifest is always written). Returns an exit code
    (0 == ok). The caller (script or CLI) decides whether to ``sys.exit`` on
    non-zero. ``infer_one``/``client_factory``/``health_check_fn`` are injected so
    tests can drive this with fakes on a CPU.

    Raises ``PreflightError`` (bad args) which callers normally let propagate as a
    hard failure; everything else is captured into the manifest.
    """
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
        print("[fatal] output filename conflict(s); refusing to overwrite", file=sys.stderr)
        return 1

    ports = [int(x) for x in args.ports.split(",") if x.strip()]
    endpoints = _endpoints(args.host, ports)
    clients = {url: client_factory(api_key="EMPTY", base_url=url, timeout=3600.0) for _, url in endpoints}
    pool = EndpointPool(
        endpoints, check=health_check_fn, failure_threshold=args.failure_threshold, cooldown=args.cooldown
    )
    pool.probe_initial()
    ep_summary = [(s["alias"], s["state"], s["initial_health"]) for s in pool.snapshot()]
    print(f"[info] endpoints: {ep_summary}", flush=True)

    if not pool.has_healthy():
        # No server is up: write a failed manifest with every page pending and stop.
        # (skipped = expected so the conservation laws balance for this no-op run.)
        runner.write_run_manifest(
            args.pred_dir,
            backend=args.backend_name,
            model=args.model,
            run_counts={"attempted": 0, "succeeded": 0, "failed": 0, "skipped": len(pages), "interrupted": 0},
            final_state={"expected": len(pages), "complete": 0, "failed": 0, "pending": len(pages)},
            ports=ports,
            max_pixels=args.max_pixels,
            max_tokens=MAX_TOKENS,
            status="failed",
            backend_provenance=_provenance(args, endpoints),
            extra={"endpoints": pool.snapshot()},
        )
        print(
            f"[fatal] no healthy endpoint among ports {ports}; aborting before dispatch (see manifest)",
            file=sys.stderr,
        )
        return 1

    max_pixels = args.max_pixels or None
    todo, skipped = runner.select_todo(pages, args.pred_dir, overwrite=args.overwrite, retry_failed=args.retry_failed)
    print(f"[info] {len(todo)} to do ({skipped} skipped) across ports {ports}", flush=True)

    def work(item):
        idx, (stem, img) = item
        del idx
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

    crash = None
    results: list[dict] = []
    try:
        with runner.acquire_run_lock(args.pred_dir, command=["run_phase2", args.backend_name, str(args.pred_dir)]):
            results, crash = run_workers(todo, args.concurrency, work)
            # Finalize WHILE the lock is held, so the manifest reflects this run's
            # writes and never races a concurrent writer. Runs even if the workers
            # crashed (crash is then encoded in the manifest status + extensions).
            status = finalize(args, pool, endpoints, pages, todo, skipped, results, crash, ports)
    except runner.RunLockHeld as exc:
        print(f"[fatal] {exc}", file=sys.stderr)
        return 1

    if status != "ok":
        final_failed = sum(1 for s, _ in pages if runner.page_status(args.pred_dir, s) == "failed")
        final_pending = len(pages) - sum(
            1 for s, _ in pages if runner.page_status(args.pred_dir, s) in ("complete", "failed")
        )
        print(
            f"[error] run did not complete cleanly (status={status}): "
            f"{final_failed} failed, {final_pending} pending; see _errors/ and run_manifest.json",
            file=sys.stderr,
        )
        return 1
    print("[done] all pages complete", flush=True)
    return 0
