# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 AIwork4me
"""OmniDocBench-ROCm standard CLI contract surface (ADR-0011).

Locked to central commit ``ccd466ef317fd6a710131db3a19ec9d55a65ce2e``
(see ``.rocmdoc/spec-lock.json``). Four JSON subcommands that share ONE inference
core with the expert commands (``predict``) and the adapter:

    hunyuan-ocr version        --json
    hunyuan-ocr capabilities   --json
    hunyuan-ocr doctor         --json      (enhanced by cli._doctor)
    hunyuan-ocr parse --img-dir D --out-dir O --platform P [--backend B] [--json]

``--json`` prints EXACTLY one JSON document to stdout; every log/warning goes to
stderr. Exit codes are normative and match ``contracts/cli-contract.md`` @ the
locked commit:

    0 OK · 1 PARTIAL · 2 USAGE · 3 BACKEND_MISMATCH · 4 CONTRACT · 5 FATAL

The inference core reused here is :func:`hunyuan_ocr.backends.vllm_client.infer_one`
— the SAME function the expert ``predict`` path and the adapter use, so the CLI
and the OmniDocBench adapter never maintain two inference implementations.
"""

from __future__ import annotations

import json
import os
import sys
import time
from collections.abc import Callable
from pathlib import Path

# Exit codes (mirror omnidocbench_rocm.cli_contract @ ccd466e; frozen by contract).
EXIT_OK = 0
EXIT_PARTIAL = 1
EXIT_USAGE = 2
EXIT_BACKEND_MISMATCH = 3
EXIT_CONTRACT = 4
EXIT_FATAL = 5

SCHEMA_VERSION = 1
NAME = "hunyuan-ocr"

# Declared capabilities. MUST agree with rocmdoc.yaml — `omnidocbench-rocm manifest
# rocmdoc.yaml --card model_card_v2.json` enforces result↔capability alignment.
DECLARED_PLATFORMS = [
    {"platform": "linux-rocm", "backend": "vllm", "precision": "bf16", "interface": "standard-cli"},
    {"platform": "linux-rocm", "backend": "llama-cpp", "precision": "bf16", "interface": "standard-cli"},
    {"platform": "linux-rocm", "backend": "transformers", "precision": "bf16", "interface": "adapter-script"},
    {"platform": "windows-hip", "backend": "llama-cpp", "precision": "bf16", "interface": "adapter-script"},
]
DECLARED_INTERFACES = ["standard-cli", "adapter-script"]

# The default model id for the OpenAI-compatible backends.
DEFAULT_MODEL = os.environ.get("HUNYUANOCR_MODEL_NAME", "tencent/HunyuanOCR")

# Image extensions recognized by `parse` (sorted, deterministic).
_IMAGE_EXTS = (".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff")


def emit_json(obj: dict) -> None:
    """Print exactly one JSON document to stdout (contract: no banners/logs)."""
    sys.stdout.write(json.dumps(obj, indent=2, ensure_ascii=False) + "\n")


def _stderr(msg: str) -> None:
    print(msg, file=sys.stderr)


def cmd_version() -> int:
    from hunyuan_ocr import __version__

    emit_json(
        {
            "name": NAME,
            "version": __version__,
            "engine_version": "vllm 0.16.1 / llama.cpp 0.16.0 / transformers 5.13.0 (declared)",
            "schema_version": SCHEMA_VERSION,
            "central_spec_commit": "ccd466ef317fd6a710131db3a19ec9d55a65ce2e",
            "cli_contract": "omnidocbench-rocm cli-contract.md (ADR-0011)",
        }
    )
    return EXIT_OK


def cmd_capabilities() -> int:
    emit_json({"platforms": DECLARED_PLATFORMS, "interfaces": DECLARED_INTERFACES})
    return EXIT_OK


# --- parse -------------------------------------------------------------------
#
# `parse` reuses the SAME per-image inference core (infer_one) as the expert
# `predict` command and the adapter. The infer callable is a module attribute so
# tests can swap in a stub WITHOUT touching production code paths (no fixture
# shortcut lives in the production infer path).


def _real_infer(client, image_path: str, *, model: str, max_pixels: int | None) -> str:
    """Production inference: delegate to the shared vLLM/llama.cpp infer_one."""
    from hunyuan_ocr import contract as _contract
    from hunyuan_ocr.backends.vllm_client import infer_one

    return infer_one(client, image_path, _contract.CONTRACT.prompt, model=model, max_pixels=max_pixels)


# Swappable for tests. Signature mirrors _real_infer so a stub is drop-in.
INFER: Callable[..., str] = _real_infer


def _list_images(img_dir: Path) -> list[Path]:
    return sorted(p for p in img_dir.iterdir() if p.suffix.lower() in _IMAGE_EXTS and p.is_file())


def _resolve_server(backend: str, server_url: str | None) -> str:
    url = server_url or os.environ.get("HUNYUANOCR_SERVER_URL") or os.environ.get("VLLM_SERVER_URL")
    if not url:
        raise _UsageError(
            f"no server URL for backend {backend!r}: pass --server-url or set HUNYUANOCR_SERVER_URL "
            "(parse needs a running OpenAI-compatible server; see `hunyuan-ocr doctor`)"
        )
    return url


class _UsageError(Exception):
    pass


def _build_client(server_url: str):
    try:
        from openai import OpenAI  # local import: client is an opt-in extra
    except ImportError as exc:  # pragma: no cover - exercised via doctor
        raise _UsageError('openai client not installed: pip install ".[client]"') from exc
    return OpenAI(api_key="EMPTY", base_url=server_url, timeout=3600.0)


def cmd_parse(
    *,
    img_dir: Path,
    out_dir: Path,
    platform: str,
    backend: str | None,
    server_url: str | None,
    model: str | None,
    max_pixels: int | None,
    limit: int | None,
) -> int:
    """Parse every image in ``img_dir`` to ``<out_dir>/<stem>.md``; emit cli_result.

    R2 robustness: a per-page failure is recorded and the run CONTINUES (never
    raises). >=1 failed page -> status ``partial`` + exit 1. Backend mismatch
    (requested != the one that actually ran) -> exit 3. ``page_count`` always
    equals the number of input images (full-set honesty).
    """
    backend = (backend or "vllm").lower()
    # transformers is a separate GPU-only driver (repo checkout + ROCm torch);
    # it is not reachable from the standard parse path.
    if backend == "transformers":
        _stderr(
            "parse does not serve the transformers backend (GPU-only repo driver); use `predict --backend transformers`."
        )
        return EXIT_USAGE

    if not img_dir.is_dir():
        _stderr(f"img-dir not found: {img_dir}")
        return EXIT_USAGE

    images = _list_images(img_dir)
    if limit is not None:
        images = images[:limit]
    out_dir.mkdir(parents=True, exist_ok=True)

    # The backend that actually runs is the OpenAI-compatible server backend
    # (vllm/llama-cpp/openai). We report what ran honestly.
    actual_backend = backend

    try:
        url = _resolve_server(backend, server_url)
    except _UsageError as exc:
        _stderr(str(exc))
        return EXIT_USAGE

    try:
        client = _build_client(url)
    except _UsageError as exc:
        _stderr(str(exc))
        return EXIT_USAGE

    mdl = model or DEFAULT_MODEL
    pages: list[dict] = []
    ok = failed = 0
    for img in images:
        rel = img.name
        md_path = out_dir / (img.stem + ".md")
        t0 = time.monotonic()
        try:
            md = INFER(client, str(img), model=mdl, max_pixels=max_pixels)
            if not isinstance(md, str) or not md.strip():
                raise RuntimeError("empty prediction")
            md_path.write_text(md, encoding="utf-8")
            pages.append({"image": rel, "status": "ok", "seconds": round(time.monotonic() - t0, 3)})
            ok += 1
        except Exception as exc:  # R2: per-page failure caught, run continues
            if md_path.exists():
                try:
                    md_path.unlink()  # never leave an empty/corrupt .md
                except OSError:
                    pass
            pages.append({"image": rel, "status": "failed", "error": str(exc)[:300]})
            failed += 1

    status = "ok" if failed == 0 else ("partial" if ok > 0 else "failed")
    # Conservation invariant: every input page is accounted for exactly once.
    assert len(pages) == len(images), "page conservation violated"

    result = {
        "schema_version": SCHEMA_VERSION,
        "status": status,
        "backend": actual_backend,
        "engine": actual_backend,
        "page_count": len(images),
        "ok": ok,
        "failed": failed,
        "skipped": 0,
        "output_dir": str(out_dir),
        "full_set": limit is None,
        "pages": pages,
        "requested_backend": backend,
    }
    emit_json(result)
    if status == "ok":
        return EXIT_OK
    if status == "partial":
        return EXIT_PARTIAL
    return EXIT_FATAL  # all pages failed
