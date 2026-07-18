# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 AIwork4me
"""Materialize the 148-page canary subset from the full OmniDocBench GT.

The committed ``eval/canary_148.manifest.json`` is the source of truth for the
canary: it lists the pages IN FILE ORDER and records the source-GT SHA256. This
module reconstructs ``OmniDocBench_150.json`` byte-identically from the full GT
by selecting pages in manifest order and re-serializing compactly — so the
canary can be regenerated without committing the (large) subset, and the result
is verifiable against the recorded SHA256.

No GPU, no model deps. Pure stdlib.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


class CanaryError(ValueError):
    """Raised when the canary cannot be materialized exactly."""


def materialize(full_gt, manifest_path, out_path) -> str:
    """Rebuild the canary subset from the full GT using the manifest's page order.

    Selects full-GT pages by ``image_path`` in the manifest's recorded order,
    re-serializes with ``json.dumps(..., ensure_ascii=False)`` (compact, matching
    the original subset), writes to ``out_path``, and verifies the SHA256 equals
    the manifest's ``source_json_sha256``. Returns the written SHA256.

    Raises CanaryError on missing/unreadable inputs, missing pages, duplicate
    image_paths, count mismatch, or a SHA256 mismatch (materialization not
    byte-identical). All error paths raise CanaryError so callers never see a
    raw FileNotFoundError/JSONDecodeError traceback.
    """
    try:
        manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise CanaryError(f"manifest not found: {manifest_path}") from exc
    except (OSError, json.JSONDecodeError) as exc:
        raise CanaryError(f"manifest not readable/parseable: {manifest_path}: {exc}") from exc
    try:
        full = json.loads(Path(full_gt).read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise CanaryError(f"full GT not found: {full_gt}") from exc
    except (OSError, json.JSONDecodeError) as exc:
        raise CanaryError(f"full GT not readable/parseable: {full_gt}: {exc}") from exc
    by_image = {p["page_info"]["image_path"]: p for p in full}

    order = [pg["image_path"] for pg in manifest["pages"]]
    if len(order) != len(set(order)):
        raise CanaryError("manifest contains duplicate image_paths")
    expected = manifest.get("expected_count")
    if expected is not None and expected != len(order):
        raise CanaryError(f"expected_count={expected} but manifest lists {len(order)} pages")

    missing = [ip for ip in order if ip not in by_image]
    if missing:
        raise CanaryError(f"{len(missing)} manifest page(s) not found in full GT; first: {missing[0]}")

    subset = [by_image[ip] for ip in order]
    blob = json.dumps(subset, ensure_ascii=False).encode("utf-8")
    digest = hashlib.sha256(blob).hexdigest()

    recorded = manifest.get("source_json_sha256")
    if recorded and digest != recorded:
        raise CanaryError(
            f"materialized canary SHA256 {digest[:12]}... != recorded {recorded[:12]}... "
            "(not byte-identical; the full GT or serialization differs from the locked subset)"
        )

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    Path(out_path).write_bytes(blob)
    return digest
