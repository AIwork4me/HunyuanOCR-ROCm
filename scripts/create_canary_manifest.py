#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 AIwork4me
"""Generate a verifiable canary manifest from an OmniDocBench GT json.

Emits JSON: subset name, source dataset/version, expected_count, sorted page
stems + image_paths, source JSON SHA256, manifest SHA256. Deterministic.

The manifest_sha256 is the sha256 of the canonical JSON of the manifest WITH
that field omitted (so a reader can drop it and recompute to verify integrity).

Usage:
  python scripts/create_canary_manifest.py \
      --gt-json /workspace/OmniDocBench_data/OmniDocBench_150.json \
      --name canary-148 --dataset OmniDocBench --dataset-version v1.6 \
      --out eval/canary_148.manifest.json
"""

from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path


def sha256_file(p) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def build_manifest(gt_json, *, name, dataset, dataset_version) -> dict:
    pages = json.load(open(gt_json, encoding="utf-8"))
    # Preserve the SOURCE FILE ORDER (not sorted): the order is load-bearing —
    # `hunyuan_ocr.canary.materialize` rebuilds the subset byte-identically from
    # the full GT by selecting pages in this exact order.
    entries = [(Path(p["page_info"]["image_path"]).stem, p["page_info"]["image_path"]) for p in pages]
    return {
        "subset_name": name,
        "source_dataset": dataset,
        "source_dataset_version": dataset_version,
        "serialization": "json_compact_utf8",  # json.dumps(subset, ensure_ascii=False)
        "expected_count": len(entries),
        "pages": [{"stem": s, "image_path": ip} for s, ip in entries],
        "source_json_sha256": sha256_file(gt_json),
    }


def manifest_sha256(d: dict) -> str:
    body = {k: v for k, v in d.items() if k != "manifest_sha256"}
    text = json.dumps(body, ensure_ascii=False, sort_keys=True, indent=2)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--gt-json", required=True)
    p.add_argument("--name", required=True)
    p.add_argument("--dataset", default="OmniDocBench")
    p.add_argument("--dataset-version", default="v1.6")
    p.add_argument("--out", required=True)
    args = p.parse_args()

    m = build_manifest(args.gt_json, name=args.name, dataset=args.dataset, dataset_version=args.dataset_version)
    m["manifest_sha256"] = manifest_sha256(m)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(m, ensure_ascii=False, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    print(
        f"wrote {out}: {m['expected_count']} pages, "
        f"source_sha256={m['source_json_sha256'][:12]}..., "
        f"manifest_sha256={m['manifest_sha256'][:12]}..."
    )


if __name__ == "__main__":
    main()
