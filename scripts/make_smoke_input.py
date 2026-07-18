#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 AIwork4me
"""Materialize a deterministic 1-page OmniDocBench GT for the GPU smoke, using the
canary manifest's first page. Output + images are machine-local (gitignored)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def select_one_page(full_gt: list[dict], manifest: dict, images_dir: Path) -> dict:
    """Return the full-GT page whose image_path is the manifest's first page;
    raise FileNotFoundError if that page or its image is missing."""
    order = [p["image_path"] for p in manifest["pages"]]
    target = order[0]
    by_path = {p["page_info"]["image_path"]: p for p in full_gt}
    if target not in by_path:
        raise FileNotFoundError(f"manifest first page {target!r} not in full GT")
    if not (images_dir / target).is_file():
        raise FileNotFoundError(f"image not found under {images_dir}: {target}")
    return by_path[target]


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--full-gt", required=True)
    p.add_argument("--manifest", required=True)
    p.add_argument("--images-dir", required=True)
    p.add_argument("--out", required=True)
    args = p.parse_args(argv)

    full = json.loads(Path(args.full_gt).read_text(encoding="utf-8"))
    manifest = json.loads(Path(args.manifest).read_text(encoding="utf-8"))
    page = select_one_page(full, manifest, Path(args.images_dir))
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps([page], ensure_ascii=False), encoding="utf-8")
    print(f"[smoke-input] wrote 1 page ({page['page_info']['image_path']}) -> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
