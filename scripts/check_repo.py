#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 AIwork4me
"""Repo integrity checks for CI (and local pre-commit). No GPU, no torch.

Verifies, with one exit code:
  1. reproducibility.lock.yaml has the required top-level sections.
  2. eval/canary_148.manifest.json is self-consistent (recomputed manifest_sha256).
  3. every relative link in README.md + docs/**/*.md resolves to an existing file.
  4. every src/**/*.py and scripts/**/*.py carries an SPDX-License-Identifier line.
"""

from __future__ import annotations

import hashlib
import json
import re
import sys
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parents[1]
LINK_RE = re.compile(r"\[[^\]]+\]\(([^)]+)\)")
REQUIRED_LOCK_SECTIONS = ("hunyuanocr_rocm", "llama_cpp", "model", "omnidocbench", "environment", "benchmark")
errors: list[str] = []


def check_lock() -> None:
    lock_path = REPO / "reproducibility.lock.yaml"
    try:
        lock = yaml.safe_load(lock_path.read_text(encoding="utf-8"))
    except Exception as exc:  # noqa: BLE001
        errors.append(f"reproducibility.lock.yaml not parseable: {exc}")
        return
    for key in REQUIRED_LOCK_SECTIONS:
        if key not in lock:
            errors.append(f"reproducibility.lock.yaml missing section: {key}")


def check_canary_manifest() -> None:
    mp = REPO / "eval" / "canary_148.manifest.json"
    try:
        m = json.loads(mp.read_text(encoding="utf-8"))
    except Exception as exc:  # noqa: BLE001
        errors.append(f"canary manifest not parseable: {exc}")
        return
    body = {k: v for k, v in m.items() if k != "manifest_sha256"}
    recomputed = hashlib.sha256(
        json.dumps(body, ensure_ascii=False, sort_keys=True, indent=2).encode("utf-8")
    ).hexdigest()
    if recomputed != m.get("manifest_sha256"):
        errors.append(
            "eval/canary_148.manifest.json: manifest_sha256 does not match "
            "the recomputed hash (manifest is not self-consistent)"
        )


def check_doc_links() -> None:
    # User-facing docs only; docs/superpowers/ are internal planning artifacts
    # (intentionally repo-root-relative links) and are excluded.
    mds = [REPO / "README.md"]
    for md in sorted((REPO / "docs").rglob("*.md")):
        if "docs/superpowers/" in md.as_posix():
            continue
        mds.append(md)
    for md in mds:
        if not md.exists():
            continue
        text = md.read_text(encoding="utf-8")
        for link in LINK_RE.findall(text):
            if link.startswith(("http://", "https://", "mailto:", "#")):
                continue
            path_part = link.split("#", 1)[0].split("?")[0]
            if not path_part:
                continue
            target = (md.parent / path_part).resolve()
            if not target.exists():
                errors.append(f"{md.relative_to(REPO)}: broken link -> {link}")


def check_spdx() -> None:
    for sub in ("src", "scripts"):
        for py in (REPO / sub).rglob("*.py"):
            head = "\n".join(py.read_text(encoding="utf-8").splitlines()[:3])
            if "SPDX-License-Identifier" not in head:
                errors.append(f"{py.relative_to(REPO)}: missing SPDX-License-Identifier header")


def main() -> None:
    check_lock()
    check_canary_manifest()
    check_doc_links()
    check_spdx()
    if errors:
        for e in errors:
            print("FAIL:", e)
        sys.exit(1)
    n = sum(1 for _ in (REPO / "src").rglob("*.py"))
    print(f"OK: repo integrity checks passed ({n} src files, links, lock, manifest).")


if __name__ == "__main__":
    main()
