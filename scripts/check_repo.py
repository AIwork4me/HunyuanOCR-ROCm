#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 AIwork4me
"""Repo integrity checks for CI (and local pre-commit). No GPU, no torch.

Verifies, with one exit code:
  1. REPRO.yaml (or reproducibility.lock.yaml) has the required top-level sections.
  2. eval/canary_148.manifest.json is self-consistent (recomputed manifest_sha256).
  3. every relative link in README.md + docs/**/*.md resolves to an existing file.
  4. every src/**/*.py and scripts/**/*.py carries an SPDX-License-Identifier line.
  5. README + Makefile + user docs do not drift from the lock / canonical naming:
       - no stale "not_recorded" reproducibility claims in README;
       - no forbidden legacy canary tokens (canary-150 / Canary 150 / oracle-150);
       - the four formal results in README match REPRO.yaml;
       - no positive "precision-aligned AMD ROCm port" claim;
       - the Overall metric formula is consistent across README / methodology / lock.
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
FORMULA_RE = re.compile(r"Overall\s*=\s*`([^`]+)`")
SCRIPT_REF_RE = re.compile(r"scripts/([A-Za-z0-9_]+\.(?:py|sh))")
BASH_BLOCK_RE = re.compile(r"```bash\n(.*?)```", re.DOTALL)
# A line-continuation backslash that is NOT the final character: either trailing
# whitespace after it (``\ `` breaks the continuation — bash sees an escaped
# space) or an inline comment after it (``\ # ...``). Both are silent footguns.
BAD_BACKSLASH_RE = re.compile(r"\\( +.*$|[ \t]*#.*)", re.MULTILINE)
REQUIRED_LOCK_SECTIONS = ("hunyuanocr_rocm", "llama_cpp", "model", "omnidocbench", "environment", "benchmark")

# Tokens that must NOT appear in user-facing docs / Makefile (legacy canary-150
# naming). The lock file is exempt (it records the legacy name as provenance).
FORBIDDEN_TOKENS = ("canary-150", "Canary 150", "oracle = transformers canary (150)")
# A positive precision-alignment claim (negations like "Not precision-aligned" are fine).
FORBIDDEN_POSITIVE_CLAIM = "precision-aligned AMD ROCm port"

# The four formal results, with their README label and their path in the lock YAML.
FORMAL_RESULTS = [
    ("94.81", ("benchmark", "canary_148", "vllm_overall")),
    ("94.11", ("benchmark", "canary_148", "transformers_overall")),
    ("93.33", ("benchmark", "canary_148", "llamacpp_overall")),
    ("92.09", ("benchmark", "full_1651", "llamacpp_overall")),
]


def _user_docs() -> list[Path]:
    """README + user-facing docs/*.md (internal docs/superpowers/ excluded)."""
    docs = [REPO / "README.md"]
    for md in sorted((REPO / "docs").rglob("*.md")):
        if "docs/superpowers/" in md.as_posix():
            continue
        docs.append(md)
    return [d for d in docs if d.exists()]


def _load_lock():
    for name in ("REPRO.yaml", "reproducibility.lock.yaml"):
        lock_path = REPO / name
        if lock_path.exists():
            try:
                return yaml.safe_load(lock_path.read_text(encoding="utf-8")), None
            except Exception as exc:
                return None, f"{name} not parseable: {exc}"
    return None, "neither REPRO.yaml nor reproducibility.lock.yaml found"


def _normalize_formula(expr: str) -> str:
    """Canonicalize a metric formula so equivalent phrasings compare equal:
    lowercase; unify operators (−/–/·/× -> -/*); canonicalize variable names
    (text[_...] -> T, formula_cdm/cdm -> C, table_teds/teds -> E); drop whitespace.
    """
    s = expr.lower()
    for a, b in (("−", "-"), ("–", "-"), ("—", "-"), ("·", "*"), ("×", "*")):
        s = s.replace(a, b)
    for tok in ("text_editdist", "text_edit_dist", "text"):
        s = s.replace(tok, "T")
    for tok in ("formula_cdm", "cdm"):
        s = s.replace(tok, "C")
    for tok in ("table_teds", "teds"):
        s = s.replace(tok, "E")
    return re.sub(r"\s+", "", s)


# --- individual checks (each returns a list of error strings) ----------------


def check_lock_sections(lock) -> list[str]:
    if lock is None:
        return []  # parse error already reported by the loader
    return [f"REPRO.yaml missing section: {k}" for k in REQUIRED_LOCK_SECTIONS if k not in lock]


def check_canary_manifest() -> list[str]:
    mp = REPO / "eval" / "canary_148.manifest.json"
    try:
        m = json.loads(mp.read_text(encoding="utf-8"))
    except Exception as exc:
        return [f"canary manifest not parseable: {exc}"]
    body = {k: v for k, v in m.items() if k != "manifest_sha256"}
    recomputed = hashlib.sha256(
        json.dumps(body, ensure_ascii=False, sort_keys=True, indent=2).encode("utf-8")
    ).hexdigest()
    if recomputed != m.get("manifest_sha256"):
        return [
            (
                "eval/canary_148.manifest.json: manifest_sha256 does not match "
                "the recomputed hash (manifest is not self-consistent)"
            )
        ]
    return []


def check_doc_links() -> list[str]:
    errs: list[str] = []
    for md in _user_docs():
        text = md.read_text(encoding="utf-8")
        for link in LINK_RE.findall(text):
            if link.startswith(("http://", "https://", "mailto:", "#")):
                continue
            path_part = link.split("#", 1)[0].split("?")[0]
            if not path_part:
                continue
            target = (md.parent / path_part).resolve()
            if not target.exists():
                errs.append(f"{md.relative_to(REPO)}: broken link -> {link}")
    return errs


def check_spdx() -> list[str]:
    errs: list[str] = []
    for sub in ("src", "scripts"):
        for py in (REPO / sub).rglob("*.py"):
            head = "\n".join(py.read_text(encoding="utf-8").splitlines()[:3])
            if "SPDX-License-Identifier" not in head:
                errs.append(f"{py.relative_to(REPO)}: missing SPDX-License-Identifier header")
    return errs


def check_no_stale_not_recorded(readme: str) -> list[str]:
    """README must not claim HF revision / GGUF LFS oid are 'not_recorded' — the
    lock now records them (current_remote_artifact)."""
    if "not_recorded" in readme.lower():
        return [
            (
                "README.md still references 'not_recorded' for reproducibility fields, but "
                "REPRO.yaml now records all model/GGUF revisions + LFS oids "
                "(current_remote_artifact). Remove the stale claim."
            )
        ]
    return []


def check_no_forbidden_canary_tokens() -> list[str]:
    """User docs + Makefile must not carry legacy canary-150 naming."""
    errs: list[str] = []
    targets = [REPO / "README.md", REPO / "Makefile", *_user_docs()]
    seen: set[Path] = set()
    for md in targets:
        if md in seen or not md.exists():
            continue
        seen.add(md)
        text = md.read_text(encoding="utf-8")
        for tok in FORBIDDEN_TOKENS:
            if tok in text:
                errs.append(f"{md.relative_to(REPO)}: forbidden legacy token {tok!r} (use canary_148 / Canary 148)")
    return errs


def check_canonical_canary_name(readme: str) -> list[str]:
    if "canary_148" not in readme:
        return ["README.md: canonical canary name 'canary_148' not referenced (legacy naming may still be in use)"]
    return []


def check_formal_results_match_lock(readme: str, lock) -> list[str]:
    if lock is None:
        return []
    errs: list[str] = []
    for expected, path in FORMAL_RESULTS:
        node = lock
        for key in path:
            node = node.get(key, {}) if isinstance(node, dict) else None
            if node is None:
                break
        locked = None if node in (None, {}) else str(node)
        if locked is None:
            errs.append(f"lock missing formal result at {'.'.join(path)}")
        elif locked != expected:
            errs.append(f"lock {'.'.join(path)} = {locked} but README/expected = {expected}")
        elif expected not in readme:
            errs.append(f"README.md: formal result {expected} ({'.'.join(path)}) not present")
    return errs


def check_no_positive_precision_claim(readme: str) -> list[str]:
    if FORBIDDEN_POSITIVE_CLAIM in readme:
        return [
            (
                f"README.md contains the positive claim {FORBIDDEN_POSITIVE_CLAIM!r}; this project is "
                "'evaluation-backed', never 'precision-aligned' (no same-page-set CUDA control)."
            )
        ]
    return []


def check_readme_scripts_exist(readme: str) -> list[str]:
    """Every scripts/<name>.{py,sh} referenced in README must exist in the repo."""
    errs: list[str] = []
    seen: set[str] = set()
    for name in SCRIPT_REF_RE.findall(readme):
        if name in seen:
            continue
        seen.add(name)
        if not (REPO / "scripts" / name).exists():
            errs.append(f"README.md references scripts/{name}, which does not exist")
    return errs


def check_readme_bash_continuation(readme: str) -> list[str]:
    """In README bash blocks, a line-continuation backslash must be the final
    character of its line (no trailing whitespace, no inline comment after it)."""
    errs: list[str] = []
    for block in BASH_BLOCK_RE.findall(readme):
        for line in block.splitlines():
            if BAD_BACKSLASH_RE.search(line):
                errs.append(
                    f"README.md bash block has an invalid line continuation (backslash not last char "
                    f"or followed by a comment): {line.rstrip()!r}"
                )
    return errs


REPO_CLONE_RE = re.compile(r"git clone https://github\.com/AIwork4me/HunyuanOCR-ROCm\.git")
REQUIRED_QUICKSTART_ENVVARS = ("HUNYUAN_ROCM_DIR", "LLAMA_DIR", "GGUF_DIR", "DATA_DIR")


def check_readme_quickstart_structure(readme: str) -> list[str]:
    """The Quick Start must clone BEFORE exporting HUNYUAN_ROCM_DIR (the old flow
    exported the var then re-cloned, a contradiction), define all four env vars,
    use the canonical ``run_inference.py`` command, and not publish the legacy
    ``run_phase2_vllm.py`` command."""
    errs: list[str] = []
    clone_m = REPO_CLONE_RE.search(readme)
    export_idx = readme.find("export HUNYUAN_ROCM_DIR=")
    if clone_m is None:
        errs.append("README.md: Quick Start missing the `git clone .../HunyuanOCR-ROCm.git` step")
    elif export_idx == -1:
        errs.append("README.md: Quick Start missing `export HUNYUAN_ROCM_DIR=...`")
    elif clone_m.start() > export_idx:
        errs.append(
            "README.md: Quick Start exports HUNYUAN_ROCM_DIR before the `git clone` — clone must come first (Step 0)"
        )
    for var in REQUIRED_QUICKSTART_ENVVARS:
        if f"export {var}=" not in readme:
            errs.append(f"README.md: Quick Start does not define required env var {var}")
    if "python scripts/run_inference.py" not in readme:
        errs.append("README.md: Quick Start does not use the canonical `python scripts/run_inference.py` command")
    if "python scripts/run_phase2_vllm.py" in readme:
        errs.append(
            "README.md: publishes the legacy `python scripts/run_phase2_vllm.py` command — use run_inference.py"
        )
    return errs


def check_metric_formula_consistency(lock) -> list[str]:
    """The Overall formula must be consistent across README, the methodology doc,
    and the lock. Only docs that actually state the Overall formula are checked
    (issue drafts / design briefs legitimately do not)."""
    errs: list[str] = []
    lock_formula = None
    if isinstance(lock, dict):
        lock_formula = (lock.get("omnidocbench", {}) or {}).get("metric", {}).get("overall_formula")
    formula_docs = [REPO / "README.md", REPO / "docs" / "benchmark-methodology.md"]
    norms: dict[str, str | None] = {}
    if lock_formula:
        norms["REPRO.yaml"] = _normalize_formula(lock_formula)
    for md in formula_docs:
        if not md.exists():
            continue
        m = FORMULA_RE.search(md.read_text(encoding="utf-8"))
        if m is None:
            errs.append(f"{md.relative_to(REPO)}: could not find an 'Overall = `...`' formula to cross-check")
        else:
            norms[md.relative_to(REPO).as_posix()] = _normalize_formula(m.group(1))
    distinct = {v for v in norms.values() if v}
    if len(distinct) > 1:
        errs.append(
            "Overall metric formula is inconsistent across README / methodology / lock:\n  "
            + "\n  ".join(f"{k}: {v}" for k, v in norms.items())
        )
    return errs


def all_errors() -> list[str]:
    errs: list[str] = []
    lock, load_err = _load_lock()
    if load_err:
        errs.append(load_err)
    readme = (REPO / "README.md").read_text(encoding="utf-8") if (REPO / "README.md").exists() else ""
    errs += check_lock_sections(lock)
    errs += check_canary_manifest()
    errs += check_doc_links()
    errs += check_spdx()
    errs += check_no_stale_not_recorded(readme)
    errs += check_no_forbidden_canary_tokens()
    errs += check_canonical_canary_name(readme)
    errs += check_formal_results_match_lock(readme, lock)
    errs += check_no_positive_precision_claim(readme)
    errs += check_readme_scripts_exist(readme)
    errs += check_readme_bash_continuation(readme)
    errs += check_readme_quickstart_structure(readme)
    errs += check_metric_formula_consistency(lock)
    return errs


def main() -> None:
    errs = all_errors()
    if errs:
        for e in errs:
            print("FAIL:", e)
        sys.exit(1)
    n = sum(1 for _ in (REPO / "src").rglob("*.py"))
    print(f"OK: repo integrity checks passed ({n} src files, links, lock, manifest, doc/lock consistency).")


if __name__ == "__main__":
    main()
