# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 AIwork4me
"""Unified ``hunyuan-ocr`` CLI.

Self-contained, package-only subcommands (work from a wheel install):
  doctor            — environment/dataset/scorer readiness (no model load)
  validate          — pre-score validation of a prediction dir
  manifest verify   — conservation-law check of a run_manifest.json
  canary materialize— rebuild the 148-page canary from the full GT

Repo-required subcommands (delegate to scripts/, need a checkout):
  predict           — multi-server predict via the OpenAI-compatible / transformers driver
  score             — OmniDocBench scoring (also needs the OmniDocBench scorer venv)
"""

from __future__ import annotations

import argparse
import importlib
import json
import os
import platform
import shutil
import sys
from pathlib import Path


def _ok(label, detail=""):
    print(f"  [ OK ] {label}{': ' + detail if detail else ''}")


def _warn(label, detail, advice):
    print(f"  [MISS] {label}: {detail}")
    print(f"        -> {advice}")


def _info(label, detail=""):
    print(f"  [INFO] {label}{': ' + detail if detail else ''}")


def _import_script(name):
    """Import a repo ``scripts/<name>.py`` if running from a checkout (editable/
    src layout). Returns the module or None (e.g. under a wheel install)."""
    pkg_dir = Path(__file__).resolve().parent  # .../src/hunyuan_ocr
    scripts = pkg_dir.parents[1] / "scripts"  # .../repo/scripts (editable only)
    if (scripts / f"{name}.py").exists():
        sys.path.insert(0, str(scripts))
        return importlib.import_module(name)
    return None


# --- doctor ------------------------------------------------------------------


def _doctor(args) -> int:
    print("hunyuan-ocr doctor")
    _info("python", platform.python_version())
    _info("platform", f"{platform.system()} {platform.machine()}")

    for mod in ("openai",):
        try:
            m = importlib.import_module(mod)
            _ok(f"{mod}", getattr(m, "__version__", "?"))
        except Exception:
            _warn(mod, "not installed", 'pip install ".[client]"')

    for mod in ("torch", "transformers", "vllm"):
        try:
            m = importlib.import_module(mod)
            _info(mod, getattr(m, "__version__", "?"))
        except Exception:
            _info(mod, "not installed (optional; needed only for that backend)")

    ls = shutil.which("llama-server")
    if ls:
        _ok("llama-server", ls)
    else:
        _warn("llama-server", "not on PATH", "build llama.cpp with HIP (see README Quick Start)")

    if Path("/opt/rocm").exists() or shutil.which("rocm-smi"):
        _ok("ROCm", "/opt/rocm present")
    else:
        _info("ROCm", "not detected (optional; required only on AMD GPUs)")

    model = os.environ.get("HUNYUANOCR_MODEL")
    if model:
        (
            _ok("model dir", model)
            if Path(model).is_dir()
            else _warn("model dir", f"{model} missing", "set HUNYUANOCR_MODEL to the weights dir")
        )
    else:
        _info("model dir", "HUNYUANOCR_MODEL env not set")

    gt = os.environ.get("HUNYUANOCR_GT")
    if gt:
        if not Path(gt).is_file():
            _warn("GT json", f"{gt} missing", "set HUNYUANOCR_GT to OmniDocBench.json")
        else:
            try:
                pages = json.loads(Path(gt).read_text(encoding="utf-8"))
                _ok("GT json", f"{len(pages)} pages ({gt})")
            except Exception as exc:
                _warn("GT json", f"unparseable: {exc}", "check the file")
    else:
        _info("GT json", "HUNYUANOCR_GT env not set")

    from hunyuan_ocr import scoring

    vp = Path(scoring.DEFAULT_VENV_PYTHON)
    (
        _ok("scorer venv", str(vp))
        if vp.exists()
        else _warn("scorer venv", f"{vp} missing", "install the OmniDocBench scorer (see reproducibility.lock.yaml)")
    )

    out = os.environ.get("HUNYUANOCR_OUT_DIR")
    if out:
        try:
            Path(out).mkdir(parents=True, exist_ok=True)
            (Path(out) / ".w").write_text("x")
            (Path(out) / ".w").unlink()
            _ok("out dir", f"writable ({out})")
        except OSError as exc:
            _warn("out dir", f"{out} not writable: {exc}", "choose a writable OUT_DIR")
    else:
        _info("out dir", "HUNYUANOCR_OUT_DIR env not set")
    print("doctor is advisory; it never loads the model. Fix any [MISS] above.")
    return 0


# --- validate / manifest / canary (package-only) -----------------------------


def _validate(args) -> int:
    from hunyuan_ocr.validation import validate_predictions

    rep = validate_predictions(args.gt_json, args.pred_dir, strict=not args.lenient)
    print(f"expected={rep.expected} valid={rep.valid} errors={len(rep.errors())} warnings={len(rep.warnings())}")
    for prob in rep.problems:
        tag = "ERROR" if prob.severity == "error" else "WARN "
        print(f"  [{tag}] {prob.code}: {prob.message}")
    ok = rep.ok if args.lenient else rep.ok_strict
    print("RESULT: PASS" if ok else "RESULT: FAIL")
    return 0 if ok else 1


def _manifest_verify(args) -> int:
    from hunyuan_ocr import runner

    mp = Path(args.pred_dir) / "run_manifest.json"
    if not mp.is_file():
        print(f"[error] no run_manifest.json in {args.pred_dir}", file=sys.stderr)
        return 2
    m = json.loads(mp.read_text(encoding="utf-8"))
    errs = runner.validate_manifest(m)
    if errs:
        print(f"[FAIL] manifest violates {len(errs)} invariant(s):")
        for e in errs:
            print(f"  - {e}")
        return 1
    print(f"[OK] manifest is valid: backend={m.get('backend')} run={m.get('run_counts')} final={m.get('final_state')}")
    return 0


def _canary_materialize(args) -> int:
    from hunyuan_ocr import canary

    try:
        sha = canary.materialize(args.full_gt, args.manifest, args.out)
    except canary.CanaryError as exc:
        print(f"[error] {exc}", file=sys.stderr)
        return 1
    print(f"[OK] wrote {args.out} (sha256 {sha[:12]}...) — byte-identical to locked canary")
    return 0


# --- predict / score (repo scripts required) ---------------------------------


def _predict(args) -> int:
    if args.backend == "transformers":
        drv = _import_script("run_phase1_transformers")
        if drv is None:
            print("[error] predict needs a repo checkout (scripts/ not found).", file=sys.stderr)
            print("        run instead: python scripts/run_phase1_transformers.py ...", file=sys.stderr)
            return 2
        drv.main_with_args(args.extra)
        return 0
    drv = _import_script("run_phase2_vllm")
    if drv is None:
        print("[error] predict needs a repo checkout (scripts/ not found).", file=sys.stderr)
        print(
            f"        run instead: python scripts/run_phase2_vllm.py --backend-name {args.backend} ...", file=sys.stderr
        )
        return 2
    drv.main_with_args(["--backend-name", args.backend, *args.extra])
    return 0


def _score(args) -> int:
    sp = _import_script("score_predictions")
    if sp is None:
        print("[error] score needs a repo checkout (scripts/ not found).", file=sys.stderr)
        print("        run instead: python scripts/score_predictions.py ...", file=sys.stderr)
        return 2
    sp.main_with_args(args.extra)
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="hunyuan-ocr", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    sub = p.add_subparsers(dest="cmd", required=True)

    sub.add_parser("doctor", help="environment + dataset + scorer readiness")

    v = sub.add_parser("validate", help="validate a prediction dir against GT")
    v.add_argument("--gt-json", required=True)
    v.add_argument("--pred-dir", required=True)
    v.add_argument("--lenient", action="store_true", help="warnings are non-fatal")

    man = sub.add_parser("manifest", help="run-manifest utilities")
    msub = man.add_subparsers(dest="mcmd", required=True)
    mv = msub.add_parser("verify", help="verify run_manifest.json conservation laws")
    mv.add_argument("--pred-dir", required=True)

    can = sub.add_parser("canary", help="canary-subset utilities")
    csub = can.add_subparsers(dest="ccmd", required=True)
    cmv = csub.add_parser("materialize", help="rebuild the 148-page canary from the full GT")
    cmv.add_argument("--full-gt", required=True)
    cmv.add_argument("--manifest", required=True)
    cmv.add_argument("--out", required=True)

    pr = sub.add_parser("predict", help="multi-server predict (repo scripts required)")
    pr.add_argument("--backend", default="llamacpp", choices=["llamacpp", "vllm", "openai", "transformers"])
    pr.add_argument("extra", nargs=argparse.REMAINDER, help="driver flags")

    sc = sub.add_parser("score", help="OmniDocBench scoring (repo scripts + scorer required)")
    sc.add_argument("extra", nargs=argparse.REMAINDER, help="score_predictions.py flags")
    return p


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    dispatch = {
        "doctor": _doctor,
        "validate": _validate,
        "manifest": lambda a: _manifest_verify(a) if a.mcmd == "verify" else 2,
        "canary": lambda a: _canary_materialize(a) if a.ccmd == "materialize" else 2,
        "predict": _predict,
        "score": _score,
    }
    handler = dispatch[args.cmd]
    rc = handler(args)
    return int(rc or 0)


if __name__ == "__main__":
    sys.exit(main())
