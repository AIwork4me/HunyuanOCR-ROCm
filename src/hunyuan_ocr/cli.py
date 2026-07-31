# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 AIwork4me
"""Unified ``hunyuan-ocr`` CLI.

Self-contained subcommands (work from a wheel install, no repo checkout):
  doctor            — environment/dataset/scorer readiness (no model load)
  validate          — pre-score validation of a prediction dir
  manifest verify   — conservation-law check of a run_manifest.json
  canary materialize— rebuild the 148-page canary from the full GT
  predict           — multi-server predict via hunyuan_ocr.driver (llamacpp/vllm/openai)
  score             — OmniDocBench scoring via hunyuan_ocr.scoring (needs the scorer venv)
  benchmark         — print the verified results from reproducibility.lock.yaml (read-only)
  report            — assemble a benchmark release-artifact bundle from a run_manifest.json

``predict --backend transformers`` is the one exception: it still delegates to the
repo-only ``scripts/run_phase1_transformers.py`` driver and needs a ROCm torch.
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


def _check(name, state, detail, *, critical=False, advice=""):
    """One doctor check. ``state`` in {"ok","miss","info"}; ``critical`` marks
    checks whose failure makes ``--strict`` exit non-zero for the target backend."""
    return {"name": name, "state": state, "detail": detail, "critical": critical, "advice": advice}


def _try_version(modname):
    try:
        m = importlib.import_module(modname)
        return getattr(m, "__version__", None)
    except Exception:
        return None


def _has_rocm_toolchain():
    return Path("/opt/rocm").exists() or bool(shutil.which("rocm-smi"))


def _torch_hip():
    """Return the torch version + hip string if a ROCm torch is importable, else None."""
    try:
        import torch  # type: ignore
    except Exception:
        return None
    hip = getattr(getattr(torch, "version", None), "hip", None)
    if not hip:  # a CUDA/stock torch is NOT a ROCm torch
        return None
    return {"torch": getattr(torch, "__version__", "?"), "hip": hip}


def _collect_doctor_checks(backend: str | None) -> list[dict]:
    """Build the check list. Checks not relevant to the (optional) target backend
    are still reported for visibility, but only critical checks for the target
    backend count toward ``--strict`` failure."""
    checks: list[dict] = []
    backend = (backend or "").lower()

    checks.append(_check("python", "ok", platform.python_version()))
    checks.append(_check("platform", "ok", f"{platform.system()} {platform.machine()}"))

    # ROCm toolchain — critical for every GPU backend.
    rocm = _has_rocm_toolchain()
    checks.append(
        _check(
            "ROCm toolchain",
            "ok" if rocm else "miss",
            "/opt/rocm present" if rocm else "not detected",
            critical=backend in {"llamacpp", "transformers", "vllm"},
            advice="install the ROCm stack (gfx1100/RDNA3); required for every GPU backend",
        )
    )

    # openai client (optional; needed by predict for OAI backends).
    ov = _try_version("openai")
    checks.append(
        _check("openai client", "ok" if ov else "miss", ov or "not installed", advice='pip install ".[client]"')
    )

    # llama.cpp backend: llama-server binary + GGUF weights.
    ls = shutil.which("llama-server")
    checks.append(
        _check(
            "llama-server",
            "ok" if ls else "miss",
            ls or "not on PATH",
            critical=backend == "llamacpp",
            advice="build llama.cpp with HIP (see README Quick Start)",
        )
    )
    gguf_dir = os.environ.get("HUNYUANOCR_GGUF_DIR") or os.environ.get("GGUF_DIR")
    gguf_ok = bool(gguf_dir) and Path(gguf_dir).is_dir()
    gguf_detail = gguf_dir or "HUNYUANOCR_GGUF_DIR env not set"
    if gguf_ok:
        main_gguf = any(Path(gguf_dir).glob("HunyuanOCR-bf16.gguf"))
        mmproj = any(Path(gguf_dir).glob("mmproj-HunyuanOCR-bf16.gguf"))
        gguf_ok = main_gguf and mmproj
        gguf_detail = f"{gguf_dir} (main={main_gguf}, mmproj={mmproj})"
    checks.append(
        _check(
            "GGUF weights",
            "ok" if gguf_ok else ("miss" if gguf_dir else "info"),
            gguf_detail,
            critical=backend == "llamacpp",
            advice="set HUNYUANOCR_GGUF_DIR to a dir with HunyuanOCR-bf16.gguf + mmproj-HunyuanOCR-bf16.gguf",
        )
    )

    # ROCm torch — critical for transformers + vllm.
    thip = _torch_hip()
    checks.append(
        _check(
            "ROCm torch (hip)",
            "ok" if thip else "miss",
            f"{thip['torch']} (hip {thip['hip']})" if thip else "no ROCm torch (CUDA/stock/absent)",
            critical=backend in {"transformers", "vllm"},
            advice="install a ROCm torch wheel (see reproducibility.lock.yaml); stock PyPI torch is not ROCm",
        )
    )

    # transformers importable — critical for transformers.
    tv = _try_version("transformers")
    checks.append(
        _check(
            "transformers",
            "ok" if tv else "miss",
            tv or "not installed",
            critical=backend == "transformers",
            advice='pip install ".[transformers]" (and a ROCm torch)',
        )
    )

    # vLLM importable — critical for vllm.
    vv = _try_version("vllm")
    checks.append(
        _check(
            "vLLM",
            "ok" if vv else "miss",
            vv or "not installed",
            critical=backend == "vllm",
            advice="install a ROCm vLLM build (see reproducibility.lock.yaml)",
        )
    )

    # model dir (safetensors) — critical for transformers + vllm.
    model = os.environ.get("HUNYUANOCR_MODEL")
    model_ok = bool(model) and Path(model).is_dir()
    checks.append(
        _check(
            "model dir (safetensors)",
            "ok" if model_ok else ("miss" if model else "info"),
            model or "HUNYUANOCR_MODEL env not set",
            critical=backend in {"transformers", "vllm"},
            advice="set HUNYUANOCR_MODEL to the tencent/HunyuanOCR weights dir",
        )
    )

    # GT json (informational; needed to actually run a benchmark, not to install).
    gt = os.environ.get("HUNYUANOCR_GT")
    if gt and Path(gt).is_file():
        try:
            pages = json.loads(Path(gt).read_text(encoding="utf-8"))
            checks.append(_check("GT json", "ok", f"{len(pages)} pages ({gt})"))
        except Exception as exc:
            checks.append(_check("GT json", "miss", f"unparseable: {exc}", advice="check the file"))
    else:
        checks.append(_check("GT json", "info", gt or "HUNYUANOCR_GT env not set"))

    # OmniDocBench scorer venv (needed to score; informational unless scoring).
    try:
        from hunyuan_ocr import scoring

        vp = Path(scoring.DEFAULT_VENV_PYTHON)
        checks.append(
            _check(
                "scorer venv",
                "ok" if vp.exists() else "miss",
                str(vp),
                advice="install the OmniDocBench scorer (see reproducibility.lock.yaml) or set OMNIDOCBENCH_VENV",
            )
        )
    except Exception:
        pass

    return checks


def _doctor(args) -> int:
    backend = getattr(args, "backend", None)
    checks = _collect_doctor_checks(backend)
    critical_failures = [c for c in checks if c["critical"] and c["state"] != "ok"]
    ok = not critical_failures

    if getattr(args, "json", False):
        env = {
            "python": platform.python_version(),
            "platform": f"{platform.system()} {platform.machine()}",
            "has_rocm_toolchain": _has_rocm_toolchain(),
            "openai": _try_version("openai"),
            "torch_hip": _torch_hip(),
            "transformers": _try_version("transformers"),
            "vllm": _try_version("vllm"),
            "llama_server": shutil.which("llama-server"),
            "gguf_dir": os.environ.get("HUNYUANOCR_GGUF_DIR") or os.environ.get("GGUF_DIR"),
            "model_dir": os.environ.get("HUNYUANOCR_MODEL"),
            "gt_json": os.environ.get("HUNYUANOCR_GT"),
        }
        # Central doctor contract (cli-contract.md @ ccd466e) requires a `status`
        # field in {"ready","not-ready"}; `ok` is retained for back-compat.
        payload = {
            "status": "ready" if ok else "not-ready",
            "ok": ok,
            "backend": backend,
            "checks": checks,
            "environment": env,
        }
        print(json.dumps(payload, indent=2))
        return 0 if ok else 1

    print("hunyuan-ocr doctor" + (f" (--strict --backend {backend})" if backend else ""))
    for c in checks:
        if c["state"] == "ok":
            _ok(c["name"], c["detail"])
        elif c["state"] == "miss":
            _warn(c["name"], c["detail"], c["advice"])
        else:
            _info(c["name"], c["detail"])
    if backend:
        if ok:
            print(f"strict: all critical checks for backend '{backend}' passed.")
        else:
            print(
                f"strict: {len(critical_failures)} critical check(s) for backend '{backend}' failed: "
                + ", ".join(c["name"] for c in critical_failures),
                file=sys.stderr,
            )
    else:
        print("doctor is advisory (no --backend); it never loads the model. Fix any [MISS] above.")
    return 0 if ok else 1


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
    try:
        raw = mp.read_text(encoding="utf-8")
    except OSError as exc:
        print(f"[error] cannot read {mp}: {exc}", file=sys.stderr)
        return 2
    if not raw.strip():
        print(f"[error] {mp} is empty", file=sys.stderr)
        return 1
    try:
        m = json.loads(raw)
    except json.JSONDecodeError as exc:
        print(f"[error] {mp} is not valid JSON: {exc.msg} (line {exc.lineno} col {exc.colno})", file=sys.stderr)
        return 1
    if not isinstance(m, dict):
        print(f"[error] {mp} is valid JSON but not an object (got {type(m).__name__})", file=sys.stderr)
        return 1
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


# --- predict / score (package-resident; work from a wheel install) -----------


def _predict(args) -> int:
    """Self-contained for the OpenAI-compatible backends (llamacpp/vllm/openai):
    runs the package driver directly, no ``scripts/`` checkout required. The
    ``transformers`` backend still needs the repo checkout + a ROCm torch install
    (it is a separate, GPU-only driver)."""
    if args.backend == "transformers":
        drv = _import_script("run_phase1_transformers")
        if drv is None:
            print(
                "[error] predict --backend transformers needs a repo checkout AND a ROCm torch install;\n"
                "        it is a separate GPU-only driver. Run instead:\n"
                "        python scripts/run_phase1_transformers.py ...",
                file=sys.stderr,
            )
            return 2
        drv.main_with_args(args.extra)
        return 0
    try:
        from openai import OpenAI
    except ImportError:
        print('[error] predict needs the OpenAI client: pip install ".[client]"', file=sys.stderr)
        return 2
    from hunyuan_ocr import driver
    from hunyuan_ocr.backends.vllm_client import infer_one

    drv_args = driver.parse_args(["--backend-name", args.backend, *_clean_extra(args.extra)])
    return int(driver.dispatch(drv_args, infer_one=infer_one, client_factory=OpenAI) or 0)


def _score(args) -> int:
    """Score a prediction dir via the package scorer (shared with the
    ``scripts/score_predictions.py`` wrapper). Needs the OmniDocBench scorer venv
    (set via --venv-python or OMNIDOCBENCH_VENV)."""
    from hunyuan_ocr import scoring

    try:
        result = scoring.score_directory(
            gt_json=args.gt_json,
            pred_dir=args.pred_dir,
            omnidocbench_repo=args.omnidocbench_repo,
            venv_python=args.venv_python,
            skip_validation=args.skip_validation,
        )
    except scoring.ScoringError as exc:
        print(f"[error] {exc}", file=sys.stderr)
        return 1
    print(scoring.format_score_table(args.label, result["metrics"]))
    return 0


def _clean_extra(extra):
    """Drop a leading '--' that argparse.REMAINDER may capture."""
    extra = list(extra or [])
    if extra and extra[0] == "--":
        extra = extra[1:]
    return extra


def _benchmark(args) -> int:
    """Print the verified benchmark results from reproducibility.lock.yaml (read-only)."""
    import yaml

    from hunyuan_ocr.results import render_results_block

    lock_path = Path(args.lock) if getattr(args, "lock", None) else Path.cwd() / "REPRO.yaml"
    if not lock_path.exists():
        legacy = Path.cwd() / "reproducibility.lock.yaml"
        if legacy.exists():
            lock_path = legacy
    if not lock_path.is_file():
        print(f"[error] lock not found: {lock_path}", file=sys.stderr)
        return 2
    lock = yaml.safe_load(lock_path.read_text(encoding="utf-8"))
    print(render_results_block(lock))
    return 0


def _report(args) -> int:
    """Assemble a benchmark release-artifact bundle (see docs/release-artifact.md)."""
    from hunyuan_ocr.report import assemble_release_artifact

    repo_root = Path(args.repo_root) if getattr(args, "repo_root", None) else Path(__file__).resolve().parents[2]
    try:
        out = assemble_release_artifact(args.pred_dir, args.out, repo_root)
    except FileNotFoundError as exc:
        print(f"[error] {exc}", file=sys.stderr)
        return 2
    print(f"[OK] wrote release artifact -> {out} (run_manifest, environment, commands, lock, checksums)")
    return 0


# --- standard CLI contract (version / capabilities / parse) ------------------
# Thin handlers over hunyuan_ocr.standard_cli (ADR-0011, locked to central
# commit ccd466ef317fd6a710131db3a19ec9d55a65ce2e). Exit codes 0/1/2/3/4/5.


def _version_cmd(args) -> int:
    from hunyuan_ocr.standard_cli import cmd_version

    return cmd_version()


def _capabilities_cmd(args) -> int:
    from hunyuan_ocr.standard_cli import cmd_capabilities

    return cmd_capabilities()


def _parse_cmd(args) -> int:
    from hunyuan_ocr.standard_cli import cmd_parse

    return cmd_parse(
        img_dir=Path(args.img_dir),
        out_dir=Path(args.out_dir),
        platform=getattr(args, "platform", "linux-rocm"),
        backend=getattr(args, "backend", None),
        server_url=getattr(args, "server_url", None),
        model=getattr(args, "model", None),
        max_pixels=getattr(args, "max_pixels", None),
        limit=getattr(args, "limit", None),
    )


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="hunyuan-ocr", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    sub = p.add_subparsers(dest="cmd", required=True)

    # --- standard CLI contract (ADR-0011) ---
    ver = sub.add_parser("version", help="print package + contract version as JSON")
    ver.add_argument("--json", action="store_true", help="emit JSON (always JSON; accepted for contract uniformity)")

    cap = sub.add_parser("capabilities", help="declared platforms/backends as JSON")
    cap.add_argument("--json", action="store_true", help="emit JSON (always JSON; accepted for contract uniformity)")

    par = sub.add_parser("parse", help="parse an image dir -> canonical result.json (standard CLI)")
    par.add_argument("--img-dir", required=True, help="directory of input page images")
    par.add_argument("--out-dir", required=True, help="output directory for <stem>.md + result")
    par.add_argument("--platform", default="linux-rocm", choices=["linux-rocm", "windows-hip"])
    par.add_argument("--backend", help="backend to run (vllm|llama-cpp|openai|transformers)")
    par.add_argument("--benchmark", default="omnidocbench-v16", help="benchmark tag (informational)")
    par.add_argument("--server-url", help="OpenAI-compatible server URL (or HUNYUANOCR_SERVER_URL)")
    par.add_argument("--model", default=None, help="api_model_name (default tencent/HunyuanOCR)")
    par.add_argument("--max-pixels", type=int, default=None, help="optional client-side image pixel cap")
    par.add_argument("--limit", type=int, default=None, help="process only the first N images (debug; not a full set)")
    par.add_argument(
        "--json", action="store_true", help="emit the cli_result JSON (always JSON; accepted for contract uniformity)"
    )

    doc = sub.add_parser("doctor", help="environment + dataset + scorer readiness")
    doc.add_argument("--strict", action="store_true", help="exit non-zero if critical checks fail")
    doc.add_argument(
        "--backend",
        choices=["llamacpp", "transformers", "vllm"],
        help="target backend whose critical checks gate --strict",
    )
    doc.add_argument("--json", action="store_true", help="emit stable JSON (no secrets)")

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

    pr = sub.add_parser(
        "predict",
        help="multi-server predict (llamacpp/vllm/openai are self-contained; transformers needs checkout)",
    )
    pr.add_argument("--backend", default="llamacpp", choices=["llamacpp", "vllm", "openai", "transformers"])
    pr.add_argument(
        "extra", nargs=argparse.REMAINDER, help="driver flags (--gt-json, --images-dir, --pred-dir, --ports, ...)"
    )

    from hunyuan_ocr import scoring

    sc = sub.add_parser("score", help="OmniDocBench scoring (scorer venv required)")
    sc.add_argument("--pred-dir", required=True)
    sc.add_argument("--gt-json", required=True)
    sc.add_argument("--label", default="backend")
    sc.add_argument("--omnidocbench-repo", default=scoring.DEFAULT_OMNIDOCBENCH_REPO)
    sc.add_argument("--venv-python", default=scoring.DEFAULT_VENV_PYTHON)
    sc.add_argument("--skip-validation", action="store_true", help="DANGEROUS: bypass pre-score validation")

    bm = sub.add_parser("benchmark", help="print the verified results from the lock (read-only)")
    bm.add_argument("--lock", help="path to REPRO.yaml (default: ./REPRO.yaml or ./reproducibility.lock.yaml)")

    rep = sub.add_parser("report", help="assemble a benchmark release-artifact bundle")
    rep.add_argument("--pred-dir", required=True, help="prediction dir containing run_manifest.json")
    rep.add_argument("--out", required=True, help="output artifact directory")
    rep.add_argument("--repo-root", help="repo root (to copy reproducibility.lock.yaml); default: this package's repo")
    return p


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    dispatch = {
        "version": _version_cmd,
        "capabilities": _capabilities_cmd,
        "parse": _parse_cmd,
        "doctor": _doctor,
        "validate": _validate,
        "manifest": lambda a: _manifest_verify(a) if a.mcmd == "verify" else 2,
        "canary": lambda a: _canary_materialize(a) if a.ccmd == "materialize" else 2,
        "predict": _predict,
        "score": _score,
        "benchmark": _benchmark,
        "report": _report,
    }
    handler = dispatch[args.cmd]
    rc = handler(args)
    return int(rc or 0)


if __name__ == "__main__":
    sys.exit(main())
