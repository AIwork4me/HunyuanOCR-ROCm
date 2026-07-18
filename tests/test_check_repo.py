# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 AIwork4me
"""Unit + integration tests for scripts/check_repo.py (the CI integrity gate)."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]

# check_repo lives in scripts/, not an importable package; load it by path.
_spec = importlib.util.spec_from_file_location("check_repo", REPO / "scripts" / "check_repo.py")
check_repo = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(check_repo)
sys.modules["check_repo"] = check_repo


def test_normalize_formula_equivalence():
    """The README/methodology unicode phrasing and the lock underscore phrasing
    must normalize to the same canonical form."""
    readme = "((1−text)·100 + CDM·100 + TEDS·100)/3"
    methodology = "((1−text)·100 + CDM·100 + TEDS·100) / 3"
    lock = "((1 - text_EditDist)*100 + formula_CDM*100 + table_TEDS*100) / 3"
    n = check_repo._normalize_formula
    assert n(readme) == n(methodology) == n(lock)
    # a genuinely different formula does not match
    assert n("text*100 + cdm*100") != n(lock)


def test_check_no_stale_not_recorded():
    assert check_repo.check_no_stale_not_recorded("... HF revision is not_recorded ...")
    assert check_repo.check_no_stale_not_recorded("all fields verified") == []


def test_check_canonical_canary_name():
    assert check_repo.check_canonical_canary_name("no canary mention here")
    assert check_repo.check_canonical_canary_name("see eval/canary_148.manifest.json") == []


def test_check_no_positive_precision_claim():
    bad = "This is a precision-aligned AMD ROCm port of HunyuanOCR."
    assert check_repo.check_no_positive_precision_claim(bad)
    # negation is fine
    assert check_repo.check_no_positive_precision_claim("Not a precision-aligned port.") == []


def test_check_formal_results_match_lock_happy_and_drift():
    lock = {
        "benchmark": {
            "canary_148": {"vllm_overall": 94.81, "transformers_overall": 94.11, "llamacpp_overall": 93.33},
            "full_1651": {"llamacpp_overall": 92.09},
        }
    }
    readme = "vLLM 94.81, transformers 94.11, llama.cpp 93.33, full 92.09"
    assert check_repo.check_formal_results_match_lock(readme, lock) == []
    # a drifted lock value is caught
    bad_lock = {"benchmark": {"canary_148": {"vllm_overall": 94.99}, "full_1651": {}}}
    errs = check_repo.check_formal_results_match_lock(readme, bad_lock)
    assert errs and any("94.99" in e for e in errs)


def test_check_lock_sections_missing():
    errs = check_repo.check_lock_sections({"hunyuanocr_rocm": {}})
    assert any("benchmark" in e for e in errs)


def test_check_readme_scripts_exist():
    # a real repo script -> no error
    assert check_repo.check_readme_scripts_exist("run scripts/run_phase2_vllm.py then") == []
    # a fabricated script reference -> error
    assert check_repo.check_readme_scripts_exist("run scripts/totally_made_up.py")


def test_check_readme_bash_continuation():
    # an inline comment after a continuation backslash is a silent bug
    bad = "```bash\nfoo \\\\      # comment\nbar\n```\n"
    assert check_repo.check_readme_bash_continuation(bad)
    # trailing whitespace after the backslash breaks the continuation too
    bad2 = "```bash\nfoo \\\\   \nbar\n```\n"
    assert check_repo.check_readme_bash_continuation(bad2)
    # a clean continuation (backslash is the final char) is fine
    good = "```bash\nfoo \\\\\nbar\n```\n"
    assert check_repo.check_readme_bash_continuation(good) == []


def test_check_readme_quickstart_structure_flags_clone_after_export():
    # the contradiction: export BEFORE clone
    bad = (
        "export HUNYUAN_ROCM_DIR=$PWD\n"
        "git clone https://github.com/AIwork4me/HunyuanOCR-ROCm.git\n"
        "export LLAMA_DIR=x\nexport GGUF_DIR=x\nexport DATA_DIR=x\n"
        "python scripts/run_inference.py --x\n"
    )
    errs = check_repo.check_readme_quickstart_structure(bad)
    assert any("clone must come first" in e for e in errs)


def test_check_readme_quickstart_structure_flags_missing_envvar_and_legacy_cmd():
    bad = (
        "git clone https://github.com/AIwork4me/HunyuanOCR-ROCm.git\n"
        "export HUNYUAN_ROCM_DIR=$PWD\n"
        "export LLAMA_DIR=x\nexport GGUF_DIR=x\n"  # DATA_DIR missing
        "python scripts/run_phase2_vllm.py --x\n"  # legacy command
    )
    errs = check_repo.check_readme_quickstart_structure(bad)
    assert any("DATA_DIR" in e for e in errs)
    assert any("run_inference.py" in e for e in errs)  # canonical missing
    assert any("run_phase2_vllm.py" in e for e in errs)  # legacy present


def test_check_readme_quickstart_structure_happy_path():
    good = (
        "git clone https://github.com/AIwork4me/HunyuanOCR-ROCm.git\n"
        "cd HunyuanOCR-ROCm\n"
        "export HUNYUAN_ROCM_DIR=$PWD\n"
        "export LLAMA_DIR=x\nexport GGUF_DIR=x\nexport DATA_DIR=x\n"
        "python scripts/run_inference.py --backend-name llamacpp\n"
    )
    assert check_repo.check_readme_quickstart_structure(good) == []


def test_check_repo_clean_on_repo():
    """Integration: the real repo (after the README/Makefile/lock fixes) must pass
    every check — this is the gate CI enforces."""
    assert check_repo.all_errors() == []
