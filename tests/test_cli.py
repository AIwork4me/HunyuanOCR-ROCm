# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 AIwork4me
"""CPU tests for the unified ``hunyuan-ocr`` CLI (no GPU, no torch required).

Covers --help, doctor (--json / --strict --backend), manifest verify on valid +
corrupt inputs, and that predict/score are wired through to the package (arg
parsing reaches the driver/scorer without needing scripts/).
"""

from __future__ import annotations

import json

import pytest

from hunyuan_ocr import cli, runner


def test_help_exits_zero(capsys):
    with pytest.raises(SystemExit) as ei:
        cli.main(["--help"])
    assert ei.value.code == 0
    out = capsys.readouterr().out
    assert "doctor" in out and "manifest" in out and "predict" in out and "score" in out


def test_doctor_json_shape(capsys):
    rc = cli.main(["doctor", "--json"])
    assert rc == 0  # advisory (no --backend) always exits 0
    payload = json.loads(capsys.readouterr().out)
    assert set(payload) >= {"ok", "backend", "checks", "environment"}
    assert payload["backend"] is None
    assert isinstance(payload["checks"], list) and payload["checks"]
    assert "python" in payload["environment"]
    # the environment summary must not echo anything secret-shaped
    blob = json.dumps(payload["environment"])
    assert "token" not in blob.lower() and "api_key" not in blob.lower()


def test_doctor_advisory_exits_zero(capsys):
    assert cli.main(["doctor"]) == 0
    assert "hunyuan-ocr doctor" in capsys.readouterr().out


def test_doctor_strict_llamacpp_fails_without_gguf(monkeypatch):
    monkeypatch.setattr(cli, "_has_rocm_toolchain", lambda: True)
    monkeypatch.setattr(cli.shutil, "which", lambda name: "/bin/llama-server" if name == "llama-server" else None)
    monkeypatch.setattr(cli, "_torch_hip", lambda: None)
    monkeypatch.setattr(cli, "_try_version", lambda m: "1.0" if m == "openai" else None)
    monkeypatch.delenv("HUNYUANOCR_GGUF_DIR", raising=False)
    monkeypatch.delenv("GGUF_DIR", raising=False)
    assert cli.main(["doctor", "--strict", "--backend", "llamacpp", "--json"]) == 1


def test_doctor_strict_llamacpp_passes_when_ready(monkeypatch, tmp_path):
    gguf = tmp_path / "gguf"
    gguf.mkdir()
    (gguf / "HunyuanOCR-bf16.gguf").write_bytes(b"x")
    (gguf / "mmproj-HunyuanOCR-bf16.gguf").write_bytes(b"x")
    monkeypatch.setattr(cli, "_has_rocm_toolchain", lambda: True)
    monkeypatch.setattr(cli.shutil, "which", lambda name: "/bin/llama-server" if name == "llama-server" else None)
    monkeypatch.setattr(cli, "_torch_hip", lambda: None)
    monkeypatch.setattr(cli, "_try_version", lambda m: "1.0" if m == "openai" else None)
    monkeypatch.setenv("HUNYUANOCR_GGUF_DIR", str(gguf))
    assert cli.main(["doctor", "--strict", "--backend", "llamacpp", "--json"]) == 0


def _write_manifest(pred_dir, manifest: dict | str):
    pred_dir.mkdir(parents=True, exist_ok=True)
    (pred_dir / "run_manifest.json").write_text(
        manifest if isinstance(manifest, str) else json.dumps(manifest), encoding="utf-8"
    )


def _valid_manifest():
    return {
        "schema_version": 2,
        "repo_commit": "abc123",
        "backend": "llamacpp",
        "model": "HYVL",
        "timestamp_iso": "2026-07-17T12:00:00+00:00",
        "status": "ok",
        "run_counts": {"attempted": 2, "succeeded": 2, "failed": 0, "skipped": 0, "interrupted": 0},
        "final_state": {"expected": 2, "complete": 2, "failed": 0, "pending": 0},
    }


def test_manifest_verify_ok(tmp_path, capsys):
    _write_manifest(tmp_path, _valid_manifest())
    assert cli.main(["manifest", "verify", "--pred-dir", str(tmp_path)]) == 0
    assert "[OK]" in capsys.readouterr().out


def test_manifest_verify_missing_dir(tmp_path, capsys):
    assert cli.main(["manifest", "verify", "--pred-dir", str(tmp_path / "nope")]) == 2
    assert "no run_manifest.json" in capsys.readouterr().err


def test_manifest_verify_empty_json(tmp_path, capsys):
    _write_manifest(tmp_path, "")
    assert cli.main(["manifest", "verify", "--pred-dir", str(tmp_path)]) == 1
    assert "empty" in capsys.readouterr().err


def test_manifest_verify_truncated_json(tmp_path, capsys):
    _write_manifest(tmp_path, '{"schema_version": 2, "backend": "llama')  # truncated
    assert cli.main(["manifest", "verify", "--pred-dir", str(tmp_path)]) == 1
    err = capsys.readouterr().err
    assert "not valid JSON" in err


def test_manifest_verify_missing_run_counts(tmp_path, capsys):
    m = _valid_manifest()
    del m["run_counts"]
    _write_manifest(tmp_path, m)
    assert cli.main(["manifest", "verify", "--pred-dir", str(tmp_path)]) == 1


def test_manifest_verify_status_ok_with_failed(tmp_path, capsys):
    m = _valid_manifest()
    m["final_state"]["failed"] = 1
    m["run_counts"]["failed"] = 1
    _write_manifest(tmp_path, m)
    assert cli.main(["manifest", "verify", "--pred-dir", str(tmp_path)]) == 1
    assert "status is 'ok'" in capsys.readouterr().out


def test_manifest_verify_writes_no_traceback_on_corrupt(tmp_path, capsys):
    # corrupt JSON must produce a friendly one-line error, never a Python traceback
    _write_manifest(tmp_path, "{not json")
    rc = cli.main(["manifest", "verify", "--pred-dir", str(tmp_path)])
    assert rc == 1
    combined = capsys.readouterr().err + capsys.readouterr().out
    assert "Traceback" not in combined


def test_predict_reaches_driver_arg_check(monkeypatch, tmp_path):
    # `predict` is self-contained: with no required driver args it must reach the
    # driver's argparse and fail with a usage error (SystemExit 2), not a
    # "scripts/ not found" message.
    import hunyuan_ocr.driver as driver_mod

    # ensure openai client resolves (it is installed in dev/CI)
    with pytest.raises(SystemExit) as ei:
        cli.main(["predict", "--backend", "llamacpp"])
    assert ei.value.code == 2
    del driver_mod  # silence linter


def test_score_invalid_preddir_is_friendly(tmp_path, capsys):
    # score must validate first and refuse (ScoringError -> friendly, rc 1),
    # never raise a traceback, even with no scorer venv configured.
    gt = tmp_path / "gt.json"
    gt.write_text(json.dumps([{"page_info": {"image_path": "a.png"}}]), encoding="utf-8")
    pred = tmp_path / "pred"
    pred.mkdir()
    rc = cli.main(["score", "--pred-dir", str(pred), "--gt-json", str(gt)])
    assert rc == 1
    out = capsys.readouterr()
    assert "Traceback" not in (out.out + out.err)


def test_canary_materialize_missing_inputs_friendly(tmp_path, capsys):
    rc = cli.main(
        [
            "canary",
            "materialize",
            "--full-gt",
            str(tmp_path / "no.json"),
            "--manifest",
            str(tmp_path / "no.json"),
            "--out",
            str(tmp_path / "o.json"),
        ]
    )
    assert rc == 1
    assert "Traceback" not in capsys.readouterr().err
    # keep runner import referenced (used by other tests in the module set)
    assert callable(runner.validate_manifest)
