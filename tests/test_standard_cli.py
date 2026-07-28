# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 AIwork4me
"""Standard CLI contract tests (ADR-0011, central commit ccd466e).

Covers the core of the shared no-GPU test matrix: version/capabilities/doctor
JSON purity + schema validity, and `parse` page-conservation / partial-success /
exit-code behavior with a stubbed inference core (no GPU, no server, no model).
"""

from __future__ import annotations

import json
from pathlib import Path


def _run_main(capsys, *argv):
    """Invoke the CLI in-process (no subprocess); return (rc, stdout, stderr)."""
    from hunyuan_ocr.cli import main

    rc = main(list(argv))
    captured = capsys.readouterr()
    return rc, captured.out, captured.err


def _pure_json(stdout):
    return json.loads(stdout.strip())  # raises if not pure JSON


def test_version_json_is_pure_and_valid(capsys):
    rc, out, _ = _run_main(capsys, "version", "--json")
    assert rc == 0
    obj = _pure_json(out)
    assert obj["name"] == "hunyuan-ocr"
    assert isinstance(obj["version"], str) and obj["version"]


def test_capabilities_json_declares_platforms(capsys):
    rc, out, _ = _run_main(capsys, "capabilities", "--json")
    assert rc == 0
    obj = _pure_json(out)
    plats = obj["platforms"]
    assert plats and all("platform" in p and "backend" in p for p in plats)
    backends = {p["backend"] for p in plats}
    assert {"vllm", "llama-cpp"} <= backends


def test_doctor_json_has_central_status_field(capsys):
    rc, out, _ = _run_main(capsys, "doctor", "--json")
    assert rc == 0
    obj = _pure_json(out)
    assert obj["status"] in ("ready", "not-ready")


def test_version_stdout_has_no_log_noise(capsys):
    rc, out, _ = _run_main(capsys, "version", "--json")
    assert rc == 0
    _pure_json(out)  # stdout must be exactly one JSON doc (no banners/logs mixed in)


# --- parse (unit, stubbed inference core) ------------------------------------


def _stub_parse(monkeypatch, infer):
    """Wire cmd_parse to a stubbed client + inference core (no server/openai)."""
    from hunyuan_ocr import standard_cli

    monkeypatch.setattr(standard_cli, "_build_client", lambda url: ("stub-client", url))
    monkeypatch.setattr(standard_cli, "INFER", infer)
    return standard_cli


def _make_imgs(tmp_path, n=3):
    d = tmp_path / "imgs"
    d.mkdir()
    for i in range(n):
        (d / f"page_{i:04d}.png").write_bytes(b"\x89PNG\r\n\x1a\n")
    return d


def test_parse_ok_page_conservation_and_exit_zero(monkeypatch, tmp_path, capsys):
    imgs = _make_imgs(tmp_path, 3)
    out = tmp_path / "out"
    mod = _stub_parse(monkeypatch, lambda client, img, **kw: f"# md {Path(img).name}")
    rc = mod.cmd_parse(
        img_dir=imgs,
        out_dir=out,
        platform="linux-rocm",
        backend="vllm",
        server_url="http://x",
        model=None,
        max_pixels=None,
        limit=None,
    )
    assert rc == mod.EXIT_OK
    obj = _pure_json(capsys.readouterr().out)
    assert obj["status"] == "ok"
    assert obj["page_count"] == 3 and obj["ok"] == 3 and obj["failed"] == 0
    assert len(obj["pages"]) == 3  # conservation: every page recorded exactly once
    assert obj["backend"] == "vllm"
    assert all((out / f"page_{i:04d}.md").exists() for i in range(3))


def test_parse_partial_continues_and_exit_one(monkeypatch, tmp_path, capsys):
    imgs = _make_imgs(tmp_path, 4)

    def infer(client, img, **kw):
        if "page_0001" in img or "page_0003" in img:
            raise RuntimeError("boom")
        return "# ok"

    out = tmp_path / "out"
    mod = _stub_parse(monkeypatch, infer)
    rc = mod.cmd_parse(
        img_dir=imgs,
        out_dir=out,
        platform="linux-rocm",
        backend="vllm",
        server_url="http://x",
        model=None,
        max_pixels=None,
        limit=None,
    )
    assert rc == mod.EXIT_PARTIAL
    obj = _pure_json(capsys.readouterr().out)
    assert obj["status"] == "partial"
    assert obj["ok"] == 2 and obj["failed"] == 2
    assert obj["page_count"] == 4  # conservation: failed pages did not disappear
    assert not (out / "page_0001.md").exists()  # failed page leaves no empty .md


def test_parse_missing_img_dir_is_usage_error(monkeypatch, tmp_path):
    mod = _stub_parse(monkeypatch, lambda *a, **k: "")
    rc = mod.cmd_parse(
        img_dir=tmp_path / "nope",
        out_dir=tmp_path / "o",
        platform="linux-rocm",
        backend="vllm",
        server_url="http://x",
        model=None,
        max_pixels=None,
        limit=None,
    )
    assert rc == mod.EXIT_USAGE


def test_parse_transformers_backend_unreachable_from_standard_path(monkeypatch, tmp_path):
    imgs = _make_imgs(tmp_path, 1)
    mod = _stub_parse(monkeypatch, lambda *a, **k: "")
    rc = mod.cmd_parse(
        img_dir=imgs,
        out_dir=tmp_path / "o",
        platform="linux-rocm",
        backend="transformers",
        server_url=None,
        model=None,
        max_pixels=None,
        limit=None,
    )
    assert rc == mod.EXIT_USAGE
