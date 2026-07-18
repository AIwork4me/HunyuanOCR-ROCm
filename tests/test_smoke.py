# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 AIwork4me


def _pyproject_version() -> str:
    import re
    from pathlib import Path

    text = (Path(__file__).resolve().parents[1] / "pyproject.toml").read_text(encoding="utf-8")
    m = re.search(r'^version\s*=\s*"([^"]+)"', text, re.MULTILINE)
    assert m, "could not find version in pyproject.toml"
    return m.group(1)


def test_package_imports_and_version():
    import hunyuan_ocr

    # The package version must match the single source of truth in pyproject.toml,
    # so a version bump never leaves the two out of sync.
    assert hunyuan_ocr.__version__ == _pyproject_version()
