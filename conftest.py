# conftest.py
"""Make the in-tree `src/` layout importable for pytest without an editable install.

The package is NOT installed editable in CI/dev; this adds `src` to sys.path so
`pytest -q` (the documented acceptance command) collects tests that import
`hunyuan_ocr`.
"""
import sys
from pathlib import Path

_SRC = Path(__file__).resolve().parent / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))
