# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 AIwork4me
"""OmniDocBench v1.6 dataset iteration + prediction filename mapping.

Ground-truth JSON (e.g. /workspace/OmniDocBench_data/OmniDocBench.json) is a list
of page dicts; each page_info.image_path is a BARE basename resolved under the
dataset's images/ directory. Subsets OmniDocBench_150.json / OmniDocBench_30.json
share the same format.
"""

from __future__ import annotations
import json
from pathlib import Path
from typing import Iterator


def derive_prediction_filename(image_path: str | Path) -> str:
    """Map an image path to its OmniDocBench prediction filename: ``<stem>.md``."""
    return f"{Path(image_path).stem}.md"


def iter_page_images(gt_json: str | Path, images_dir: str | Path) -> Iterator[tuple[str, Path]]:
    """Yield (image_stem, abs_image_path) for every page in the ground-truth JSON.

    ``images_dir`` is the directory holding the page images (e.g. .../images).
    """
    images_dir = Path(images_dir)
    with open(gt_json, encoding="utf-8") as f:
        pages = json.load(f)
    for page in pages:
        rel = page["page_info"]["image_path"]
        abs_path = images_dir / rel
        yield Path(rel).stem, abs_path
