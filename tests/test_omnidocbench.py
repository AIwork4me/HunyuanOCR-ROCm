from pathlib import Path

from hunyuan_ocr.omnidocbench import derive_prediction_filename, iter_page_images

FIX = Path(__file__).parent / "fixtures"


def test_derive_prediction_filename():
    assert derive_prediction_filename("images/page-aaaa-1111.png") == "page-aaaa-1111.md"
    assert derive_prediction_filename("/anywhere/PPT_eng_page_002.png") == "PPT_eng_page_002.md"


def test_iter_page_images_resolves_under_images_dir():
    pairs = list(iter_page_images(FIX / "mini_omnidocbench.json", FIX / "images"))
    stems = [s for s, _ in pairs]
    assert stems == ["page-aaaa-1111", "PPT_eng_page_002"]
    for _, p in pairs:
        assert p.exists()
        assert p.parent == (FIX / "images")
