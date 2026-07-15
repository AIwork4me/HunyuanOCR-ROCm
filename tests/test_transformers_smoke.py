import os
import pytest

MODEL_PATH = os.environ.get("HUNYUANOCR_MODEL", "/root/models/HunyuanOCR")
SAMPLE_IMG = os.environ.get("HUNYUANOCR_SAMPLE_IMG", "")


@pytest.mark.skipif(
    not os.path.isdir(MODEL_PATH) or not SAMPLE_IMG,
    reason="needs HUNYUANOCR_MODEL dir + HUNYUANOCR_SAMPLE_IMG on a gfx1100 box",
)
def test_infer_one_returns_markdown():
    from hunyuan_ocr.backends.transformers import load_model_and_processor, infer_one
    from hunyuan_ocr.contract import CONTRACT
    model, processor = load_model_and_processor(MODEL_PATH)
    md = infer_one(model, processor, SAMPLE_IMG, CONTRACT.prompt)
    assert isinstance(md, str) and len(md) > 0
