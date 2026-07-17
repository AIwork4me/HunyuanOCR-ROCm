from hunyuan_ocr.postprocess import (
    has_tail_repetition,
    clean_repeated_substrings,
    process_one,
)


def test_has_tail_repetition_detects_loop():
    assert has_tail_repetition("x" * 200) is True


def test_has_tail_repetition_clean_text():
    assert has_tail_repetition("正常的中文文档内容，没有重复。" * 1) is False


def test_clean_repeated_substrings_trims_long_loop():
    body = "正文内容。" * 5
    loop = "ABCD" * 3000  # >> 2000 chars, repeats > 10x
    out = clean_repeated_substrings(body + loop)
    # upstream keeps ONE surviving copy of the unit: text[: n - length*(count-1)]
    assert out == body + "ABCD"
    assert "ABCDABCD" not in out  # the degenerate loop is collapsed to one copy


def test_clean_repeated_substrings_short_text_untouched():
    assert clean_repeated_substrings("短文本") == "短文本"


def test_process_one_splits_table_caption():
    # Pattern T: <table><caption>X</caption>... -> X\n\n<table>...
    md = "<table><caption>表1 标题</caption><tr><td>a</td></tr></table>"
    out, stats = process_one(md)
    assert out.startswith("表1 标题\n\n<table>")
    assert stats["T_captions"] == 1


def test_process_one_idempotent_on_clean_doc():
    md = "# 标题\n\n这是一段正文。\n\n$$ a^2 + b^2 = c^2 $$\n"
    out, stats = process_one(md)
    assert out == md
    assert all(v == 0 for v in stats.values())
