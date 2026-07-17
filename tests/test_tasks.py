import pytest
from hunyuan_ocr.tasks import TASK_PROMPTS, get_prompt, DEFAULT_TASK


def test_doc_parse_prompt_matches_upstream():
    assert get_prompt("doc_parse") == (
        "提取文档图片中正文的所有信息用markdown格式表示，其中页眉、页脚部分忽略，"
        "表格用html格式表达，文档中公式用latex格式表示，按照阅读顺序组织进行解析。"
    )


def test_default_task_is_doc_parse():
    assert DEFAULT_TASK == "doc_parse"


def test_get_prompt_unknown_raises():
    with pytest.raises(KeyError):
        get_prompt("nope")


def test_all_twelve_tasks_present():
    expected = {
        "doc_parse",
        "structured_parse",
        "spotting_json",
        "spotting_hunyuan",
        "layout",
        "layout_parse",
        "chart_parse",
        "formula",
        "table",
        "doc_trans_en2zh",
        "trans_other2en",
        "trans_other2zh",
    }
    assert expected == set(TASK_PROMPTS.keys())
