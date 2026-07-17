from hunyuan_ocr.contract import CONTRACT, SAMPLING


def test_sampling_is_frozen_and_greedy():
    assert SAMPLING["do_sample"] is False  # temp=0 -> greedy
    assert SAMPLING["repetition_penalty"] == 1.08
    assert SAMPLING["max_new_tokens"] == 32768
    assert SAMPLING["use_cache"] is True


def test_contract_task_and_postprocessors():
    assert CONTRACT.task_type == "doc_parse"
    assert CONTRACT.prompt == (
        "提取文档图片中正文的所有信息用markdown格式表示，其中页眉、页脚部分忽略，"
        "表格用html格式表达，文档中公式用latex格式表示，按照阅读顺序组织进行解析。"
    )
    assert CONTRACT.postprocessors == ("clean_repeated_substrings", "process_one")


def test_contract_model_load_flags():
    assert CONTRACT.dtype == "bfloat16"
    assert CONTRACT.attn_implementation == "eager"
    assert CONTRACT.repeat_min_repeats == 8


def test_contract_decode_flags():
    assert CONTRACT.skip_special_tokens is True
    assert CONTRACT.clean_up_tokenization_spaces is False
