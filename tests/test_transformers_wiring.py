from hunyuan_ocr.backends.transformers import build_messages
from hunyuan_ocr.contract import CONTRACT


def test_build_messages_matches_upstream_shape():
    msgs = build_messages("/x/y/page-1.png", CONTRACT.prompt)
    assert msgs[0] == {"role": "system", "content": ""}
    user = msgs[1]
    assert user["role"] == "user"
    assert {"type": "image", "image": "/x/y/page-1.png"} in user["content"]
    assert {"type": "text", "text": CONTRACT.prompt} in user["content"]
