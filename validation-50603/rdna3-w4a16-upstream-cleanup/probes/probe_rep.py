# 探针 J：custom paged attention 同输入重复调用确定性测试。
# 对每个 sw==-1（NoPE→custom kernel）decode 调用：真实调用后立即用
# 完全相同的参数再调一次（新 output buffer），比较两次输出哈希。
import hashlib
import json

import torch

import vllm.v1.attention.backends.rocm_attn as ra
from vllm.v1.attention.ops.chunked_prefill_paged_decode import (
    chunked_prefill_paged_decode as REAL_ATTN)

REC = {"on": False, "log": []}


def sha(t):
    if t is None or not torch.is_tensor(t) or t.numel() == 0:
        return "e"
    return hashlib.sha256(
        t.detach().contiguous().cpu().view(torch.uint8).numpy().tobytes()
    ).hexdigest()[:16]


def rec_attn(query, key, value, output, **kw):
    do_rep = (REC["on"] and query.shape[0] == 1
              and int(kw.get("sliding_window") or 0) == -1)
    out = REAL_ATTN(query=query, key=key, value=value, output=output, **kw)
    if do_rep:
        out2 = torch.empty_like(output)
        REAL_ATTN(query=query, key=key, value=value, output=out2, **kw)
        REC["log"].append({"s0": int(kw["seq_lens"][0]),
                           "o1": sha(output), "o2": sha(out2),
                           "q": sha(query)})
    return out


ra.chunked_prefill_paged_decode = rec_attn

from vllm import LLM, SamplingParams  # noqa: E402
from vllm.inputs import TokensPrompt  # noqa: E402

llm = LLM(model="/workspace/vllm-50603-version-ab/models/muse",
          tensor_parallel_size=1, max_model_len=8704, max_num_seqs=128,
          gpu_memory_utilization=0.92, disable_log_stats=True,
          enforce_eager=True, limit_mm_per_prompt={"image": 0, "video": 0})

LOGD = "/workspace/vllm-50603-upstream-cleanup/logs"


def gen(depth, n, tag=None):
    p = TokensPrompt(prompt_token_ids=[1000 + (i % 20000) for i in range(depth)])
    sp = SamplingParams(max_tokens=n, temperature=0.0, ignore_eos=True)
    if tag:
        REC["on"] = True
        REC["log"] = []
    out = llm.generate([p], sp)[0].outputs[0].token_ids
    if tag:
        REC["on"] = False
        json.dump({"tag": tag, "tokens": list(out), "log": REC["log"]},
                  open(f"{LOGD}/rep-{tag}.json", "w"))
        print(tag, "nrep", len(REC["log"]), flush=True)
    return list(out)


gen(512, 8)
for i in range(8):
    gen(512, 64)
gen(8192, 8)
t0 = gen(8192, 64, "g0")
t1 = gen(8192, 64, "g1")
print("TOK0==TOK1", t0 == t1, flush=True)

for tag in ("g0", "g1"):
    L = json.load(open(f"{LOGD}/rep-{tag}.json"))["log"]
    bad = [i for i, e in enumerate(L) if e["o1"] != e["o2"]]
    print(f"{tag}: reps={len(L)} same-input-differs={len(bad)} "
          f"first={bad[:5]}", flush=True)
    for i in bad[:3]:
        print(f"   #{i} s0={L[i]['s0']}", flush=True)
print("REP DONE", flush=True)
