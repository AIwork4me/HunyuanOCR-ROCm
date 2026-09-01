# 探针 I：stale 槽位 A/B 因果验证。
# g0 阶段：对 sw==-1（NoPE/custom kernel）decode 调用，事后保存最后块 stale 槽
# （block 512 的 slots 7..15，K+V）字节。
# g1 阶段：真实调用得 o1 后，把 g0 的 stale 字节写入 cache，再用相同 q/table/seq
# 重调一次得 o2，恢复 g1 stale。若 o2 == g0 的 o ⇒ stale 字节因果决定输出。
import hashlib
import json

import torch

import vllm.v1.attention.backends.rocm_attn as ra
from vllm.v1.attention.ops.chunked_prefill_paged_decode import (
    chunked_prefill_paged_decode as REAL_ATTN)

REC = {"on": False, "phase": None, "saved": {}, "log": []}


def sha(t):
    if t is None or not torch.is_tensor(t) or t.numel() == 0:
        return "e"
    return hashlib.sha256(
        t.detach().contiguous().cpu().view(torch.uint8).numpy().tobytes()
    ).hexdigest()[:16]


def rec_attn(query, key, value, output, **kw):
    do_ab = (REC["on"] and query.shape[0] == 1
             and int(kw.get("sliding_window") or 0) == -1
             and key is not None)
    if not do_ab:
        return REAL_ATTN(query=query, key=key, value=value, output=output,
                         **kw)
    sl = kw["seq_lens"]
    bt = kw["block_table"]
    kc, vc = kw["key_cache"], kw["value_cache"]
    s0 = int(sl[0])
    bs = kc.shape[3]
    nblk = (s0 + bs - 1) // bs
    last_id = int(bt[0, nblk - 1])
    blk_k = kc[last_id].clone()
    blk_v = vc[last_id].clone()
    out = REAL_ATTN(query=query, key=key, value=value, output=output, **kw)
    ent = {"s0": s0, "last": last_id, "rem": s0 - (nblk - 1) * bs,
           "o": sha(output), "q": sha(query), "kblk": sha(blk_k)}
    if REC["phase"] == "g0":
        REC["saved"]["k"] = blk_k
        REC["saved"]["v"] = blk_v
        ent["role"] = "save"
    elif REC["phase"] == "g1" and blk_k.shape == REC["saved"]["k"].shape \
            and sha(blk_k) != sha(REC["saved"]["k"]):
        # A/B：整块换成 g0 的（有效槽相同 ⇒ 实际只换 stale 字节），重调一次
        kc[last_id] = REC["saved"]["k"]
        vc[last_id] = REC["saved"]["v"]
        out2 = torch.empty_like(output)
        REAL_ATTN(query=query, key=key, value=value, output=out2, **kw)
        kc[last_id] = blk_k
        vc[last_id] = blk_v
        ent["role"] = "ab"
        ent["o_swap"] = sha(out2)
    REC["log"].append(ent)
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
        REC["phase"] = tag
        REC["log"] = []
        if tag == "g0":
            REC["saved"] = {}
    out = llm.generate([p], sp)[0].outputs[0].token_ids
    if tag:
        REC["on"] = False
        json.dump({"tag": tag, "tokens": list(out), "log": REC["log"]},
                  open(f"{LOGD}/stale-{tag}.json", "w"))
        print(tag, "nlog", len(REC["log"]), flush=True)
    return list(out)


gen(512, 8)
for i in range(8):
    gen(512, 64)
gen(8192, 8)
t0 = gen(8192, 64, "g0")
t1 = gen(8192, 64, "g1")
print("TOK0==TOK1", t0 == t1, flush=True)

A = json.load(open(f"{LOGD}/stale-g0.json"))["log"]
B = json.load(open(f"{LOGD}/stale-g1.json"))["log"]
n = min(len(A), len(B))
checks = 0
agree = 0
first_o = None
for i in range(n):
    if A[i]["o"] != B[i]["o"] and first_o is None:
        first_o = i
    if B[i].get("role") == "ab":
        checks += 1
        match_g0 = B[i]["o_swap"] == A[i]["o"]
        if checks <= 6 or (first_o is not None and i >= first_o
                           and checks <= 20):
            print(f"  #{i} s0={B[i]['s0']} rem={B[i]['rem']} "
                  f"o1_neq_o0={A[i]['o'] != B[i]['o']} "
                  f"swap_matches_g0={match_g0}", flush=True)
        agree += int(match_g0)
print(f"AB checks={checks} swap_matches_g0={agree} first_o_diff={first_o}",
      flush=True)
print("STALE DONE", flush=True)
