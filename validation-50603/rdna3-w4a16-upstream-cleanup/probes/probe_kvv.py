# 探针 H：valid-only KV 哈希 + 分叉点 o 张量数值 diff。
# - kh/vh 只哈希 [0, nblk-1) 完整块 + 最后块的前 rem 个槽（排除 stale padding）
# - 对 step 5-7 的所有层保存 o 张量（CPU 拷贝），事后数值 diff（NaN? ULP?）
import hashlib
import json

import torch

import vllm.v1.attention.backends.rocm_attn as ra
from vllm.v1.attention.ops.chunked_prefill_paged_decode import (
    chunked_prefill_paged_decode as REAL_ATTN)

REC = {"on": False, "log": [], "otensors": []}


def sha(t):
    if t is None or not torch.is_tensor(t) or t.numel() == 0:
        return "e"
    return hashlib.sha256(
        t.detach().contiguous().cpu().view(torch.uint8).numpy().tobytes()
    ).hexdigest()[:16]


def valid_kv_hash(cache, ids):
    # 完整块（全部槽位均为有效逻辑位置）与最后一块分开哈希，
    # 避开不同 layer 布局差异；seq 的完整块 = ids[:-1]（decode 时最后一块含 append）。
    kreg = cache.detach().index_select(0, ids)
    return sha(kreg[:-1]), sha(kreg[-1:])


def rec_attn(query, key, value, output, **kw):
    rec = None
    if REC["on"] and query.shape[0] == 1:
        sl = kw.get("seq_lens")
        bt = kw.get("block_table")
        kc = kw.get("key_cache")
        vc = kw.get("value_cache")
        if sl is not None and bt is not None and kc is not None:
            s0 = int(sl[0])
            bs = kc.shape[-1] if kc.dim() == 4 else kc.shape[3]
            nblk = (s0 + bs - 1) // bs
            ids = bt[0, :nblk].long()
            kh, khl = valid_kv_hash(kc, ids)
            vh, vhl = valid_kv_hash(vc, ids)
            rec = {"q": sha(query), "kh": kh, "vh": vh,
                   "khl": khl, "vhl": vhl, "nblk": nblk}
    out = REAL_ATTN(query=query, key=key, value=value, output=output, **kw)
    if rec is not None:
        rec["o"] = sha(output)
        REC["log"].append(rec)
        REC["otensors"].append(output.detach().float().cpu().clone())
    return out


ra.chunked_prefill_paged_decode = rec_attn

from vllm import LLM, SamplingParams  # noqa: E402
from vllm.inputs import TokensPrompt  # noqa: E402

llm = LLM(model="/workspace/vllm-50603-version-ab/models/muse",
          tensor_parallel_size=1, max_model_len=8704, max_num_seqs=128,
          gpu_memory_utilization=0.92, disable_log_stats=True,
          enforce_eager=True, limit_mm_per_prompt={"image": 0, "video": 0})

LOGD = "/workspace/vllm-50603-upstream-cleanup/logs"
OT = {}


def gen(depth, n, tag=None):
    p = TokensPrompt(prompt_token_ids=[1000 + (i % 20000) for i in range(depth)])
    sp = SamplingParams(max_tokens=n, temperature=0.0, ignore_eos=True)
    if tag:
        REC["on"] = True
        REC["log"] = []
        REC["otensors"] = []
    out = llm.generate([p], sp)[0].outputs[0].token_ids
    if tag:
        REC["on"] = False
        json.dump({"tag": tag, "tokens": list(out), "log": REC["log"]},
                  open(f"{LOGD}/kvv-{tag}.json", "w"))
        OT[tag] = torch.stack(REC["otensors"])  # [ncalls, 1, 4096]
        print(tag, "nlog", len(REC["log"]), flush=True)
    return list(out)


gen(512, 8)
for i in range(8):
    gen(512, 64)
gen(8192, 8)
t0 = gen(8192, 64, "g0")
t1 = gen(8192, 64, "g1")
print("TOK0==TOK1", t0 == t1, flush=True)

A = json.load(open(f"{LOGD}/kvv-g0.json"))["log"]
B = json.load(open(f"{LOGD}/kvv-g1.json"))["log"]
n = min(len(A), len(B))
first = {}
for field in ("q", "kh", "vh", "khl", "vhl", "o"):
    for i in range(n):
        if A[i][field] != B[i][field]:
            first[field] = i
            break
    first.setdefault(field, None)
print("first(valid-blocks/last-block split):", first, flush=True)

if first.get("o") is not None:
    i = first["o"]
    print(f"  o@{i}: step={i // 52} layer={i % 52} nblk={A[i]['nblk']}", flush=True)
    oa, ob = OT["g0"][i], OT["g1"][i]
    d = (oa - ob).abs()
    nan_a = int(torch.isnan(oa).sum())
    nan_b = int(torch.isnan(ob).sum())
    nz = int((d > 0).sum())
    print(f"  o-diff: elements_differing={nz}/{oa.numel()} max_abs={d.max():.3e} "
          f"nan_a={nan_a} nan_b={nan_b}", flush=True)
    if nz:
        idx = (d > 0).nonzero()[:5]
        for ix in idx:
            k = tuple(ix.tolist())
            print(f"    at {k}: g0={oa[k].item():.6f} g1={ob[k].item():.6f} "
                  f"delta={d[k].item():.3e}", flush=True)
print("KVV DONE", flush=True)
