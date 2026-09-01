# 探针（53856 A/B 用）：stale V-tail 因果测试。
# 对每个 NoPE（sw==-1）decode attention 调用：
#   1) 引擎真实调用得 o1（哈希）
#   2) 快照末引用块的 V-tail（slots [rem:16]），统计 NaN/Inf/有限值范围（Phase 7）
#   3) 将 V-tail 全部改为 +1.0（bf16 0x3F80，有限值）
#   4) 用完全相同参数重调（新输出缓冲）得 o2
#   5) 位级比较 o1/o2（SHA）+ 数值 max|Δ| + 首个不同元素（Phase 4）
#   6) 恢复 V-tail
# 另统计 custom kernel 路由（包装 _custom_ops.paged_attention_rocm 计数，Phase 5）。
import hashlib
import json

import torch

import vllm._custom_ops as real_ops
import vllm.v1.attention.backends.rocm_attn as ra
from vllm.v1.attention.ops.chunked_prefill_paged_decode import (
    chunked_prefill_paged_decode as REAL_ATTN)

REC = {"on": False, "log": [], "custom_calls": 0, "import_from": None}

_orig_par = real_ops.paged_attention_rocm


def counting_par(*a, **kw):
    REC["custom_calls"] += 1
    return _orig_par(*a, **kw)


real_ops.paged_attention_rocm = counting_par


def sha(t):
    if t is None or not torch.is_tensor(t) or t.numel() == 0:
        return "e"
    return hashlib.sha256(
        t.detach().contiguous().cpu().view(torch.uint8).numpy().tobytes()
    ).hexdigest()[:16]


def rec_attn(query, key, value, output, **kw):
    do_ab = (REC["on"] and query.shape[0] == 1
             and int(kw.get("sliding_window") or 0) == -1
             and key is not None and kw.get("key_cache") is not None)
    if not do_ab:
        return REAL_ATTN(query=query, key=key, value=value, output=output,
                         **kw)
    vc = kw["value_cache"]
    bt = kw["block_table"]
    s0 = int(kw["seq_lens"][0])
    bs = vc.shape[3]
    nblk = (s0 + bs - 1) // bs
    last_id = int(bt[0, nblk - 1])
    rem = s0 - (nblk - 1) * bs
    out = REAL_ATTN(query=query, key=key, value=value, output=output, **kw)
    if 0 < rem < bs:  # 只有末块未满才有 stale 尾巴
        vblk = vc[last_id]
        tail = vblk[..., rem:]
        tf = tail.detach().float()
        ent = {
            "s0": s0, "last": last_id, "rem": rem,
            "tail_nan": int(torch.isnan(tf).sum()),
            "tail_inf": int(torch.isinf(tf).sum()),
            "tail_min": (float(tf.min()) if torch.isfinite(tf).all()
                         else float("nan")),
            "tail_max": (float(tf.max()) if torch.isfinite(tf).all()
                         else float("nan")),
            "o1": sha(output),
        }
        o1 = output.detach().float().cpu().clone()
        snap = tail.clone()
        tail.fill_(1.0)  # 变异：有限值 +1.0
        out2 = torch.empty_like(output)
        REAL_ATTN(query=query, key=key, value=value, output=out2, **kw)
        tail.copy_(snap)  # 恢复
        ent["o2"] = sha(out2)
        ent["changed"] = ent["o1"] != ent["o2"]
        if ent["changed"]:
            o2 = out2.detach().float().cpu()
            d = (o2 - o1).abs()
            nz = (d > 0).nonzero()
            ent["max_abs"] = float(d.max())
            ent["n_diff"] = int((d > 0).sum())
            ent["first_elem"] = nz[0].tolist() if len(nz) else None
        REC["log"].append(ent)
    return out


ra.chunked_prefill_paged_decode = rec_attn

import vllm  # noqa: E402

REC["import_from"] = vllm.__file__

from vllm import LLM, SamplingParams  # noqa: E402
from vllm.inputs import TokensPrompt  # noqa: E402

llm = LLM(model="/workspace/vllm-50603-version-ab/models/muse",
          tensor_parallel_size=1, max_model_len=8704, max_num_seqs=128,
          gpu_memory_utilization=0.92, disable_log_stats=True,
          enforce_eager=True, limit_mm_per_prompt={"image": 0, "video": 0})

OUT = None
TAG = None


def gen(depth, n):
    p = TokensPrompt(prompt_token_ids=[1000 + (i % 20000) for i in range(depth)])
    sp = SamplingParams(max_tokens=n, temperature=0.0, ignore_eos=True)
    return list(llm.generate([p], sp)[0].outputs[0].token_ids)


gen(512, 8)
for _ in range(8):
    gen(512, 64)
gen(8192, 8)  # warmup（不记录）
REC["on"] = True
REC["log"] = []
REC["custom_calls"] = 0
tokens = gen(8192, 64)
REC["on"] = False

import os  # noqa: E402

out_path = os.environ["STALE_OUT"]
json.dump({"tag": os.environ.get("STALE_TAG", "?"),
           "import_from": REC["import_from"],
           "custom_paged_attn_calls": REC["custom_calls"],
           "tokens": tokens, "log": REC["log"]},
          open(out_path, "w"))
tested = len(REC["log"])
changed = sum(1 for e in REC["log"] if e["changed"])
nan_t = sum(e["tail_nan"] for e in REC["log"])
inf_t = sum(e["tail_inf"] for e in REC["log"])
mx = max((e.get("max_abs", 0.0) for e in REC["log"]), default=0.0)
print("IMPORT:", REC["import_from"], flush=True)
print("CUSTOM_CALLS:", REC["custom_calls"], flush=True)
print(f"STALE-AB: tested={tested} changed={changed} "
      f"nan={nan_t} inf={inf_t} max_abs_delta={mx:.3e}", flush=True)
