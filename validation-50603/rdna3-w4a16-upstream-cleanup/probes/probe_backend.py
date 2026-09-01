# 实验 2：强制 TRITON_ATTN backend 的两段探针（当前 clean build，原 dispatch）。
import sys

from vllm import LLM, SamplingParams
from vllm.inputs import TokensPrompt

backend = sys.argv[1] if len(sys.argv) > 1 else "TRITON_ATTN"
llm = LLM(model="/workspace/vllm-50603-version-ab/models/muse",
          tensor_parallel_size=1, max_model_len=8704, max_num_seqs=128,
          gpu_memory_utilization=0.92, disable_log_stats=True,
          enforce_eager=True, limit_mm_per_prompt={"image": 0, "video": 0},
          attention_backend=backend)


def phase(depth):
    p = TokensPrompt(prompt_token_ids=[1000 + (i % 20000) for i in range(depth)])
    llm.generate([p], SamplingParams(max_tokens=8, temperature=0.0, ignore_eos=True))
    sp = SamplingParams(max_tokens=64, temperature=0.0, ignore_eos=True)
    seqs = [list(llm.generate([p], sp)[0].outputs[0].token_ids) for _ in range(8)]
    print(depth, len({tuple(s) for s in seqs}), flush=True)


phase(512)
phase(8192)
print("PROBE-BACKEND DONE", backend, flush=True)
