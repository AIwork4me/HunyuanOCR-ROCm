"""Phase 2 fit check: real TP=1 engine initialization + one tiny generation.

Verifies each model can actually load and run at TP=1 on the 48 GB W7900D in
the 0.25.1 environment, and captures VRAM evidence. Engine kwargs identical to
the measurement harness.
Usage: fit_check.py <muse|gemma3> <log-tag>
"""
import json
import subprocess
import sys

MODELS = {
    "muse": "/workspace/vllm-50603-version-ab/models/muse",
    "gemma3": "/workspace/vllm-50603-version-ab/models/gemma3",
}

def vram_used():
    try:
        out = subprocess.run(["rocm-smi", "--showmeminfo", "vram", "--csv"],
                             capture_output=True, text=True, timeout=15).stdout
        for line in out.splitlines():
            if line.startswith("card"):
                return int(line.split(",")[2])
        return None
    except Exception as e:
        return f"unavailable: {e}"

which = sys.argv[1]
from vllm import LLM, SamplingParams
from vllm.inputs import TokensPrompt

print(f"[fit] model={which} path={MODELS[which]} vram_before={vram_used()}", flush=True)
llm = LLM(model=MODELS[which], tensor_parallel_size=1,
          max_model_len=8192 + 512, max_num_seqs=128,
          gpu_memory_utilization=0.92, disable_log_stats=True)
print(f"[fit] engine up vram_after_load={vram_used()}", flush=True)
out = llm.generate([TokensPrompt(prompt_token_ids=[1000 + (i % 20000) for i in range(32)])],
                   SamplingParams(max_tokens=4, temperature=0.0, ignore_eos=True))
ids = list(out[0].outputs[0].token_ids)
print(f"[fit] generate ok tokens={ids} vram_after_gen={vram_used()}", flush=True)
print("[fit] done")
del llm
print("=== FIT OK ===", flush=True)
