"""TP=1 port of cadamcat's nondet_eager.py (dual-radeon-vllm @ 782c7d4).

Measures greedy-decoding determinism of one vLLM engine: per context depth,
one max_tokens=8 warm-up then eight identical greedy generations of 64 tokens,
all token sequences + SHA256s + decoded text saved to JSON.

Executable differences vs upstream nondet_eager.py are exactly:
  1. tensor_parallel_size 2 -> 1 (the axis under test; W7900D has 48 GB)
  2. model paths -> pinned local HF snapshots of the same repositories
  3. richer JSON evidence block (hashes, decoded text, env/version metadata)
  4. explicit engine teardown (del + gc + empty_cache) before process exit
Generation semantics (prompt, warm-up, sampling, repeats, engine kwargs other
than TP) are byte-identical to upstream. See artifacts/harness.diff.

Usage: nondet_eager_tp1.py <muse|gemma3> <engine-id> <0|1 eager> <out-json>
"""
import gc
import hashlib
import json
import os
import subprocess
import sys

MODELS = {
    "muse": "/workspace/vllm-50603-version-ab/models/muse",
    "gemma3": "/workspace/vllm-50603-version-ab/models/gemma3",
}
MODEL_REVISIONS = {
    "muse": "RedHatAI/Muse-Glimmer-30B-INT4@f5b410ce4234fad70eef8be99b4680ee4e30b418",
    "gemma3": "RedHatAI/gemma-3-27b-it-quantized.w4a16@2b537554d6c6f6368945e8df4e5fb7bbbb5d56c9",
}
DEPTHS = [512, 8192]
NGEN = 64
REPEATS = 8
os.environ.setdefault("VLLM_ROCM_CLONE_MMAP_WEIGHTS", "1")
os.environ.pop("VLLM_CLONE_MMAP", None)


def sha256_list(ids):
    payload = ",".join(str(i) for i in ids).encode()
    return hashlib.sha256(payload).hexdigest()


def collect_environment():
    import torch
    import vllm
    env = {
        "vllm_version": vllm.__version__,
        "vllm_path": os.path.dirname(os.path.abspath(vllm.__file__)),
        "torch_version": torch.__version__,
        "torch_hip": getattr(torch.version, "hip", None),
        "device_name": torch.cuda.get_device_name(0),
        "device_capability": ".".join(map(str, torch.cuda.get_device_capability(0))),
        "device_count": torch.cuda.device_count(),
        "python": sys.version.split()[0],
        "env": {k: os.environ.get(k) for k in
                ("VLLM_ROCM_CLONE_MMAP_WEIGHTS", "VLLM_CLONE_MMAP",
                 "HSA_OVERRIDE_GFX_VERSION", "VLLM_USE_V1", "VLLM_WORKER_MULTIPROC_METHOD")},
    }
    try:
        env["vllm_git_sha"] = subprocess.run(
            ["git", "-C", env["vllm_path"], "rev-parse", "HEAD"],
            capture_output=True, text=True, timeout=10).stdout.strip()
        env["vllm_git_dirty"] = bool(subprocess.run(
            ["git", "-C", env["vllm_path"], "status", "--short"],
            capture_output=True, text=True, timeout=10).stdout.strip())
    except Exception as e:  # not a git checkout (wheel install)
        env["vllm_git_sha"] = f"unavailable: {e}"
    try:
        env["rocm_product"] = subprocess.run(
            ["rocm-smi", "--showproductname", "--showmeminfo", "vram"],
            capture_output=True, text=True, timeout=15).stdout.strip()
    except Exception as e:
        env["rocm_product"] = f"unavailable: {e}"
    return env


def main():
    which, engine_id, eager, out_path = (
        sys.argv[1], sys.argv[2], bool(int(sys.argv[3])), sys.argv[4])
    from vllm import LLM, SamplingParams
    from vllm.inputs import TokensPrompt

    env = collect_environment()
    print(f"[env] vllm={env['vllm_version']} torch={env['torch_version']} "
          f"hip={env['torch_hip']} dev={env['device_name']}", flush=True)

    # AB_MM_LIMIT_ZERO=1 passes limit_mm_per_prompt={"image":0,"video":0}.
    # Environment-compat necessity for the 0.23.1.dev1 arm (see harness.diff):
    # without the AMD flash_attn package (present in cadamcat's rocm/vllm
    # container, not in the wheel stack), 0.23.1.dev1 profiles the ViT encoder
    # with 2 max-size image items and Torch-SDPA materializes a 16 GiB
    # attention matrix -> OOM at engine init on the 48 GB card. The measured
    # path is text-only synthetic tokens; the ViT never runs at generate time.
    # Applied symmetrically to BOTH arms for the primary comparison.
    mm_limit = ({"image": 0, "video": 0}
                if os.environ.get("AB_MM_LIMIT_ZERO") == "1" else None)
    llm = LLM(model=MODELS[which], tensor_parallel_size=1,
              max_model_len=max(DEPTHS) + 512, max_num_seqs=128,
              gpu_memory_utilization=0.92, disable_log_stats=True,
              enforce_eager=eager, limit_mm_per_prompt=mm_limit)
    print(f"[engine] limit_mm_per_prompt={mm_limit}", flush=True)
    sp = SamplingParams(max_tokens=NGEN, temperature=0.0, ignore_eos=True)
    warmup_sp = SamplingParams(max_tokens=8, temperature=0.0, ignore_eos=True)
    print(f"[sampling] measured={sp} warmup={warmup_sp}", flush=True)
    tokenizer = llm.get_tokenizer()

    cells = []
    for d in DEPTHS:
        p = TokensPrompt(prompt_token_ids=[1000 + (i % 20000) for i in range(d)])
        llm.generate([p], warmup_sp)
        seqs, texts = [], []
        for i in range(REPEATS):
            out = llm.generate([p], sp)
            seqs.append(list(out[0].outputs[0].token_ids))
            texts.append(out[0].outputs[0].text)
        distinct = {tuple(s) for s in seqs}
        first = [next((j for j, (x, y) in enumerate(zip(seqs[0], s)) if x != y), None)
                 for s in seqs[1:]]
        cells.append({
            "context": d, "warmups": 1, "measured_generations": REPEATS,
            "unique_outputs": len(distinct),
            "hashes": [sha256_list(s) for s in seqs],
            "token_ids": seqs, "decoded_text": texts,
            "first_divergence_vs_run0": first,
        })
        print(f"[within] {which} tp=1 eager={int(eager)} eng={engine_id} ctx={d:>6} -> "
              f"{len(distinct)} distinct of {REPEATS}  first_div={first}", flush=True)

    del llm
    gc.collect()
    import torch
    torch.cuda.empty_cache()

    with open(out_path, "w") as f:
        json.dump({
            "environment": env,
            "model": which,
            "model_source": MODELS[which],
            "model_revision": MODEL_REVISIONS[which],
            "vllm_version": env["vllm_version"],
            "vllm_git_sha": env.get("vllm_git_sha"),
            "torch_version": env["torch_version"],
            "tp": 1,
            "enforce_eager": eager,
            "engine_id": int(engine_id),
            "cells": cells,
        }, f, indent=1)
    print("=== NONDET DONE ===", flush=True)


if __name__ == "__main__":
    main()
