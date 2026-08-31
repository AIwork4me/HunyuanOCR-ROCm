"""First-divergence logit forensics — instrumented primary cell (Phase 3).

Cell (identical to the validated version-topology-ab reproducer):
Muse-Glimmer-30B-INT4 @ f5b410ce, vLLM 0.25.1 (752a3a5044), TP=1, eager,
ctx=512, warm-up x1 (max_tokens=8, uninstrumented), measured x8 greedy
(max_tokens=64, temperature=0.0, ignore_eos=True), prompt
[1000 + (i % 20000) for i in range(512)].

Instrumentation: one non-argmax-invariant IDENTITY logits processor
(vLLM 0.25.1 AdapterLogitsProcessor). It runs inside apply_logits_processors,
i.e. AFTER the float32 cast and BEFORE greedy_sample — its input row is the
exact tensor argmax sees. It records sha256, top-64 ids/values, margins and an
independent argmax per step, archives the full fp32 row per generation, and
returns the SAME tensor object (no numerical perturbation; the engine's chain
for this config is otherwise identity — see artifacts/token-selection-path.md).
Warm-up requests (max_tokens=8) get no processor at all.

Usage: FORENSICS_OUT=<dir> forensics_probe.py <engine-id> <out-json>
"""
import gc
import hashlib
import json
import os
import sys

import torch
from vllm.v1.sample.logits_processor import AdapterLogitsProcessor

MODEL = "/workspace/vllm-50603-version-ab/models/muse"
DEPTHS = [512]  # primary forensics cell: ctx=512 only
NGEN = 64
REPEATS = 8
os.environ.setdefault("VLLM_ROCM_CLONE_MMAP_WEIGHTS", "1")
os.environ.pop("VLLM_CLONE_MMAP", None)

OUT = os.environ["FORENSICS_OUT"]
os.makedirs(OUT, exist_ok=True)
_META = open(os.path.join(OUT, "steps.jsonl"), "a", buffering=1)
_CALL = 0            # recorded-call counter (64 per measured generation)
_GEN_BUF = []        # archived rows for the current generation


class GreedyProbeLogitsProcessor(AdapterLogitsProcessor):
    """Identity per-request probe at the exact argmax input."""

    def is_argmax_invariant(self) -> bool:
        # False => grouped in non_argmax_invariant => applied BEFORE
        # greedy_sample. Identity return keeps it numerically inert.
        return False

    def new_req_logits_processor(self, params):
        if params.max_tokens != NGEN:
            return None  # warm-up (max_tokens=8) stays uninstrumented
        return _probe_row


def _probe_row(token_ids, logits):
    global _CALL
    _CALL += 1
    row = logits.detach()
    assert row.dtype == torch.float32, f"unexpected argmax input dtype {row.dtype}"
    cpu = row.contiguous().cpu()                      # exact fp32 copy
    sha = hashlib.sha256(cpu.numpy().tobytes()).hexdigest()
    vals, ids = torch.topk(cpu, 64)
    own_argmax = int(cpu.argmax())
    rec = {
        "call": _CALL,
        "gen": (_CALL - 1) // NGEN,
        "step": (_CALL - 1) % NGEN,
        "prompt_len": len(token_ids) - len(cpu) if False else None,  # unused
        "in_tokens": len(token_ids),
        "sha256_fp32": sha,
        "top_ids": [int(x) for x in ids.tolist()],
        "top_vals": [float(x) for x in vals.tolist()],
        "margin_top1_top2": float(vals[0] - vals[1]),
        "own_cpu_argmax": own_argmax,
        "selected_must_match": own_argmax,  # engine argmax acts on same tensor
    }
    rec.pop("prompt_len")
    _META.write(json.dumps(rec) + "\n")
    _GEN_BUF.append(cpu)
    if _CALL % NGEN == 0:
        g = (_CALL - 1) // NGEN
        torch.save(torch.stack(_GEN_BUF), os.path.join(OUT, f"gen{g}_fp32.pt"))
        _GEN_BUF.clear()
    return logits  # same object -> engine sees an identical tensor


def main():
    engine_id, out_json = sys.argv[1], sys.argv[2]
    from vllm import LLM, SamplingParams
    from vllm.inputs import TokensPrompt

    print(f"[probe] engine={engine_id} out={OUT}", flush=True)
    llm = LLM(model=MODEL, tensor_parallel_size=1,
              max_model_len=8704, max_num_seqs=128,   # exact validated kwargs
              gpu_memory_utilization=0.92, disable_log_stats=True,
              enforce_eager=True,
              limit_mm_per_prompt={"image": 0, "video": 0},
              logits_processors=[GreedyProbeLogitsProcessor])
    sp = SamplingParams(max_tokens=NGEN, temperature=0.0, ignore_eos=True)
    warmup_sp = SamplingParams(max_tokens=8, temperature=0.0, ignore_eos=True)
    print(f"[sampling] measured={sp} warmup={warmup_sp}", flush=True)

    p = TokensPrompt(prompt_token_ids=[1000 + (i % 20000) for i in range(DEPTHS[0])])
    llm.generate([p], warmup_sp)

    seqs, texts = [], []
    for i in range(REPEATS):
        out = llm.generate([p], sp)
        seqs.append(list(out[0].outputs[0].token_ids))
        texts.append(out[0].outputs[0].text)
    distinct = {tuple(s) for s in seqs}
    print(f"[within] muse tp=1 eager=1 PROBED eng={engine_id} ctx=512 -> "
          f"{len(distinct)} distinct of {REPEATS}", flush=True)

    del llm
    gc.collect()
    torch.cuda.empty_cache()

    with open(out_json, "w") as f:
        json.dump({
            "model": "muse", "model_revision":
                "RedHatAI/Muse-Glimmer-30B-INT4@f5b410ce4234fad70eef8be99b4680ee4e30b418",
            "vllm_version": __import__("vllm").__version__,
            "tp": 1, "enforce_eager": True, "engine_id": int(engine_id),
            "context": DEPTHS[0], "warmups": 1,
            "measured_generations": REPEATS,
            "unique_outputs": len(distinct),
            "token_ids": seqs, "decoded_text": texts,
            "instrumentation": "identity non-argmax-invariant logits processor",
        }, f, indent=1)
    _META.close()
    print("=== FORENSICS PROBE DONE ===", flush=True)


if __name__ == "__main__":
    main()
