@cadamcat thanks for the eager A/B and for spelling out which axis your box can't test — the topology one. I ran that side on the 48 GB W7900D (gfx1100, single GPU), using your harness semantics at TP=1: same prompt construction `[1000 + (i % 20000)]`, same per-depth warm-up (max_tokens=8) then 8 identical greedy generations (max_tokens=64, temperature=0.0, ignore_eos=True), same engine kwargs (max_model_len=8704, max_num_seqs=128, util 0.92), your Muse-Glimmer-30B-INT4 and gemma-3-27b-it-w4a16, and — the part your box couldn't do — tensor_parallel_size=1.

**Result: the nondeterminism reproduces at TP=1 on both vLLM versions, graphs on and eager, and neither the version axis nor the topology axis explains it.**

| model | ctx | vLLM 0.25.1 (git 752a3a5044) graphs / eager, unique-of-8 per 3 independent engines | vLLM 0.23.1.dev1 (git 9ddef71179, the +g commit of your version string) graphs / eager | your TP=2 cells (1 engine) |
|---|--:|---|---|---|
| Muse INT4 | 512 | 8,6,7 / 7,8,8 — varies in 6/6 | 8,7,8 / 8,7,8 — varies in 6/6 | 5 / 5 |
| Muse INT4 | 8192 | 1,1,3 / 3,3,2 | 2,3,1 / 2,1,2 | 1 / 3 |
| gemma3 w4a16 | 512 | 1 / 1 (stable, both) | 1 / 1 (stable, both) | 1 / 1 |
| gemma3 w4a16 | 8192 | 3 / 2 | 3 / 3 | 5 / 2 |

Muse primary ran 3 independent engine processes per cell (12 cells per arm); gemma ran 1 per cell as you published. Every generation's token IDs, SHA256s and decoded text are in the artifacts; all counts above are recomputed from the raw sequences.

Two things in the data worth flagging beyond the binary varies/doesn't:

- The **model fingerprint is identical across topology**: gemma is byte-stable at 512 in all six cells on both machines while varying at 8192, Muse varies at 512 everywhere. Whatever carries the effect travels with the model/regime, not with world_size.
- At Muse/8192, each arm's 48 generations pool into only **5–6 distinct sequences, 4 of them byte-identical across the two vLLM versions** — both versions sample from the same small attractor set, and which attractor wins is what varies. At 512 divergence is near-total (41–46 distinct of 48 per arm). Your "first three generations one way, last five another" pattern at Muse/512/eager did **not** reproduce at TP=1 — our 18 Muse/512 engines show scattered groupings, ten of them fully singleton; reported as absent, not forced.

This rules out, within this experiment's scope: TP=2/distributed execution as a requirement (varies at TP=1, world_size=1, no all-reduce in the path); a 0.23→0.25 fix (clean v0.25.1 tag varies identically); and CUDA graphs as the trigger (eager varies on both versions, matching your eager A/B). It also means our earlier HunyuanOCR no-repro on this same box/stack/0.25.1 was model-specific, not stack-wide — that's a correction to how far our earlier result should be read.

What it leaves open, and where I'd point next: the shared layer below the version difference. Both arms here pin the same torch 2.12.0+rocm7.14.0, triton 3.7.1, transformers 5.15.1 — the pip-freeze diff between the two arms is vllm itself plus an unused `gguf` — and both select the same quant kernel (`RDNA3W4A16LinearKernel`). The single most valuable next probe on my list: on a varying cell (0.25.1, TP=1, ctx=512), log top-2 logits and their gap per generated token — if flips happen only at gap≈0 the mechanism is numeric noise on near-ties and the hunt narrows to the W4A16 dequant-GEMM chain; if flips happen at healthy gaps, something above the logits is implicated.

Boundaries of the comparison, so nobody over-reads it: your exact runtime is torch 2.11.0, which does not initialize on this container (hipErrorInvalidValue at engine init — evidence from our earlier validation), so both arms here run torch 2.12.0+rocm7.14.0 with vLLM version as the only variable; your image's `flash_attn` isn't installable on this wheel stack, so both arms run with `limit_mm_per_prompt={"image":0,"video":0}` (your harness's prompts are text-only; a 0.25.1 control at default mm limits gives the same verdict, 11/12 vary); your kernel patches are not present here (your campaign found the effect symmetric across patch states); Python 3.12 vs your 3.14.

Artifacts (README, environment captures, both pip freezes, harness diff, per-engine JSONs, raw engine logs, analysis scripts, SHA256SUMS): https://github.com/AIwork4me/HunyuanOCR-ROCm/tree/main/validation-50603/version-topology-ab
