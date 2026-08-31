# Phase 2 + Phase 3 Status — PASS / PASS

## Phase 2 gate (Muse initializes at TP=1) — PASS

Real engine init on the W7900D in the vLLM 0.25.1 environment (env-0.25.1), log `logs/model-fit-muse-0.25.1.log`:

- Engine up: `Resolved architecture: TransformersMultiModalForCausalLM` (Transformers-backend auto-fallback, as cadamcat's TP=2 log also shows via its `mm_encoder_attention` ViT lines)
- `Found incompatible backend(s) [TURBOQUANT] ... Overriding with ROCM_ATTN` — same attention-backend selection line as cadamcat's muse log
- `Available KV cache memory: 21.56 GiB`, `GPU KV cache size: 434,720 tokens`
- One real greedy generation succeeded (`tokens=[8281, 8281, 8281, 198]`)
- Peak VRAM evidence: engine budget 0.92 × 47.98 GiB ≈ 44.1 GiB; weights ~21 GiB + 21.56 GiB KV + runtime ≈ budget (by construction of the vLLM allocator)
- Fit-script cosmetic bug: a trailing optional `engine_config` access raised AttributeError AFTER the successful generation (fixed in harness/fit_check.py); does not affect the evidence above.

Gemma (secondary) not yet fit-checked — does not gate the primary matrix.

## Phase 3 gate (smoke cell) — PASS

Cell: Muse / vLLM 0.25.1 (git 752a3a504485 = clean v0.25.1 tag) / TP=1 / ctx=512 / warm-up ×1 / measured ×8. Evidence `results/vllm-0.25.1/muse-e0-eng1.json` + `logs/vllm-0.25.1/muse-e0-eng1.log`:

- engine starts, all 8 generations complete (both depths in the same engine, per upstream harness shape)
- token IDs saved (8 × 64 per depth), SHA256 per generation saved, decoded text saved
- unique-output calculation works: ctx=512 → **7 distinct of 8** (first_div vs run0 = [7,7,7,18,18,7,18]); ctx=8192 → **3 distinct of 8** (first_div = [1,1,1,1,1,1,None])
- JSON contains exact engine/config metadata: vllm_version 0.25.1, vllm_git_sha 752a3a504485..., torch 2.12.0+rocm7.14.0, device AMD Radeon Pro W7900D (11,0), tp=1, enforce_eager=False, model_revision pinned

**Headline: the nondeterminism REPRODUCES at TP=1 on vLLM 0.25.1 on the W7900D.** The smoke cell alone (7/8 distinct) already rules out "TP=2-only" and "0.23.1.dev1-only" explanations. Full matrix running.
