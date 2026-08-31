# Phase 2 — Model availability (fit check results appended below after env build)

## Sources and pinned revisions

| harness key | HF repo | pinned revision (snapshot SHA) | on-disk size | download log |
|---|---|---|---|---|
| muse (PRIMARY) | `RedHatAI/Muse-Glimmer-30B-INT4` | `f5b410ce4234fad70eef8be99b4680ee4e30b418` (main, 2026-08-18) | 21 GiB | logs/model-download.log (MUSE_RC=0) |
| gemma3 (secondary) | `RedHatAI/gemma-3-27b-it-quantized.w4a16` | `2b537554d6c6f6368945e8df4e5fb7bbbb5d56c9` (main, 2025-06-09) | 19 GiB | logs/model-download.log (GEMMA_RC=0) |

On-disk sizes match cadamcat's recorded "21 GB" / "19 GB" exactly — corroborates the repo-identity resolution (their local dir names, quant format compressed-tensors, and sizes all agree). Downloaded 2026-08-30T23:02 via hf-mirror (huggingface.co unreachable from this box). Stable symlinks: `models/muse`, `models/gemma3`.

Muse config (governs attention routing): `MuseGlimmerForConditionalGeneration`, 52 layers, 32 Q heads / 2 KV heads (gqa_ratio 16), head_dim 128, hidden 6656, sliding_window 2048, vocab 202048, compressed-tensors INT4. NOTE: `muse_glimmer` has NO native vLLM implementation in either arm — both 0.23.1.dev1 and 0.25.1 auto-fall back to the Transformers backend implementation (registry `inspect_model_cls` auto-fallback; cadamcat's log shows the same "Resolved architecture" path plus `mm_encoder_attention` ViT lines). That makes the transformers pin part of the measured stack: `transformers==5.15.1` in BOTH arms — the first release line containing `muse_glimmer` (added 2026-08-10, PR #47867; absent in ≤5.14.0). It satisfies both vLLM versions' requirement pins. cadamcat's container (image built 2026-07-15) must likewise carry an upgraded post-5.15 transformers to run this model — recorded as an inference from model-support timing, not a measured fact about their box.

## TP=1 fit check (real engine init)

- [x] Muse @ TP=1, vLLM 0.25.1 — APPENDED AFTER RUN
- [ ] Gemma @ TP=1 — secondary, appended if run
