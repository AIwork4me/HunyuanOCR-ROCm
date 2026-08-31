# The limit_mm_per_prompt={"image":0,"video":0} decision (2026-08-31)

## Symptom

First vLLM 0.23.1.dev1 matrix attempt (default mm limits, `logs/matrix-vllm-0.23.1.dev1-failed-nommlimit.log`, engine log `logs/vllm-0.23.1.dev1-failed-nommlimit/muse-e0-eng1.log`): every engine died at init with `torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 16.00 GiB` inside `transformers/integrations/sdpa_attention.py:154` → `torch.nn.functional.scaled_dot_product_attention`, reached from the MuseGlimmer ViT encoder forward during vLLM's multimodal profiling (`Encoder cache will be initialized with a budget of 8192 tokens, and profiled with 2 image items of the maximum feature size`). The 16 GiB tensor is the materialized attention matrix of the math SDPA fallback (32 heads × ~16k² × 2 B), with only 8.56 GiB free at that point. The 0.25.1 arm profiles the same model without OOM (its Transformers-backend integration handles the encoder profile differently; no encoder-budget line is logged and init completes in ~51 s).

## Why the ViT path diverges from cadamcat's 0.23.1.dev1

`vllm/platforms/rocm.py` (0.23.1.dev1, ~line 640) routes ViT encoder attention on gfx11 to the Triton flash backend only when the `flash_attn` package (with `flash_attn.flash_attn_triton_amd`) is importable AND `FLASH_ATTENTION_TRITON_AMD_ENABLE=TRUE`. cadamcat's `rocm/vllm` 7.14.0 container ships `flash_attn` and sets the flag — their muse log records `Using Flash Attention (Triton backend) for ViT model on RDNA`. Our ROCm 7.14 **wheel** stack has no `flash_attn` (no wheel available for it on this box's reachable indexes without a source build), so 0.23.1.dev1 resolves ViT attention to Torch SDPA, whose math fallback materializes the matrix. Under cadamcat's image the same profiling forward runs through the Triton flash kernel and fits.

Also noted: cadamcat's 0.23.1.dev1 profiled "1 video items of the maximum feature size" while ours profiles "2 image items" — the resolved multimodal limits differ between their transformers (unknown version, Python 3.14 image, post-5.15 for muse_glimmer support) and our pinned transformers 5.15.1. This only affects the vision-encoder profiling budget, not the text decode path.

## Chosen fix and why

`limit_mm_per_prompt={"image":0,"video":0}` — applied **symmetrically to both arms** via `AB_MM_LIMIT_ZERO=1` in `run_matrix.sh`, gated in the harness (see `harness.diff`).

- The measured path is text-only synthetic token prompts; the ViT encoder never executes at generate time on either arm.
- The engine's decode stack (RDNA3W4A16LinearKernel for the INT4 weights — logged identically on both arms; ROCM_ATTN decode backend via the same TURBOQUANT override line; same max_model_len/max_num_seqs/util; KV cache 434,720 tokens on 0.25.1 vs 432,848 on 0.23.1.dev1) is unchanged.
- The alternative — building AMD flash_attn for gfx1100 and enabling the Triton-flash ViT path — would add a package to both arms that the already-measured 0.25.1 default run did not have, i.e. introduce a new joint variable and require rebuilding both matrices; and a source build of flash_attn on this wheel stack is a separate, failure-prone project (the machine already carries an abandoned flash-attention build attempt in /workspace/builds).
- The first 0.25.1 matrix (default mm limits, 12/12 cells completed) is preserved as `results/vllm-0.25.1-defaultmm/` — it doubles as a control showing the conclusion is insensitive to the mm-limit setting on that arm.

Probe evidence for the fix: `logs/probe-023-mmlimit.log` — 0.23.1.dev1 engine up with the limit, `Available KV cache memory: 21.47 GiB`, one greedy generation completed.
