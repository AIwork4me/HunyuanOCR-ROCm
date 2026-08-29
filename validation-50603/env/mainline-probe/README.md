# mainline-probe — why the #50603 workload cannot run on current vLLM main

Dedicated rerun on a clean git worktree of upstream main at `d1922cb5a7` (the base of PR #54210, 2026-08-28), built in an isolated venv (torch 2.12.0+rocm7.14.0, transformers 5.13.0, gfx1100), running `scripts/repro_determinism.py` verbatim.

- `wall1-position-ids.log` — upstream code, unpatched. Engine init fails in the HF forward with
  `ValueError: Expected 4 multimodal RoPE channels, got position_ids with shape (3, 1, 8192)`.
  HunyuanOCR is a 4-axis XD-RoPE model; transformers normalizes it into a 4-section m-RoPE, but
  vLLM's m-RoPE position state (`vllm/v1/worker/gpu/mm/rope.py`, `get_rope_state`) hardcodes 3 dims,
  so `init_prefill_positions` drops the 4th channel.
- `wall2-attention-reshape.log` — after applying `../attempt-mainline-enabling-patch.diff` (sizes the
  m-RoPE buffer from the model's axis count). The engine now gets past positions and fails in vLLM's
  unified attention layer (`vllm/model_executor/layers/attention/attention.py:524`) reached through
  the Transformers modeling backend:
  `RuntimeError: shape '[-1, 16, 192]' is invalid for input of size 16777216`
  (HunYuan's MLA-variant q/k: 192 = 128 nope + 64 rope). Fixing this is genuine model-support work,
  out of scope for the validation — the probe exists to document that the #50603 workload is
  currently unservable on main, before any paged-attention code is ever reached.

Root cause context: the native Hunyuan V1/VL implementations were removed upstream in #53272
(migration: #53615) after this issue was filed, so the model fell back to the generic Transformers
backend, which does not yet cover this model's position scheme or attention interface.

Note: both failures are host-side Python/C++ shape- and type-level errors that occur before
any custom device code executes, so they are independent of which HIP compiler built the
extension.
