# DRAFT — follow-up comment for vllm-project/vllm#50603

NOTES BEFORE POSTING (delete this block): artifact link placeholder `ARTIFACTS_URL` must be replaced with https://github.com/AIwork4me/HunyuanOCR-ROCm/tree/main/validation-50603 once pushed.

---

**Update (2026-08-29): cannot reproduce on a fresh 0.25.1 build on the same GPU model, on the rocm7.14 wheel stack. Three-state data for #53856/#54210 + a mainline finding.**

I re-ran my reproducer scripts verbatim on the reported version (vLLM 0.25.1 rebuilt from source) on a single Radeon Pro W7900D (gfx1100), rocm7.14 wheel stack end to end: torch 2.12.0+rocm7.14.0, triton 3.7.1+git rocm7.14 (same build as originally reported), model pinned to `de8f10ad`, greedy + `enforce_eager` exactly as in the original repro. Both symptoms are absent — while the two log lines the report anchored on appear verbatim (`chunked_prefill_paged_decode.py:419` fallback warning; `jit_monitor.py:129` `kernel_paged_attention_2d` JIT warning), so the fallback path genuinely executes. The baseline suite was re-run once (`baseline-rerun/`); all generations were byte-identical across runs.

| protocol | reported (torch 2.11) | rebuilt 0.25.1 (torch 2.12) | + #53856 | + #53856 + #54210 |
|---|---|---|---|---|
| full-res 15360 tok ×3, no warmup | `ba29b5a1, fd0cc624, fd0cc624` — nondeterministic, garbled | `d2bbe837 ×3` — deterministic, 9/9 GT lines correct | byte-identical to baseline | byte-identical to baseline |
| full-res, warmup + ×3 | `fd0cc624 ×3` — deterministic, garbled | `d2bbe837 ×3` | byte-identical | byte-identical |
| short control 3840 tok ×3 | `f00c87f2, f31888d4, f31888d4` — nondeterministic | `a2fe683f ×3` — deterministic, 6/6 correct | byte-identical | byte-identical |
| decode kernel (from logs) | Triton fallback | Triton fallback (warnings present) | Triton fallback — #53856 alone is inert here, as expected | **CK custom kernel** (fallback warning + decode Triton JIT disappear; outputs identical to the Triton path) |

Takeaways for the open PRs: #54210's gate widening demonstrably routes this `gqa_ratio` 2 model to the ROCm custom kernel end-to-end, deterministic and output-equivalent to the Triton fallback — independent support for its measurements; and #53856 matters only in the ordering sense for this workload (its fix lives in the CK kernel the model cannot reach until the gate is widened), matching #54210's own ordering argument.

Mainline finding, independent of the PRs: on `d1922cb5a7` (2026-08-28, the base of #54210) this workload cannot run at all — #53272/#53615 removed the native Hunyuan V1/VL implementations, and the generic Transformers backend first truncates this model's 4-axis XD-RoPE positions to 3 (`ValueError: Expected 4 multimodal RoPE channels, got position_ids with shape (3, 1, 8192)`), then — after a minimal position-buffer patch, kept in the artifacts — fails reshaping HunYuan's MLA-variant q/k (`shape '[-1, 16, 192]' is invalid for input of size 16777216`). Fresh logs from a clean-worktree rerun are in the artifacts.

Boundaries: the original 4× W7900D + torch 2.11 environment is not excluded — torch 2.11.0 does not initialize on this container (`hipErrorInvalidValue` at engine init, two wheel configurations tried). If the symptoms reproduce on the original box, I will rerun the same three-state harness there.

Artifacts (per-state logs, machine-readable evidence JSON, env captures, applied PR diffs, one-command rerun): ARTIFACTS_URL
