# Draft update for issue #50603 (NOT posted)

Status update separating the two independent findings from the gfx1100 (W7900D) investigation:

1. RDNA3 W4A16 split-K low-precision atomic reduction — fixed by the W4A16 patch. The scalar and WMMA split-K epilogues atomically accumulated narrowed bf16/fp16 partials in execution-dependent order; the cleaned patch replaces both epilogues with FP32 per-split partials, a fixed ascending-z FP32 reduction, and a single final cast, keeping the existing compute kernels and dispatch. Per-call interception over a full generation (16,640 GEMM calls) shows zero same-input/different-output events. A submission branch is prepared (`fix/rdna3-w4a16-determinism`); this comment does not announce an open PR.

2. Residual Muse eager/8192 stale V-tail sensitivity — covered by #53856. After the W4A16 fix, a 2/8 greedy split at ctx8192 remained whose first divergence entered through the ROCm custom paged-attention path with bit-identical q and valid KV; the output depended on V-cache slots beyond seq_len in the final referenced block (finite stale values, ~1.5 ULP effect). An A/B on the same build shows #53856 ("Mask paged attention V cache padding") eliminates it: 2/8 → 1/8, and the stale-tail mutation no longer changes the attention output (0/780 vs 1/780 on baseline).

For the captured workload, ascending, descending, and fixed-shuffled FP32 reductions were bit-identical. The empirical onset in the fixed-input probe occurred between 2 and 4 concurrent writers.

Note on history: the original HunyuanOCR long-sequence garbling report was not reproducible on the rebuilt torch 2.12 stack; the two mechanisms above are what the gfx1100 investigation could actually isolate and fix.

Evidence: W4A16 cleanup — https://github.com/AIwork4me/HunyuanOCR-ROCm/tree/validation-50603-rdna3-upstream-cleanup/validation-50603/rdna3-w4a16-upstream-cleanup ; #53856 A/B — …/53856-ab
