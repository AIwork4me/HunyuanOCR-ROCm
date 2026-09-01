Final status from my gfx1100/W7900D investigation — the reproducible nondeterminism split into two independent mechanisms:

1. **RDNA3 W4A16 split-K reduction** — the scalar and WMMA paths narrowed FP32 partials to bf16/fp16 before CAS-atomic accumulation, making results completion-order dependent. The upstream-focused fix keeps the existing compute/dispatch behavior and replaces only that epilogue with deterministic FP32 partial storage + fixed-order reduction + one final cast.

   PR: #54706

   After rebasing to current `main` (`40b2f62061575905aaac8bc360eaea62a4baeb67`), the determinism tests are 8/8 PASS; fixed-input scalar and WMMA probes are 1 distinct / 100 calls, and Muse eager ctx512 is 1/8.

2. **Residual Muse eager/8192 stale V-tail sensitivity** — after the W4A16 fix, the remaining 2/8 split entered the ROCm custom paged-attention path. An A/B showed that #53856 eliminates both the model-level split (2/8 → 1/8) and sensitivity to finite stale V-cache slots beyond `seq_len` (1/780 → 0/780).

   Validation: https://github.com/vllm-project/vllm/pull/53856#issuecomment-5488571510

So I would treat these as two independent gfx1100 mechanisms with separate fixes: the new W4A16 PR for quantized GEMM, and #53856 for invalid final-block V consumption.

The original HunyuanOCR long-sequence garbling observation remains separate: I could not reproduce it on the rebuilt torch 2.12 / ROCm 7.14 stack.

Full W4A16 evidence:
https://github.com/AIwork4me/HunyuanOCR-ROCm/tree/validation-50603-rdna3-upstream-cleanup/validation-50603/rdna3-w4a16-upstream-cleanup
