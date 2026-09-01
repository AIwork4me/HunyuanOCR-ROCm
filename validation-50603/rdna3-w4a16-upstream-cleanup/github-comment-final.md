# GitHub issue comment draft — vLLM #50603 (final)

Status: draft, not posted.

---

Quick final update from the gfx1100 (W7900D) investigation.

Root cause (details in earlier comments): the RDNA3 W4A16 split-K epilogue atomically accumulates low-precision partials, so the result depends on CAS completion order. The empirical onset in the fixed-input probe occurred between 2 and 4 concurrent writers. For the captured workload, ascending, descending, and fixed-shuffled FP32 reductions were bit-identical — the order-sensitive part was the narrowing before accumulation, not the FP32 summation itself.

I've now cleaned the fix into an upstream-focused patch that keeps the existing compute/routing behavior and only replaces the nondeterministic split-K epilogue: split blocks write FP32 partials, a fixed ascending-z FP32 reduction runs, and the output dtype cast happens exactly once (k_split == 1 keeps the original direct store). Scratch is per-call from the PyTorch caching allocator, row-tiled so the bound is independent of M — no persistent state, CUDA-graph capture safe.

Validation on real hardware: 1 distinct output per 100 repeats at every tested shape (bf16 + fp16, real Muse q_proj weights, M = 1–512); per-call interception across a full generation (16,640 GEMM calls) shows zero same-input-different-output events; regression tests fail-before / pass-after; greedy E2E is bit-identical across repeats for gemma-3 (ctx 512/8192) and Muse (graphs and eager ctx512). Accuracy vs an fp32 dequant reference improves (max abs error 0.028 → 0.0061 at M=1). Performance: M=1 +4.6%, large-M within ±5%, and the WMMA range gets faster (M=16 −17%, M=64 −6%) since the CAS epilogue it sheds was the dominant cost there.

One caveat for completeness: at Muse eager ctx=8192 the two-family residual I reported earlier is still reproducible, but it is not this epilogue — call-level hashing shows every W4A16 GEMM bit-deterministic while the first divergence enters through the ROCm custom paged-attention decode path on the NoPE (full-attention) layers, whose output turns out to depend on stale slots of the KV cache's last block (1 element, ~1.5 ULP; swap the last block, the output changes). With `attention_backend=TRITON_ATTN` the same build is fully deterministic at both depths. I'll file that separately with the reproduction probes.

I've now cleaned the fix into an upstream-focused patch that keeps the existing compute/routing behavior and only replaces the nondeterministic split-K epilogue.
