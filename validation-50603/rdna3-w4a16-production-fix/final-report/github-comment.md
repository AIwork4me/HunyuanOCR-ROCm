@cadamcat the root cause is now fixed in a production-shaped patch, validated end to end on the W7900D and ported to current vLLM main.

**Root cause (proven earlier, restated in one line):** both RDNA3 W4A16 paths (scalar and WMMA) split K across concurrent blocks that each cast their FP32 partial to bf16/fp16 and CAS-atomically accumulate it directly into the low-precision output — float addition is non-associative and every intermediate add rounds, so the result depends on block completion order. Repeated fixed-input calls differ almost every call once the writer count exceeds the empirical onset observed in the probe (between 2 and 4 concurrent writers).

**Fix design (deterministic by default, no env var needed):** split-K blocks now write their FP32 partials to a scratch buffer (no atomics, no intermediate low-precision rounding); a second pass reduces the z-slices in fixed order and rounds to the output dtype exactly once. Scalar path (M ≤ 64) and WMMA path (M ≥ 65, N ≥ 64 — the measured crossover) share this epilogue; the compute kernels are unchanged. Scratch is row-tiled (64 rows scalar, 512 rows WMMA) so memory stays bounded (≤ ~144 MiB worst case) independent of M, cached per thread, and bypassed during CUDA graph capture. `VLLM_RDNA3_W4A16_LEGACY_ATOMIC=1` remains as a benchmarking opt-out.

**End-to-end:** Muse greedy ×8, TP=1, ctx 512 and 8192 — 1 unique output in 3/3 eager engines, 1/1 graph-enabled engine, and 1/1 on current main (`dafbef15a`). gemma-3-27b-it-w4a16 likewise 1/8. A legacy-arm control engine in the same build reproduces the nondeterminism (8/8) and kernel routing is logged in every run.

**Cost:** decode M=1 +7.8% per GEMM (43.1→46.4 µs on the real q_proj); M=8 +5.4%, M=16 +1.9%, M=64 +2.3%, M=128 −1.2% (faster), M=512 +1.1%. Full-engine wall on the eager both-depth harness: +14.8% vs legacy atomic, but −30% vs the Triton fallback and −23% vs the earlier root-cause prototype. **Accuracy improves, not degrades:** max error vs an fp32 dequant reference drops at every M (e.g. M=1: 0.0061 vs 0.028; M=512: 0.017 vs 0.038), because split partials now accumulate in FP32 with a single final rounding.

Regression tests (K=1024 ⇒ Z=4 scalar; K=6656 ⇒ K_SPLIT=4 WMMA at M=16 and M=128): four tests, all fail on unpatched vLLM and pass on the patch — verified on both 0.25.1-era and current-main builds.

Artifacts and the patch: https://github.com/AIwork4me/HunyuanOCR-ROCm/tree/validation-50603-rdna3-production-fix/validation-50603/rdna3-w4a16-production-fix — PR-ready against current main; happy to open it.
