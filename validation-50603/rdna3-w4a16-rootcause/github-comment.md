@cadamcat I followed the forward-pass result to the kernel and now have the full causal chain, closed end to end on the W7900D: **the nondeterminism is the split-K epilogue of `gptq_gemm_rdna3` — each split block casts its FP32 partial to bf16 and CAS-atomically accumulates it straight into the bf16 output, so the low-precision reduction order varies — and replacing that epilogue with FP32 scratch + fixed-order reduction makes both the isolated kernel and the full Muse greedy reproducer bit-reproducible.**

| path | Muse/512 greedy ×8 (3 engines) | isolated M=1, real q_proj, 200 calls |
|---|---|---|
| current RDNA3 W4A16 (atomic) | 6/8, 8/8, 8/8 unique | 200/200 distinct (3 procs; first-call sha differs per process) |
| `VLLM_DISABLED_KERNELS=RDNA3W4A16LinearKernel` → Triton (routing verified from logs) | **1/8, 1/8, 1/8** | — |
| deterministic split-K prototype | **1/8, 1/8, 1/8** (also 1/8 ×3 at ctx=8192) | **1/200 ×3 processes, identical sha** |

The mechanism, measured at each step (K=6656 → `grid.z = 26` splits):

- The per-split **FP32 partials are deterministic** (diagnostic op: 50 calls → 1 distinct scratch tensor).
- Ascending / descending / shuffled **FP32 reduction orders are bitwise identical** — so this is not an FP32-associativity story; the variance enters exactly at the **cast-to-bf16-then-atomically-add** chain, which rounds at every one of 26 unordered adds. The 30-call atomic spread (±0.035) matches the atomic-vs-deterministic gap scale.
- On the **unmodified production op**, a K-sweep at N=4096 flips determinism on between Z=2 and Z=4: Z=1→1/100 distinct, Z=2→1/100, Z=4/Z=8/Z=26→100/100. Single-writer is stable; concurrent multi-writer is not.
- The prototype (partials to FP32 scratch, one fixed-order reduce, single final rounding) is also **more accurate**: max error vs an fp32 dequant reference 0.0061 vs the atomic path's 0.0261 (cosine 0.999999 vs 0.999981).

Cost: +6.6% at the M=1 decode shape (43.0→45.9 µs median per q_proj GEMM). One serving-path surprise mattered: generation re-runs prefill-tail forwards at M>8 (logged with a gated M-logger), so the first decode-only prototype still varied 6/8 — routing **all** M through a 64-row-tiled deterministic path (scratch ≤ ~118 MiB) is what takes the engine to 1/8. End-to-end engine wall time on the eager both-depths harness: atomic 115 s, prototype 172 s, Triton fallback 188 s — i.e. the deterministic prototype is ~9% faster than the only currently-deterministic alternative. A production design would keep the (separately validated) WMMA path for large M; the M≥16 epilogue's determinism is still unassessed.

Regression test (K=1024 ⇒ Z=4, above the measured onset), passes on the deterministic path and xfails on the legacy epilogue. Patches (diagnostic / prototype / clean upstream candidate) + all raw evidence: https://github.com/AIwork4me/HunyuanOCR-ROCm/tree/validation-50603-rdna3-w4a16-rootcause/validation-50603/rdna3-w4a16-rootcause — happy to turn the candidate into a PR; option-2 (single-split at M=1, zero scratch) is probably worth benchmarking alongside it since it avoids the extra pass entirely.
