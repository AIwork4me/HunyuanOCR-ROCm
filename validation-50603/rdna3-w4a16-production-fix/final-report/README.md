# RDNA3 W4A16 production fix — upstream candidate

## Final design

| shape | path | epilogue |
|---|---|---|
| M=1..64 (any N) | scalar kernel (unchanged compute) | FP32 split partials → fixed-order reduce → single bf16/half rounding (tile 64 rows; scratch = Z·64·N·4B) |
| M≥65 & N≥64 | WMMA (unchanged compute, incl. 128×64 V7/V8) | FP32 split partials → fixed-order reduce → single rounding (tile 512 rows; scratch = k_split·512·N·4B ≤ ~144 MiB) |
| M≥65 & N<64 (rare) | routed to scalar deterministic path | as above |

Threshold M=65 comes from the measured scalar-vs-WMMA crossover (K=6656/N=4096: M=64 scalar 276µs vs WMMA-no-split 355µs; M=128 WMMA-det 172µs vs scalar 479µs). Deterministic **by default**; `VLLM_RDNA3_W4A16_LEGACY_ATOMIC=1` is a benchmarking opt-out only. No new torch ops, no python changes — dispatch is entirely in the two kernel host files.

Key engineering notes baked into the code comments:
- low-precision unordered CAS accumulation is disallowed because float addition is non-associative and every intermediate add rounds;
- the scratch buffer is thread-local cached (bounded, resized on demand) and the cache is bypassed during CUDA graph capture (captured kernels keep baked pointers; a later resize would free them — this was a real fault caught and fixed in Phase 13).

## Why not the alternatives (all measured on the W7900D, real q_proj K=6656 N=4096)

| design | verdict | evidence |
|---|---|---|
| single-writer (grid.z=1, block loops all K) | rejected | deterministic but 567.6 µs at M=1 = 13.1× (4 of 96 CUs busy) |
| reduced split-K (Z=2..8) | rejected | still low-precision unordered accumulation; Z=2 was stable in one probe by luck, Z≥4 raced (100/100 distinct); only guaranteed-deterministic constructions accepted |
| WMMA forced K_SPLIT=1 | insufficient | deterministic, but +187% at M=16, +56% at M=128 (fine at M=512: +13%) |
| tree reduction | unnecessary | the in-thread sequential fixed-order reduce reads Z·N floats once; at Z≤26 latency is dominated by the single pass (measured det overhead at M=1 is +7.8% total incl. launch+alloc) |

## Deliverables

- `patches/upstream-production-fix.patch` — 2 kernel files + 4 regression tests, 808 lines, applies cleanly to v0.25.1 and to current main (`dafbef15a`, files byte-identical between the two)
- `final-report/upstream-pr-production-draft.md`, `final-report/github-comment.md`, `final-report/self-review.md`
- evidence: `wmma/audit.jsonl` (WMMA determinism audit), `scalar/`, `microbench/` (Z-sweep, single-writer), `correctness/` (accuracy+latency matrices both modes), `performance/production-performance.md`, `e2e/` (Muse det ×3 engines eager + graphs + legacy control, gemma, main-port), `logs/`

## Phase 14 — TP=2

Not executable on this machine: a single W7900D is installed (1 of the host's 8 GPUs); vLLM TP=2 requires two visible devices. The kernel change is topology-agnostic (per-rank local GEMM epilogue); the prior version/topology validation already showed the defect and its absence are both TP-independent. Follow-up on multi-GPU hardware: one TP=2 Muse run to confirm routing and determinism.

## Phase 5 — tree reduction closure

Closed by measurement rather than implementation: the sequential fixed-order in-thread reduce already achieves +7.8% at M=1 (including two-pass launch and scratch traffic), and a tree cannot reduce the data read below one pass over Z·N floats. No tree variant was built; flat scratch + sequential reduce is the shipped design.
