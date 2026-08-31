# Phase 1 audit result + Phase 2 acceptance targets

## Phase 1 — WMMA determinism audit (`wmma/audit.jsonl`, 2 processes, 100 calls, real q_proj K=6656 N=4096 bf16)

| M | variant tiles | computed K_SPLIT | distinct/100 (A, B) | max abs diff |
|---:|---|---:|---|---:|
| 16 | 64×16 (v3) | 4 | **100, 100** | 0.03125 |
| 32 | 64×16 (v3) | 4 | 3, 2 | 0.015625 |
| 64 | 64×16 (v3) | 4 | **100, 100** | 0.03125 |
| 128 | 64×32 (v4) | 4 | 2, 2 | 0.0078125 |
| 512 | 64×64 (v5) | 4 | 1, 1 (same sha across processes) | 0.0 |
| 128 ctrl, K=256 | 64×32 | **1** | 1, 1 (same sha across processes) | 0.0 |

Required question answered: WMMA is nondeterministic only when multiple split-K writers target the same output (K_SPLIT=1 control is bit-stable cross-process). K_SPLIT>1 is nondeterministic in general (M=16/64: every call differs); the M=512 stability observed in this probe is a scheduling coincidence, not a property to rely on — the fix must remove unordered low-precision multi-writer accumulation everywhere, in both scalar and WMMA paths.

Corroboration: this also closes the loop on the root-cause experiment's serving-path discovery — the decode-only deterministic prototype stayed nondeterministic because the serving path issues WMMA calls at M=16/128 (chunked-prefill tails), which this audit shows are exactly the nondeterministic WMMA shapes.

## Phase 2 — acceptance targets

| dimension | target |
|---|---|
| M=1 decode determinism | bit-reproducible (in-process, cross-process) |
| M=1 correctness | ≥ current atomic path vs fp32 dequant reference (root-cause prototype was 4× better) |
| M=1 latency | ≤ +10% vs atomic (stretch ≤ +5%; root-cause scratch path measured +6.6%) |
| M=2,4,8 | deterministic |
| Large M / prefill | preserve WMMA; ≤ 10–15% vs current WMMA per shape (no scalar-4× regression) |
| Scratch | bounded, formula explicit, never multi-GB; no pathological allocation at any M |
| Default behavior | deterministic by default — no env var required for correctness |
| Tests | fail-before/pass-after, gfx11-gated, CI-sized |
| Accuracy matrix | final ≥ atomic at M = 1, 8, 16, 64, 512; no NaN/Inf |
