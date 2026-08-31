# vLLM 0.25.1 — Muse-Glimmer-30B-INT4, TP=1, W7900D — Phase 4 results

Environment `env-0.25.1`: vLLM **0.25.1** (git `752a3a504485790a2e8491cacbb35c137339ad34`, clean `v0.25.1` tag), torch 2.12.0+rocm7.14.0, triton 3.7.1+git0263a6a6.rocm7.14.0, transformers 5.15.1, Python 3.12.3. Engine kwargs per cadamcat harness with TP=1: max_model_len=8704, max_num_seqs=128, gpu_memory_utilization=0.92, per depth 1 warm-up (max_tokens=8) + 8 measured greedy generations (max_tokens=64, temperature=0.0, ignore_eos=True). One engine per JSON; 3 independent engines per eager state. Run window 2026-08-30T23:19–23:53 UTC. Every engine rc=0.

| ctx | eager | engine | unique/8 | first divergence vs run0 (token index) |
|--:|--:|--:|--:|---|
| 512 | 0 (graphs) | 1 | **7** | [7, 7, 7, 18, 18, 7, 18] |
| 512 | 0 | 2 | **5** | [7, 7, ∅, ∅, ∅, 18, 9] |
| 512 | 0 | 3 | **7** | [5, 18, 7, 27, 7, 7, 7] |
| 512 | 1 | 1 | **6** | [7, ∅, ∅, 15, 13, 5, 5] |
| 512 | 1 | 2 | **6** | [23, 9, 5, 5, 7, 23, 23] |
| 512 | 1 | 3 | **5** | [6, 5, 5, 5, 5, 5, 5] |
| 8192 | 0 (graphs) | 1 | **3** | [1, 1, 1, 1, 1, 1, ∅] |
| 8192 | 0 | 2 | 1 | [∅ × 7] |
| 8192 | 0 | 3 | **3** | [1, 1, 1, 1, 1, 1, 1] |
| 8192 | 1 | 1 | **3** | [1, ∅, 1, 1, 1, 1, ∅] |
| 8192 | 1 | 2 | **3** | [1, ∅, 1, 1, 1, 5, ∅] |
| 8192 | 1 | 3 | **2** | [∅, 1, ∅, ∅, 1, ∅, ∅] |

∅ = identical to run 0.

**11 of 12 cells vary.** The single stable cell is ctx=8192/graphs/engine-2 — engine-to-engine spread of exactly the kind cadamcat documented ("2 to 3 of 8 at one depth across four independent engines"; their muse/8192 graphs-on cell was 1 of 8 stable on their box too).

Facts before interpretation:

- At ctx=512 the variation is mid-sequence (divergence indices 5–27), sequences typically share a common prefix; at ctx=8192 divergence is at index 1 or absent — two or three attractors from the second token onward.
- `--enforce-eager` does not remove the variation on this stack either (5–6 of 8 at 512, 2–3 of 8 at 8192), consistent with cadamcat's eager result on 0.23.1.dev1/TP=2.
- Full token sequences, SHA256s, decoded text, and engine metadata for all 48 generations per eager state: `muse-e<0|1>-eng<1|2|3>.json` beside this file; raw engine logs in `logs/vllm-0.25.1/`.
