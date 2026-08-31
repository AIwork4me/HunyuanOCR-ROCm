# Version A/B comparison — Muse-Glimmer-30B-INT4, TP=1, W7900D (2026-08-31)

Primary matrices (mm-zero, both arms; one engine per row; warm-up ×1 + 8 greedy ×64 tok per cell; temperature=0.0, ignore_eos=True). Machine-readable source: the JSONs beside this file (`vllm-0.25.1/`, `vllm-0.23.1.dev1/`, control `vllm-0.25.1-defaultmm/`); every number below is recomputed from the saved token sequences by `harness/analyze_ab.py` (stored unique_outputs cross-checked by recount).

| model | ctx | TP | vLLM | git | torch | eng | unique/8 | first divergence vs run0 |
|---|--:|--:|---|---|---|--:|--:|---|
| Muse | 512 | 1 | 0.25.1 | 752a3a504 | 2.12.0 | 1 | **8** | 6,5,6,6,16,6,6 |
| Muse | 512 | 1 | 0.25.1 | 752a3a504 | 2.12.0 | 2 | **6** | 8,1,1,1,1,13,1 |
| Muse | 512 | 1 | 0.25.1 | 752a3a504 | 2.12.0 | 3 | **7** | 35,24,27,40,∅,26,1 |
| Muse | 512 | 1 | 0.25.1 | 752a3a504 | 2.12.0 | 1e | **7** | 5,7,7,36,5,5,5 |
| Muse | 512 | 1 | 0.25.1 | 752a3a504 | 2.12.0 | 2e | **8** | 5,5,6,5,19,5,5 |
| Muse | 512 | 1 | 0.25.1 | 752a3a504 | 2.12.0 | 3e | **8** | 17,6,5,5,6,17,6 |
| Muse | 512 | 1 | 0.23.1.dev1 | 9ddef7117 | 2.12.0 | 1 | **8** | 14,14,14,7,14,14,14 |
| Muse | 512 | 1 | 0.23.1.dev1 | 9ddef7117 | 2.12.0 | 2 | **7** | 28,7,9,28,7,7,23 |
| Muse | 512 | 1 | 0.23.1.dev1 | 9ddef7117 | 2.12.0 | 3 | **8** | 9,7,7,9,44,7,9 |
| Muse | 512 | 1 | 0.23.1.dev1 | 9ddef7117 | 2.12.0 | 1e | **8** | 6,5,21,33,5,5,19 |
| Muse | 512 | 1 | 0.23.1.dev1 | 9ddef7117 | 2.12.0 | 2e | **7** | 15,5,5,5,7,15,5 |
| Muse | 512 | 1 | 0.23.1.dev1 | 9ddef7117 | 2.12.0 | 3e | **8** | 5,5,25,5,5,18,18 |
| Muse | 8192 | 1 | 0.25.1 | 752a3a504 | 2.12.0 | 1 | 1 | ∅×7 |
| Muse | 8192 | 1 | 0.25.1 | 752a3a504 | 2.12.0 | 2 | 1 | ∅×7 |
| Muse | 8192 | 1 | 0.25.1 | 752a3a504 | 2.12.0 | 3 | **3** | ∅,∅,∅,∅,5,2,∅ |
| Muse | 8192 | 1 | 0.25.1 | 752a3a504 | 2.12.0 | 1e | **3** | ∅,1,1,1,1,1,5 |
| Muse | 8192 | 1 | 0.25.1 | 752a3a504 | 2.12.0 | 2e | **3** | 1,1,∅,∅,∅,1,5 |
| Muse | 8192 | 1 | 0.25.1 | 752a3a504 | 2.12.0 | 3e | **2** | ∅,1,1,∅,∅,∅,1 |
| Muse | 8192 | 1 | 0.23.1.dev1 | 9ddef7117 | 2.12.0 | 1 | **2** | 2,2,2,2,∅,2,2 |
| Muse | 8192 | 1 | 0.23.1.dev1 | 9ddef7117 | 2.12.0 | 2 | **3** | ∅,5,5,∅,∅,5,2 |
| Muse | 8192 | 1 | 0.23.1.dev1 | 9ddef7117 | 2.12.0 | 3 | 1 | ∅×7 |
| Muse | 8192 | 1 | 0.23.1.dev1 | 9ddef7117 | 2.12.0 | 1e | **2** | ∅,∅,∅,∅,∅,1,∅ |
| Muse | 8192 | 1 | 0.23.1.dev1 | 9ddef7117 | 2.12.0 | 2e | 1 | ∅×7 |
| Muse | 8192 | 1 | 0.23.1.dev1 | 9ddef7117 | 2.12.0 | 3e | **2** | 1,1,1,∅,1,1,1 |

`e` suffix = `--enforce-eager` (eager=1). ∅ = identical to run 0. Control matrix (0.25.1, default mm limits): 512 → [7,5,7] graphs / [6,6,5] eager; 8192 → [3,1,3] / [3,3,2] — same conclusion, 11 of 12 cells vary.

## Cell-level rollup

| model | ctx | vLLM | graphs unique/8 (3 eng) | eager unique/8 (3 eng) | stable cells |
|---|--:|---|---|---|---|
| Muse | 512 | 0.25.1 | 8, 6, 7 | 7, 8, 8 | 0 of 6 |
| Muse | 512 | 0.23.1.dev1 | 8, 7, 8 | 8, 7, 8 | 0 of 6 |
| Muse | 8192 | 0.25.1 | 1, 1, 3 | 3, 3, 2 | 3 of 6 |
| Muse | 8192 | 0.23.1.dev1 | 2, 3, 1 | 2, 1, 2 | 1 of 6 |

vs cadamcat's TP=2 cells on 0.23.1.dev1 (one engine each): muse 512 → 5 (graphs) / 5 (eager); muse 8192 → 1 (graphs) / 3 (eager). Our TP=1 engine results land inside the same behavioral regime; at 512 our divergence is typically stronger (6–8 unique vs their 5).

## Structure of the variation (from raw token IDs)

1. **ctx=512: near-total divergence.** Pooling all 48 generations per arm: 41 distinct sequences on 0.25.1, 46 on 0.23.1.dev1 (30 on the 0.25.1 default-mm control). Divergence between concurrent runs appears at token indices 1–44, i.e. the sequences branch mid-generation with a shared prefix.
2. **ctx=8192: a small shared attractor set.** Pooled per arm: only 5–6 distinct sequences out of 48 on each arm, and **4 of those sequences are byte-identical across the 0.25.1 and 0.23.1.dev1 arms**. Both versions sample from essentially the same handful of attractor completions; which one a generation lands on is what varies. Divergence, when it happens, is almost always at index 1 or 2 (the second/third token), or not at all.
3. **cadamcat's "settles after three generations" pattern does not reproduce at TP=1.** Their muse/512/eager cell split 3-then-5 (first three generations one sequence, last five another, identical from token 1). In our 18 muse/512 engines (both versions + control), run-group structure is: 10 engines fully single-run groups (8 unique), the rest scattered pairings (e.g. [0,3,4,5] on the default-mm control, [2,4,6,7] on one eager engine) — no engine shows the clean temporal 3-vs-5 split their engine showed. We do not force the analogy; it is not present in this data.
4. **`--enforce-eager` changes nothing systematic** on either version: 512 stays at 6–8 unique; 8192 stays mixed (stable in some engines, 2–3 unique in others).
5. **Engine-to-engine spread is real on both versions** (e.g. 0.25.1/8192/graphs: 1, 1, 3) — exactly the spread cadamcat documented across their four TP=2 engines.

## Version axis verdict (facts)

- 0.23.1.dev1 TP=1 = nondeterministic (11 of 12 primary cells vary; the one stable cell is 8192/eager/eng2).
- 0.25.1 TP=1 = nondeterministic (10 of 12 primary cells vary; 11 of 12 in the default-mm control).
- No cell pattern distinguishes the versions: same depths vary, same eager-insensitivity, overlapping attractor sets at 8192, comparable engine spread.

## Secondary model — gemma-3-27b-it-w4a16 (1 engine per cell, mirroring cadamcat's published cells)

| gemma3 | ctx | TP | vLLM | eager | unique/8 | first div vs run0 | cadamcat TP=2 / 0.23.1.dev1 same cell |
|---|--:|--:|---|--:|--:|---|---|
| gemma3 | 512 | 1 | 0.25.1 | 0 | 1 | ∅×7 | 1 (stable) |
| gemma3 | 512 | 1 | 0.25.1 | 1 | 1 | ∅×7 | 1 (stable) |
| gemma3 | 512 | 1 | 0.23.1.dev1 | 0 | 1 | ∅×7 | 1 (stable) |
| gemma3 | 512 | 1 | 0.23.1.dev1 | 1 | 1 | ∅×7 | 1 (stable) |
| gemma3 | 8192 | 1 | 0.25.1 | 0 | **3** | ∅,∅,∅,∅,∅,4,4 | 5 (varies) |
| gemma3 | 8192 | 1 | 0.25.1 | 1 | **2** | 4,∅,∅,∅,4,∅,∅ | 2 (varies) |
| gemma3 | 8192 | 1 | 0.23.1.dev1 | 0 | **3** | ∅,4,4,∅,31,4,∅ | 5 (varies) |
| gemma3 | 8192 | 1 | 0.23.1.dev1 | 1 | **3** | 4,∅,∅,∅,4,37,∅ | 2 (varies) |

The model-dependent fingerprint reproduces exactly across version AND topology: gemma-3 is byte-stable at ctx=512 in all four of our TP=1 cells (and both of cadamcat's TP=2 cells) while varying at ctx=8192 in all six cells on both machines/versions; Muse varies at ctx=512 everywhere. Divergence positions for gemma/8192 sit at index 4–37. Secondary arm runs the same mm-zero protocol; one engine per cell (as published by cadamcat for this model).
