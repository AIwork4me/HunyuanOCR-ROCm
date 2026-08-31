# vLLM 0.25.1 (mm-zero matrix) — Muse-Glimmer-30B-INT4, TP=1, W7900D

Primary-protocol matrix: `limit_mm_per_prompt={"image":0,"video":0}` (both arms; rationale in `../artifacts/mm-zero-decision.md`), otherwise identical engine kwargs to the harness: max_model_len=8704, max_num_seqs=128, gpu_memory_utilization=0.92, per depth 1 warm-up (max_tokens=8) + 8 measured greedy generations (max_tokens=64, temperature=0.0, ignore_eos=True). vLLM 0.25.1 (git 752a3a504485), torch 2.12.0+rocm7.14.0, transformers 5.15.1. Run window 2026-08-31T00:09–00:41 UTC, 12/12 engines rc=0. One engine per JSON, 3 independent engines per eager state.

| ctx | eager | engine | unique/8 | first divergence vs run0 |
|--:|--:|--:|--:|---|
| 512 | 0 (graphs) | 1 | **8** | [6, 5, 6, 6, 16, 6, 6] |
| 512 | 0 | 2 | **6** | [8, 1, 1, 1, 1, 13, 1] |
| 512 | 0 | 3 | **7** | [35, 24, 27, 40, ∅, 26, 1] |
| 512 | 1 | 1 | **7** | [5, 7, 7, 36, 5, 5, 5] |
| 512 | 1 | 2 | **8** | [5, 5, 6, 5, 19, 5, 5] |
| 512 | 1 | 3 | **8** | [17, 6, 5, 5, 6, 17, 6] |
| 8192 | 0 (graphs) | 1 | 1 | [∅ × 7] |
| 8192 | 0 | 2 | 1 | [∅ × 7] |
| 8192 | 0 | 3 | **3** | [∅, ∅, ∅, ∅, 5, 2, ∅] |
| 8192 | 1 | 1 | **3** | [∅, 1, 1, 1, 1, 1, 5] |
| 8192 | 1 | 2 | **3** | [1, 1, ∅, ∅, ∅, 1, 5] |
| 8192 | 1 | 3 | **2** | [∅, 1, 1, ∅, ∅, ∅, 1] |

**10 of 12 cells vary; all 6 ctx=512 cells vary (6–8 unique of 8).** `--enforce-eager` does not remove it (512 eager: 7, 8, 8 unique). Control matrix at default mm limits (`../vllm-0.25.1-defaultmm/`): 11 of 12 vary — the conclusion is insensitive to the mm-limit setting on this arm.
