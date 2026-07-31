# Benchmark Context — External & Experimental References

This file holds benchmark numbers that are **NOT** results of this project — they
are external references (upstream/official/paper) or experimental subset
measurements. They must never appear in `model_card_v2.json` `results[]` or the
generated README results block (rules: official/paper results are not our results;
canary is experimental). The canonical, project-measured results live in
`model_card_v2.json`.

## External references (different engine / not same-page-set)

| source | Overall | engine | page set | note |
|---|---|---|---|---|
| Official HunyuanOCR table | 94.10 | **TensorRT** | unlabeled OmniDocBench version | [Tencent-Hunyuan/HunyuanOCR](https://github.com/Tencent-Hunyuan/HunyuanOCR). NOT a same-engine, same-page-set comparison. |
| Upstream-reported (canary) | 94.74 | upstream | canary 148 | reported in `reports/canary-baseline.md`; upstream figure. |

These are **not** ROCm results and are **not** comparable to this repo's numbers.

## Experimental canary-subset measurements (148 pages, 2026-07-16)

Historical/experimental. `reports/canary-baseline.md` labels the canary
"Historical — retained as experimental evidence; README.md is the single source of
current status." They are **not** formal `result_records` (the central `result_id`
tuple does not encode page-set; canary and full collide, and these lack committed
platform artifacts). Recorded here for traceability only.

| backend | Overall | text EditDist↓ | formula CDM↑ | table TEDS↑ | reading-order↓ | resolution |
|---|---|---|---|---|---|---|
| vLLM 0.16.1 (Flash-Attn ViT) | 94.81 | 0.0514 | 0.9648 | 0.9308 | 0.1135 | capped 3.4M |
| transformers 5.13.0 (SDPA ViT) | 94.11 | 0.0437 | 0.9425 | 0.9246 | 0.1184 | capped 3.4M |
| llama.cpp (C++ GGML, BF16 GGUF) | 93.33 | 0.0512 | 0.9083 | 0.9429 | 0.1270 | uncapped |

- Canary artifacts were machine-local (`/root/hunyuanocr-results/...`); not committed.
- An earlier partial 143-page transformers score of 93.16 was an artifact (missing
  pages scored ~0); the complete 148/148 score is 94.11.

## Formal project results (canonical)

See `model_card_v2.json` and the generated README block. Primary:
**vLLM full 1651 = 93.64** (`evidence-complete`). Documented-only:
**llama.cpp full 1651 = 92.09** (`submitted`).
