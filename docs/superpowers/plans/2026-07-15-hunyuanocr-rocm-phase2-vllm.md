# HunyuanOCR-ROCm Phase 2 (vLLM) Plan

**Goal:** Serve HunyuanOCR-1.5 via vLLM (OpenAI-compatible) on gfx1100, run it over OmniDocBench v1.6, and gate it against the Phase-1 transformers BASELINE (±0.3 overall / ±0.5 per-task). vLLM is also the only feasible path for the **full 1651-page set** (transformers-native is ~5.5 tok/s → ~40 h).

**Architecture:** One vLLM server per GPU (`scripts/serve_vllm.sh`, `/opt/venv` vLLM 0.16.1 ROCm build, native `HunYuanVL`). A Python driver (`scripts/run_phase2_vllm.py`) fans pages out to the servers concurrently via the shared `hunyuan_ocr.backends.vllm_client` adapter, which reuses the frozen contract (prompt, sampling, streaming early-stop, `clean_repeated_substrings` + `process_one`) so output is directly comparable to the transformers backend.

**Tech stack:** vLLM 0.16.1 (ROCm gfx1100), transformers 4.57.6 (vLLM's pin — NOT 5.13.0; vLLM has its own native HunyuanVL), `openai` python client, PyYAML. Scorer = OmniDocBench 3.11 venv (shared with Phase 1).

## Global constraints (inherited from the design + Phase-1 findings)

- Frozen decoding contract (Phase 1): `doc_parse` prompt, greedy (temp=0, top_p=1, top_k=-1), `repetition_penalty=1.08`, `max_tokens=32768`, streaming tail-repetition early-stop (`min_repeats=8`), `clean_repeated_substrings` then `process_one`. **Do not change these without re-baselining.**
- Predictions: one UTF-8 `<stem>.md` per page. OmniDocBench v1.6 Overall = `((1-text_edit)*100 + cdm*100 + teds*100)/3`.
- **Gate:** within ±0.3 overall / ±0.5 per-task of the Phase-1 transformers BASELINE (on the same page set).
- **rocmsafe.directory** is configured for `/workspace/HunyuanOCR-ROCm`. Branch `feat/phase1-transformers` (or a new `feat/phase2-vllm` off it).
- Phase-1 gfx1100 adaptations: the transformers backend caps ViT pixels (`GFX1100_VIT_MAX_PIXELS`) + uses sdpa. **vLLM has its own ViT — the cap may NOT be needed; this is Task 1's first question.**

## Status (already done in parallel with the Phase-1 canary)

- Env verified: `/opt/venv` vLLM 0.16.1 (ROCm) **natively registers `HunYuanVLForConditionalGeneration`** (`vllm/model_executor/models/hunyuan_vision.py`), imports under transformers 4.57.6. No vLLM 0.18.1 install needed.
- `openai` client installed in the transformers venv.
- Code scaffolded: `src/hunyuan_ocr/backends/vllm_client.py`, `scripts/serve_vllm.sh`, `scripts/run_phase2_vllm.py`.

## Tasks

### Task 1 — Determinism check (the pivotal question)

Before any eval, answer: **does vLLM's ViT exhibit the >14k-token instability, or is it stable at full resolution?**

- Start one vLLM server (`scripts/serve_vllm.sh`, GPU 0).
- Run the full-resolution sample page (`.../page-d1561665-...png`, ~15k tokens) through the adapter **3×**, compare outputs (and a sub-threshold page for control).
- **If stable + identical + correct** → vLLM runs **uncapped** (no `--max-pixels`); this also implies the instability is transformers/ROCm-ViT-specific (strong evidence for both filed issues).
- **If it also diverges/NaNs** → run with `--max-pixels 3400000` (client-side cap, already wired in the adapter) and note it.

This single check resolves whether vLLM can reach the true 94.74 (uncapped) or only the capped approximation.

### Task 2 — Canary (150 pages) via vLLM

- 3 servers (GPUs 0/1/2, ports 8000/8001/8002).
- `scripts/run_phase2_vllm.py --gt-json OmniDocBench_150.json --pred-dir vllm-canary-150 --ports 8000,8001,8002 [--max-pixels per Task 1]`.
- Score with `scripts/score_predictions.py` (shared scorer).
- **Gate:** compare to the Phase-1 transformers canary-150 BASELINE (±0.3 overall / ±0.5 per-task). Record in `reports/phase2-vllm.md`.

### Task 3 — Full 1651-page set via vLLM

- Same servers; `--gt-json OmniDocBench.json --pred-dir vllm-full-1651`.
- This is the headline full-set run (vLLM is fast enough; transformers isn't).
- Score → the **vLLM full-set number**. Sanity vs upstream 94.74.

### Task 4 — Phase-2 report + cross-backend score table

- `reports/phase2-vllm.md`: transformers (Phase-1, capped, canary) vs vLLM (canary + full) score table + Δ; the determinism finding (Task 1); throughput notes.

## Risks

- vLLM 0.16.1 (not the upstream-validated 0.18.1): native HunyuanVL is present and imports, but serving correctness must be confirmed (issue-#1-style fallback to transformers would fail under transformers 4.57.6). Task 1's smoke catches this.
- Multi-server throughput / OOM at `gpu-memory-utilization 0.9` — tune down if needed.
- The streaming early-stop (`infer_stream`) assumes the server streams; vLLM does.
