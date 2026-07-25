# HunyuanOCR-ROCm — Design Spec

**Date:** 2026-07-15
**Status:** Approved (brainstorm) → pending implementation plan
**Target repo:** `AIwork4me/HunyuanOCR-ROCm` — **standalone, self-contained project first**; integration into OmniDocBench-AMD is deferred to Phase 4 (post-eval)
**Local path:** `/workspace/HunyuanOCR-ROCm` (all project files — spec, code, eval, results, reports — live here)
**Platform:** `linux-rocm` / gfx1100 (RDNA3) only
**Program ref:** future sub-project under the AMD Doc-Parsing 专区 (see `2026-07-12-amd-doc-parsing-platform-foundation-design.md`) — joined only at Phase 4

---

## 1. Goal

Build **`AIwork4me/HunyuanOCR-ROCm`**: an open, eval-backed project that runs Tencent's **HunyuanOCR-1.5** (~1B VLM) on AMD gfx1100, precision-aligned with the original on **OmniDocBench v1.6 (full 1651-page set)**, across three inference backends delivered in sequence:

1. **transformers** (native, the on-machine oracle / absolute baseline)
2. **vLLM** (native `HunYuanVL`, ROCm serving)
3. **llama.cpp** (HIP on gfx1100, the headline "runs on AMD GPU via llama.cpp" path)

A frozen **decoding contract** is shared by all three backends, so the backend is the only variable. The transformers phase's full-set score is the **ground-truth baseline**; vLLM and llama.cpp must each match it within a tight tolerance.

**Standalone-first (per user direction):** the project begins as an **independent, self-contained repo**. All files — this spec, the three backends, the decoding contract, the OmniDocBench eval driver, results, and phase reports — live inside `HunyuanOCR-ROCm` at `/workspace/HunyuanOCR-ROCm`. Only after adaptation + full evaluation succeed (Phases 1–3) is it integrated into `omnidocbench-amd` as a platform sub-project (**Phase 4**): wrapped behind the adapter contract, registered in `hub/registry.yaml`, and badged. Until then the standalone project runs its **own** OmniDocBench eval directly and does **not** depend on the platform harness, the cookiecutter, or the adapter contract.

### 1.1 Non-goals (explicit)

- **License posture** — deferred by user decision. The Hunyuan Community License (source-available, not OSI-open) is marked "Powered by Tencent Hunyuan"; the hub badge stays `community` (not `verified`). Formal open-source licensing is resolved **after** adaptation + eval succeed.
- **windows-hip** platform.
- **DFlash** speculative decoding (speed-only, irrelevant to precision).
- **Q4/Q5 quantization** (precision-hostile for OCR).
- **SGLang** (not viable on gfx1100 — CDNA-only ROCm path).
- **Throughput optimization** as a gate (#20934 is tracked, reported, but does not block the precision gate).

### 1.2 Success criteria

A backend **passes a phase** iff its full-set OmniDocBench v1.6 score is within tolerance of the transformers `BASELINE`:

- **Overall:** ±0.3
- **Per-task** (text EditDist-derived, formula CDM, table TEDS, reading-order EditDist): ±0.5

The upstream self-reported **94.74** (arXiv 2607.04884, Table 12) is a **sanity check, not the gate**: expect the on-machine transformers `BASELINE` to land ≈ 94.74 ± ~1.0; a deviation larger than that flags a protocol/dataset-revision mismatch to investigate *before* trusting `BASELINE`. The transformers run on *this machine* is the actual gate, because it absorbs local hardware/numerics drift that would otherwise be mis-attributed to a backend. (Note: the 1.0 README's older 94.10 figure is a *different, pre-v1.6* OmniDocBench protocol and is not comparable — only the v1.6 94.74 is the reference.)

---

## 2. Background & key research findings (grounding)

These are the load-bearing facts the design rests on (verified 2026-07-15).

### 2.1 Model architecture — `tencent/HunyuanOCR` (root = 1.5)

- `architectures: ["HunYuanVLForConditionalGeneration"]`, `model_type: hunyuan_vl`. **Native transformers architecture** (`transformers/models/hunyuan_vl/`), requires transformers ≥ 5.13.0. The `trust_remote_code=True` in upstream snippets is defensive cargo-culting — the code is mainline.
- **~1B params total**: native-resolution **Hunyuan-ViT** (~0.4B, patch16, 27 layers, `spatial_merge_size 2`, max res 4K) → adaptive MLP connector (out_hidden 1024) → **Hunyuan-0.5B** dense LLM (24 layers, hidden 1024, GQA 16H/8KV, tied embeddings, DeepSeek-family).
- **Position encoding = XDRope** (`rope_scaling.type: "xdrope"`, `xdrope_section [16,16,16,16]`, `beta_fast 32 / beta_slow 1`). **Not mrope / M-RoPE** — no `rope_dimension_sections`. This is the critical distinction from Qwen2.5-VL.
- NextN/MTP fields present (`num_nextn_predict_layers: 1`) but used only for DFlash training — ignored for AR serving.
- Context 131072; images up to 4K.
- Weights: HF `tencent/HunyuanOCR` (root=1.5, `v1.0/`=1.0, `dflash/`=draft). 1.5 is single-file safetensors.

### 2.2 Upstream inference recipes (mutually-incompatible environments)

The repo ships `inference/{vllm_0_18_1, nightly, transformers}/` — each pins incompatible transformers versions. This forces **three separate model-venvs** (§7.3).

- **Default sampling (1.5):** `temperature=0.0, top_p=1.0, top_k=-1, repetition_penalty=1.08, max_tokens=32768`. ⚠️ Differs from 1.0 (which used `top_k=1, rep_penalty=1.0`).
- **Post-processors:** `clean_repeated_substrings` (tail-repetition degenerate-loop stripper) + `norm_formula_HYOCR` (HunyuanOCR-specific formula normalization — the model emits mixed plain+LaTeX that breaks OmniDocBench's default matcher; ported from the 1.0 `utils.py`).
- **Image preprocessing** (`preprocessor_config.json`): `HunYuanVLImageProcessor` — patch16, merge2, `min_pixels=262144` (512²), `max_pixels=4194304` (2048²), CLIP norm (mean `[0.481,0.458,0.408]`, std `[0.269,0.261,0.276]`), NaViT patch-n-pack dynamic resolution.
- **Special tokens:** `image_start=120118`, `image_token=120120`, `image_newline=120121`, `image_end=120119`.
- **12 task types** via `--task-type`; OmniDocBench uses **`doc_parse`**.

### 2.3 llama.cpp support — official, already merged

- **HunyuanOCR is a first-class model in llama.cpp** (PR #21395), listed in `docs/multimodal.md` alongside PaddleOCR-VL / GLM-OCR / Deepseek-OCR / Dots.OCR.
- `hunyuan-vl` is a registered arch in `convert_hf_to_gguf.py`; `models/hunyuan-vl.cpp` compiles. Official weights: `ggml-org/HunyuanOCR-GGUF` (Q8_0 + mmproj-f16). The repo also ships its own `llama_cpp/` dir + `docs/llama_cpp.md`.
- **Build on gfx1100:** `cmake -DGGML_HIP=ON -DGPU_TARGETS=gfx1100 -DCMAKE_BUILD_TYPE=Release`. ROCm 7.2+ = first-class gfx1100 (no `HSA_OVERRIDE_GFX_VERSION`). No open correctness bug.
- **mtmd stabilized** in `llama-server` `/chat/completions` (PR #12898, b5332, 2025-05-09). XDRope handled correctly by the HunyuanOCR path.
- **The acknowledged gap:** upstream tags the llama.cpp path *"work in progress — accuracy not yet aligned."* This is the core precision work of Phase 3 — we are closing a known gap, not pioneering.
- **Quantization reality:** at ~1B params, **F16 ≈ 2 GB**, fits trivially in 24 GB VRAM. Run **F16 unquantized** for near-bit-exact alignment; the quantization-precision question that plagues most llama.cpp VLM ports essentially evaporates here.

### 2.4 gfx1100 runtime landscape

- **vLLM** — most mature; native `HunYuanVL`, same RDNA3 stack as Unlimited-OCR. ⚠️ Watch issue **#1**: some builds fall back to the slow transformers impl instead of native `HunYuanVL` — must confirm the build actually registers the native path.
- **transformers + PyTorch ROCm** — clean correctness oracle; hard `transformers==5.13.0` pin; slower than vLLM.
- **SGLang** — not viable on gfx1100.
- **llama.cpp HIP** — functionally correct; throughput risk only (issue **#20934**: HIP token-gen ~20–30% slower than Vulkan on RX 7900, open 2026-07-15).

### 2.5 License (deferred)

Hunyuan weights are under the **Tencent Hunyuan Community License** (source-available, not OSI-open): EU/UK/KR territory exclusion, 100M-MAU clause, no-distillation clause, "Powered by Tencent Hunyuan" marking required. Handling is deferred (§1.1).

---

## 3. Decisions (locked)

| # | Decision |
|---|---|
| D1 | Repo = **standalone, self-contained** `AIwork4me/HunyuanOCR-ROCm` at `/workspace/HunyuanOCR-ROCm` (all files inside; runs its own OmniDocBench eval). Platform cookiecutter / `contracts/adapter.md` / `hub/registry.yaml` / badge = **Phase 4 only**, after eval. |
| D2 | Backend sequence **transformers → vLLM → llama.cpp**, gated on precision (§1.2). |
| D3 | Precision gate = **on-machine transformers baseline ± 0.3 overall / ± 0.5 per-task**; upstream 94.74 is sanity ceiling only. |
| D4 | **F16 unquantized** baseline for all backends (model is ~1B). Q8_0 is a stretch ablation, only after F16 alignment is proven. |
| D5 | License resolution **deferred** until after adaptation + eval. |
| D6 | Platform scope = `linux-rocm` / gfx1100 only. |
| D7 | Skeleton = **Approach A (vertical-slice-then-swap) + absorb C** (transformers retained as a regression oracle on a frozen canary subset). |
| D8 | Reporting = **per-phase report + cross-backend score table**, committed under `reports/` + `results/`. |

---

## 4. Repository structure (standalone)

Local root: `/workspace/HunyuanOCR-ROCm`. Everything lives here; the OmniDocBench-AMD platform repo is untouched until Phase 4.

```
HunyuanOCR-ROCm/
├── adapter/                          # contract surface (check_conformance.py target)
│   ├── run_adapter.py                # run_adapter(img_dir,out_dir,*,platform,config) → backend dispatch
│   ├── adapter_config.py             # BACKEND | SERVER_URL | WEIGHTS_DIR | API_MODEL_NAME
│   ├── backends/
│   │   ├── transformers_backend.py   # P1 oracle (in-process)
│   │   ├── vllm_client.py            # P2 → vLLM OpenAI server
│   │   └── llama_cpp_client.py       # P3 → llama-server
│   └── setup/                        # install scripts for the 3 mutually-incompatible model-venvs
│       ├── .env.local.example
│       ├── setup-transformers.sh
│       ├── setup-vllm.sh
│       └── setup-llamacpp.sh
├── src/hunyuan_ocr/                  # frozen decoding contract (backend-agnostic; mirrors rocm_ocr)
│   ├── contract.py                   # CONTRACT: prompt / sampling / image cfg / special tokens
│   ├── preprocess.py                 # HunyuanVL preprocessing (shared reference)
│   ├── postprocess.py                # clean_repeated_substrings + norm_formula_HYOCR
│   └── prompts.py                    # task-type prompt builders (doc_parse)
├── eval/configs/hunyuanocr-1.5_linux-rocm.yaml
├── results/omnidocbench/v16/linux-rocm/{transformers,vllm,llama-cpp}/
├── reports/{phase1-transformers,phase2-vllm,phase3-llamacpp}.md
├── scripts/{compare_scores.py, regression_canary.py, build_llamacpp_hip.sh}
├── model_card.json  Makefile( demo | eval-linux | oracle-check | publish )  docs/
├── pyproject.toml  tests/  examples/  LICENSE
```

**Storage plan** (per the 10 GB-NFS constraint): repo code under `/workspace`; weights, OmniDocBench dataset (~5 GB), and all venvs symlinked to `/root` (3.5 TB overlay). `adapter/setup/.env.local` (gitignored) holds absolute paths.

**Registry entry** (added to OmniDocBench-AMD's `hub/registry.yaml` **only at Phase 4**; until then the project is unregistered):

```yaml
- model_id: hunyuanocr-1.5
  repo: AIwork4me/HunyuanOCR-ROCm
  platforms:
    linux-rocm: {badge: community, overall: null}   # filled with the measured BASELINE at Phase 4
    windows-hip: {badge: community-wanted, overall: null}
```

Until Phase 4 the project does **not** conform to `contracts/adapter.md` and runs OmniDocBench through its own driver under `eval/`.

---

## 5. The frozen decoding contract (`src/hunyuan_ocr/contract.py`)

The **single shared layer** across backends. Frozen once P1 establishes `BASELINE`; any change re-baselines all phases.

```python
CONTRACT = Contract(
    task_type="doc_parse",
    chat_template=IMAGE_FIRST_CHAT_TEMPLATE,  # emit <image> per image part, then text
    special_tokens=dict(image_start=120118, image_token=120120, image_newline=120121, image_end=120119),
    sampling=dict(temperature=0.0, top_p=1.0, top_k=-1, repetition_penalty=1.08, max_tokens=32768),
    image=dict(
        patch_size=16,
        merge_size=2,
        min_pixels=262144,
        max_pixels=4194304,  # 512² .. 2048²
        norm_mean=[0.481, 0.458, 0.408],
        norm_std=[0.269, 0.261, 0.276],
        mode="native_pack",
    ),  # NaViT patch-n-pack
    postprocessors=["clean_repeated_substrings", "norm_formula_HYOCR"],
)
```

**Rationale:** the only way three different runtimes (in-process transformers, OpenAI-compatible vLLM server, llama-server `/chat/completions`) can be precision-compared is if everything *except* the backend is byte-identical. The contract captures prompt construction, exact sampling, image preprocessing config, and the two post-processors. Each backend client is responsible for mapping `CONTRACT` onto its runtime's knobs (e.g., llama.cpp `repetition_penalty` semantics, vLLM `extra_body` for params not in the OpenAI schema).

**Phase-3 risk concentrated here:** llama.cpp's `mmproj` performs its own image preprocessing internally and may not be byte-identical to `HunYuanVLImageProcessor`'s patch-n-pack. This is the most likely root of the upstream-acknowledged accuracy gap and the primary thing P3 must reconcile.

---

## 6. Phases (sequence, each gated on §1.2)

### Phase 1 — transformers (oracle / absolute baseline)

- **Env:** dedicated `model-venv-transformers` (Python 3.12): `transformers==5.13.0`, PyTorch ROCm (gfx1100).
- **Weights:** `tencent/HunyuanOCR` (root=1.5) → `/root/models/HunyuanOCR`.
- **Inference:** in-process `HunYuanVLForConditionalGeneration.from_pretrained` + `AutoProcessor` + `.generate()`, bf16, `CONTRACT` sampling.
- **Eval:** full 1651-page OmniDocBench v1.6 → `adapter/backends/transformers_backend.py` writes `<stem>.md` + `_run_stats.json` → `LinuxRocmBackend.score` runs `pdf_validation.py` in the eval-venv (3.11).
- **Exit criteria:** measured `BASELINE` (expected ≈ 94.74 overall). **Freeze CONTRACT.** Freeze a **canary subset** (~50–100 diverse pages) under `tests/fixtures/`. Publish `reports/phase1-transformers.md` with the per-task score table.
- **Absorb-C setup:** the transformers backend stays wired as the regression oracle (`scripts/regression_canary.py`) for the lifetime of the project.

### Phase 2 — vLLM (gate ±0.3 / ±0.5)

- **Env:** `model-venv-vllm`: vLLM 0.18.1 on ROCm gfx1100, `BUILD_FA=0`. OpenAI-compatible server, `--served-model-name tencent/HunyuanOCR` (so `CONTRACT` model resolution matches). Launched per the harness-background-task pattern (foreground `vllm serve` gets exit-144-killed).
- **Guardrail:** confirm the build **registers native `HunYuanVL`** (grep the vLLM model registry on startup) — issue #1's transformers-fallback must not trigger.
- **Inference:** `adapter/backends/vllm_client.py` posts image-first chat to the server; `CONTRACT` sampling passed through (`extra_body` for non-OpenAI params).
- **Gate:** full-set score within tolerance of `BASELINE`. If drift: bisect preprocessing / sampling passthrough / chat template (canary first, then full set).
- **Exit:** `reports/phase2-vllm.md` with the transformers-vs-vLLM table.

### Phase 3 — llama.cpp (gate ±0.3 / ±0.5 — the hard phase)

- **Build:** `scripts/build_llamacpp_hip.sh` compiles llama.cpp master with `-DGGML_HIP=ON -DGPU_TARGETS=gfx1100 -DCMAKE_BUILD_TYPE=Release`; ensure the build includes the #25373 fix (`-fno-finite-math-only`).
- **Weights:** **F16, converted locally** — `convert_hf_to_gguf.py --outtype f16` (base) + `--mmproj --outtype f16`, from the HF weights. (Not the published Q8_0 — we want the F16 oracle.)
- **Serve:** `llama-server` (base F16 + mmproj F16) `/chat/completions`, context `-c` sized for 4K pages.
- **Inference:** `adapter/backends/llama_cpp_client.py`.
- **Method:** localize the gap on the **canary** first (preprocessing parity / `repetition_penalty` mapping / chat template), then run full set. If a genuine precision bug in llama.cpp's `hunyuan-vl` path is found → **fix and upstream a PR** to `ggml-org/llama.cpp` (the substantive open-source contribution).
- **Throughput:** benchmark HIP vs `-DGGML_VULKAN=ON` (issue #20934); **report** the numbers in the phase report; throughput does **not** block the gate.
- **Stretch (after F16 passes):** Q8_0 ablation to confirm near-losslessness; record in the report.
- **Exit:** `reports/phase3-llamacpp.md` with the full three-backend table (transformers / vLLM / llama.cpp, overall + per-task + Δ).

### Phase 4 — Integration into OmniDocBench-AMD (only after Phases 1–3 pass)

Triggered once F16 llama.cpp passes the §1.2 gate. This is the **only** phase that touches the platform repo:

- Wrap the already-working standalone backends behind `contracts/adapter.md` (`run_adapter(img_dir, out_dir, *, platform, config)`), reusing the existing `src/hunyuan_ocr/` contract and per-page `.md` output — the standalone structure maps onto the adapter with minimal glue.
- Add the `hub/registry.yaml` entry (§4) with the measured `overall`.
- Run `scripts/check_conformance.py`; produce conformant, provenance-complete artifacts under `results/`.
- Set the badge to `community` (license posture unresolved → not `verified`).
- Optional later: stage a `verified` reproduction once the license is settled (separate post-eval decision).

---

## 7. Evaluation harness & precision gate

### 7.1 Shared pipeline

OmniDocBench v1.6 download (1651 pages, ~5 GB → `/root`) → `run_adapter` infer (per-page `<stem>.md` + `_run_stats.json`) → `LinuxRocmBackend.score` (`pdf_validation.py` in eval-venv 3.11: EditDist for text + reading-order, TEDS for tables, CDM for formulas opt-in) → `metric_result.json`.

### 7.2 Score comparison & gate

`scripts/compare_scores.py` reads each backend's `metric_result.json`, prints `transformers-BASELINE / vLLM / llama.cpp` (overall + per-task + Δ), and asserts the §1.2 tolerance. A backend passes its phase iff green.

### 7.3 Regression oracle (absorb C)

`scripts/regression_canary.py` runs the transformers backend on the frozen canary subset and compares any backend's canary output. Used as a minute-level diagnostic during P2/P3 debugging and as a CI-style check that no later change silently regressed alignment.

---

## 8. Error handling & robustness

- **Per-page isolation:** try/except per image, record `failed: <reason>` in `_run_stats.json`, continue, never raise (contract requirement; a missing page scores zero). Mirrors the template + Unlimited-OCR.
- **Resumable + sharded:** skip pages whose `.md` already exists; shard across GPUs. Matches Unlimited-OCR.
- **Runaway-output guard:** `max_tokens` hard cap + `is_looping_output` detection → two-pass retry with adjusted params (1.5's primary anti-loop is `rep_penalty=1.08` + `clean_repeated_substrings`; keep the two-pass fallback).
- **Three mutually-incompatible model-venvs:** `model-venv-{transformers,vllm,llamacpp}` (upstream's "validated constraint" that the three setups pin incompatible transformers). The engine dispatches the `infer` subprocess to the correct model-venv; the eval-venv (3.11) stays separate for scoring.

---

## 9. Testing

`tests/`:

- **Conformance:** `check_conformance.py` validates the adapter signature + output convention.
- **Contract integration:** adapter produces valid `_run_stats.json` + `.md`s on a 3-image smoke set.
- **Per-backend smoke:** each backend loads + produces markdown on the smoke set.
- **Precision-gate test:** a backend's canary score is within tolerance of `BASELINE`.
- **Schema:** `_run_stats.json` / `metric_result.json` validate against the artifact schema.

CI runs CPU-only (platform reality): smoke + conformance + schema. Real precision runs execute manually on gfx1100 and are documented (not in CI).

---

## 10. Risks & mitigation

| Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|
| P3 llama.cpp precision gap (mmproj preprocessing ≠ HunyuanVLImageProcessor; rep_penalty semantics) | High | High | F16 unquantized removes quant error; canary-first bisection; upstream fix + PR if a real bug. |
| Throughput #20934 (HIP tg slower than Vulkan) | High | Low (non-blocking) | Benchmark both backends; report; vLLM remains the throughput tier. |
| `transformers==5.13.0` pin fragility | Medium | Medium | Pin exactly; isolate in `model-venv-transformers`; fall back to upstream's `inference/transformers/requirements.txt` pin if needed. |
| Three mutually-incompatible model venvs | Medium | Medium | Separate venvs + setup scripts; engine dispatch; document the constraint. |
| vLLM issue #1 (transformers fallback) | Medium | Medium | Verify native `HunYuanVL` registration at server startup. |
| License blocks public release | Low (deferred) | Medium | Defer per user; mark "Powered by Tencent Hunyuan"; resolve after eval. |

---

## 11. Open questions

None blocking. The deferred license posture (§1.1) and the Q8 stretch (§6 P3) are the only explicitly-parked items.

---

## 12. Out of scope / deferred

- **Platform integration (Phase 4)** — deferred until Phases 1–3 pass; the project is standalone until then.
- License resolution (D5) — post-eval.
- windows-hip, DFlash, Q4/Q5 quants, SGLang (§1.1).
- Q8_0 near-losslessness ablation — stretch after F16 alignment.
