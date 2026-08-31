# version-topology-ab — vLLM 0.25.1 vs 0.23.1.dev1, TP=1, on the 48 GB W7900D

Evidence for [vllm-project/vllm#50603](https://github.com/vllm-project/vllm/issues/50603): the TP=1 / vLLM-version side of the gfx1100 greedy-decoding nondeterminism that @cadamcat measured at TP=2 ([their evidence](https://github.com/cadamcat/dual-radeon-vllm/tree/main/benchmarks/gfx1100-greedy-eager-ab)), which their box cannot test (both models exceed a single 19.98 GiB card).

**Headline (Muse primary, 3 engines × 2 eager-states × 2 depths × 2 versions):**

| ctx | vLLM 0.25.1 graphs/eager (unique of 8, 3 engines) | vLLM 0.23.1.dev1 graphs/eager | cadamcat TP=2 · 0.23.1.dev1 (1 engine) |
|--:|---|---|---|
| 512 | **8,6,7 / 7,8,8** — varies in 6/6 | **8,7,8 / 8,7,8** — varies in 6/6 | 5 / 5 |
| 8192 | 1,1,3 / 3,3,2 — varies in 3/6 | 2,3,1 / 2,1,2 — varies in 5/6 | 1 / 3 |

gemma-3-27b (secondary, 1 engine/cell): **byte-stable at ctx=512 on both versions (1/8, four cells) while varying at ctx=8192 (2–3/8, four cells)** — the same model-dependent fingerprint as cadamcat's TP=2 cells (gemma 512: stable both ways; 8192: 5/2).

**Answer to Q1/Q2:** yes, the nondeterminism reproduces on the W7900D at TP=1, on **both** vLLM 0.25.1 and 0.23.1.dev1, with and without `--enforce-eager`. 21 of 24 primary Muse cells vary (10/12 on 0.25.1, 11/12 on 0.23.1.dev1); a 0.25.1 control at default mm limits varies in 11/12. **Answer to Q3 (Case C+D):** the remaining axis points away from vLLM version and away from TP/distributed topology — toward the model/quantization/prompt regime (both models' fingerprints match cadamcat's TP=2 results on our TP=1 box) and beneath vLLM to the shared runtime stack. Our earlier HunyuanOCR no-repro on the same box/stack/0.25.1 does not generalize to this model/harness — that is itself an important negative result.

## 1. Question

1. Does the nondeterminism reproduce on a W7900D at TP=1?
2. Does it differ between vLLM 0.25.1 and 0.23.1.dev1?
3. Do the results point more strongly at the vLLM version, the TP/distributed topology, or another environment difference?

## 2. Hardware

Single AMD Radeon PRO W7900D, gfx1100, 51,522,830,336 B (47.98 GiB) VRAM (`rocm-smi --showproductname`: `GFX Version: gfx1100`, Card Model 0x744b, Unique ID 0x9187179a3606ca10). AMD EPYC 9334, 1 TiB RAM, Ubuntu 24.04.4. Full capture: `environment/00-machine-environment.txt`.

## 3. Software

| layer | both arms | notes |
|---|---|---|
| vLLM arm A | **0.25.1**, git `752a3a504485790a2e8491cacbb35c137339ad34` (clean `v0.25.1` tag) | source build, worktree of upstream |
| vLLM arm B | **0.23.1.dev1+g9ddef7117**, git `9ddef71179f5058983a487bb0f94ead39abba900` (2026-07-14, parent = `v0.23.0` tag) | the exact `+g` commit of cadamcat's version string; source build, worktree of upstream |
| torch | 2.12.0+rocm7.14.0 (+ amd-torch-device-gfx1100 leaf) | identical wheels in both venvs |
| triton | 3.7.1+git0263a6a6.rocm7.14.0 | identical |
| transformers | 5.15.1 | identical; first release line with `muse_glimmer`; satisfies both arms' requirement pins |
| Python | 3.12.3 | identical |
| ROCm userspace | wheel stack (rocm 7.14) for runtime; system 7.2.1 hipcc as build toolchain (same for both arms — the prior validation-50603 canonical build configuration) | |

Dependency control: `pip freeze` diff between the two arms = `vllm` itself + `gguf` (0.23 requirement, unused by safetensors models). Everything else byte-identical versions. Freeze manifests in `environment/`.

Known deviations vs cadamcat's box (disclosed, not hidden):

- **torch 2.12.0+rocm7.14.0, not their torch 2.11.0.** The exact 2.11 stack does not initialize on this container (prior evidence: `validation-50603/baseline-torch211-unrunnable/`, `hipErrorInvalidValue` at engine init, two wheel configs tried). Holding torch constant across our two arms was prioritized over matching their torch.
- **No `flash_attn` package** (their `rocm/vllm` container ships it). Consequence and mitigation: `artifacts/mm-zero-decision.md` — both arms run with `limit_mm_per_prompt={"image":0,"video":0}`; the measured path is text-only and the ViT encoder never executes at generate time.
- **No local kernel patches** — their site-packages carried a sliding-window block-skip patch (their campaign states the nondeterminism is symmetric between patch states).
- Python 3.12 here vs 3.14 in their image.

## 4. Upstream harness

cadamcat/dual-radeon-vllm @ `782c7d431ab8e821242f2a717bf5d03b0be3301d`, `benchmarks/gfx1100-greedy-eager-ab/nondet_eager.py`. Analysis: `environment/cadamcat-harness-analysis.md`; frozen-clone info: `environment/upstream-clone-info.txt`.

## 5. Local harness diff

`environment/harness.diff` (vs upstream `nondet_eager.py`). Executable changes: TP 2→1; model paths → pinned HF snapshots (`RedHatAI/Muse-Glimmer-30B-INT4@f5b410ce…`, 21 GiB on disk — matching their recorded size; revision-pinned download, `HF_HUB_OFFLINE=1` at run time); evidence JSON extended (SHA256 per generation, decoded text, environment/version metadata); explicit engine teardown; env-gated `limit_mm_per_prompt` (off by default; on for the primary matrix, symmetrically — see `environment/mm-zero-decision.md`). Generation semantics (prompt `1000 + (i % 20000)`, warm-up max_tokens=8 ×1, measured temperature=0.0/max_tokens=64/ignore_eos ×8, max_model_len=8704, max_num_seqs=128, gpu_memory_utilization=0.92) byte-identical.

## 6. Method

Per matrix cell: one fresh engine process; per depth (512 then 8192, same engine): exactly one warm-up generation (max_tokens=8) then eight identical greedy generations (max_tokens=64); all token IDs, SHA256s, and decoded text saved; distinct-sequence count and first-divergence-vs-run-0 computed. Cells: {ctx 512, 8192} × {enforce_eager=0, 1} × 3 independent engines per arm. Raw JSON per engine in `results/`, engine logs in `logs/`, driver log `logs/matrix-*.log`. A first 0.25.1 matrix at default mm limits (12/12 cells) is kept in `results/vllm-0.25.1-defaultmm/` as a config-robustness control.

Reproduce: `scripts/run_matrix.sh <env> <tag>` with `scripts/build_env.sh <env> <worktree>` and `scripts/download_models.sh` for the environments.

## 7. Exact matrix

Primary matrix (12 cells/arm): Muse-Glimmer-30B-INT4 (primary). gemma-3-27b-it-w4a16 (secondary) — secondary cells present for both versions (1 engine per cell, `results/*/gemma3-*.json`).

## 8. Results

All numbers recomputable from `results/*.json` (raw token IDs + SHA256s + decoded text per generation; `results/ab-analysis.json` is the derived table source).

- Muse primary matrix: `results/vllm-0.25.1/`, `results/vllm-0.23.1.dev1/` (+ per-arm summary.md). Control: `results/vllm-0.25.1-defaultmm/`.
- Full comparison incl. divergence indices and run-group structure: `results/version-ab-comparison.md`.
- Structure findings: muse/8192 pools 48 generations/arm into **5–6 distinct sequences, 4 byte-identical across the two arms** (a shared attractor set); muse/512 gives 41–46 distinct per arm. Divergence at 512 lands at token 1–44; at 8192 at index 1–2 or never.
- cadamcat's "settles after three generations" pattern (muse/512/eager, TP=2) does **not** reproduce at TP=1 (18 muse/512 engines show scattered groups, 10 fully singleton); reported as absent, not forced.
- Failed cells shown: the initial 0.23.1.dev1 matrix without the mm limit failed 4/4 engines at init (ViT-profiling OOM, raw log `logs/matrix-vllm-0.23.1.dev1-failed-nommlimit.log` + regenerated engine log `logs/muse-e0-eng9-regen.log`); diagnosis and fix: `environment/mm-zero-decision.md`.

## 9. Interpretation

Facts: both versions vary at TP=1; `--enforce-eager` does not remove it on either; engine-to-engine spread on both; model fingerprint identical to the TP=2 box; 8192 attractor sets shared across versions.

Reading (weaker than the evidence is marked): **Case C** — the issue reproduces at TP=1 and is not explained by 0.23-vs-0.25 nor by distributed topology; **Case D facet** — our prior #50603 no-repro (HunyuanOCR, same box, same stack, clean v0.25.1) does not generalize: the phenomenon is model/regime-dependent. Consistent with argmax flipping on near-tied logits whose low-order bits vary between executions, but the mechanism is NOT established here. Version bisect skipped: its precondition (0.23 bad / 0.25 good) is not met.

## 10. Limitations

- torch is 2.12.0+rocm7.14.0 in both arms; cadamcat's exact runtime is torch 2.11.0, which does not initialize on this container (prior evidence kept in validation-50603/baseline-torch211-unrunnable/). vLLM version and torch are jointly different from their box; within our A/B, torch is held constant and only vLLM varies.
- `flash_attn` (present in their rocm/vllm image) is absent here; consequence and symmetric mitigation (mm-limit zero) documented in `environment/mm-zero-decision.md`. A 0.25.1 control at default mm limits shows the same conclusion.
- Their site-packages carried local kernel patches (window block-skip); ours are stock upstream. Their campaign found the effect symmetric across patch states.
- Python 3.12 here vs 3.14 in their image; transformers necessarily ≥5.15.0 here for muse_glimmer support (5.15.1 pinned, both arms).
- Secondary model ran 1 engine per cell (as cadamcat published); primary ran 3.
- The raw per-engine log of the first failed 0.23 no-mmlimit attempt was overwritten by cleanup; the failure was re-reproduced and captured as `logs/muse-e0-eng9-regen.log` (identical 16.00 GiB OOM), and the driver log of the original attempt is kept.

## 11. Reproduce

```bash
# environments (one per arm; ~10 min each on this box)
harness/build_env.sh env-0.25.1      worktrees/v0.25.1
harness/build_env.sh env-0.23.1.dev1 worktrees/g9ddef7117
# models (pinned revisions, via hf-mirror)
harness/download_models.sh
# matrices (one engine per JSON, 3 engines × eager 0/1 per arm)
harness/run_matrix.sh env-0.25.1      vllm-0.25.1
harness/run_matrix.sh env-0.23.1.dev1 vllm-0.23.1.dev1
# tables
python3 harness/analyze_ab.py results/
```

Artifact checksums: `SHA256SUMS` in this directory.
