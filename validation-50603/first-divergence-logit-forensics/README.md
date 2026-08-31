# First-Divergence Logit Forensics

## Question

Where does the gfx1100 greedy nondeterminism ([vllm#50603](https://github.com/vllm-project/vllm/issues/50603)) first become observable — in the model's logits, in logits processing, or in the sampler? Follow-up to [version-topology-ab](../version-topology-ab), which established the effect reproduces at TP=1 on both vLLM 0.25.1 and 0.23.1.dev1 with and without CUDA graphs.

## Reproducer

Single Radeon PRO W7900D (gfx1100, 47.98 GiB), vLLM **0.25.1** (git `752a3a504485790a2e8491cacbb35c137339ad34`, source worktree), torch 2.12.0+rocm7.14.0, triton 3.7.1+git0263a6a6.rocm7.14.0, transformers 5.15.1, Python 3.12.3. Model `RedHatAI/Muse-Glimmer-30B-INT4@f5b410ce…` (compressed-tensors W4A16, group 128, symmetric, static act-order; routed by `RDNA3W4A16LinearKernel` — same selection line as cadamcat's logs). Engine: TP=1, `enforce_eager=True`, max_model_len=8704, max_num_seqs=128, gpu_memory_utilization=0.92, `limit_mm_per_prompt={"image":0,"video":0}` (as in version-topology-ab). Prompt `[1000 + (i % 20000) for i in range(512)]`, warm-up ×1 (max_tokens=8) then 8 measured greedy generations (max_tokens=64, temperature=0.0, ignore_eos=True), one engine per process. Environment capture: `artifacts/00-forensics-environment.txt`.

## Why first divergence matters

At the first generated-token position `D` where two runs differ, both runs have consumed identical token histories and attended the same cached prompt KV, so any state difference observed at D is caused upstream of any token-choice difference — comparison there avoids downstream contamination.

## Baseline reproduction

3 fresh engines, original uninstrumented harness (`results/baseline/`): **8/8 distinct sequences at ctx=512 in every engine**; earliest divergent pair per engine at generated-token step 5, competing tokens 112740 vs 200007 in all three engines. `results/baseline-forensics.json`, `results/divergence-pairs.json`.

## Instrumentation and non-masking

- Probe 1 (`scripts/forensics_probe.py`): one **identity** logits processor (vLLM 0.25.1 `AdapterLogitsProcessor`, non-argmax-invariant group ⇒ runs after the float32 cast, immediately before `greedy_sample`); returns the same tensor object — see the pipeline map in `artifacts/token-selection-path.md` (for this config the processor chain is provably cast+identity, and bf16→fp32 is injective, so Stage-B equality ⟺ Stage-A equality). Records per step: sha256, top-64 ids/values, margins, own argmax; archives full fp32 rows per generation. **8/8 unique with the probe attached** — the bug is not masked; quant kernel, engine config and eager mode unchanged (`logs/probe-eng1.log`).
- Probe 2 (same + gated `sitecustomize` hook wrapping `LogitsProcessor._get_logits`, `scripts/lmhead-hook-sitecustomize.py`, injected via PYTHONPATH, inert without `FORENSICS_LMHEAD_HOOK=1`): records LM-head input-hidden and output-logits hashes per decode step. **8/8 unique with the hook attached** (`logs/probe-eng2.log`). Two failed hook iterations (stdout-polluting eager import; bf16-vs-numpy hash) are documented in the logs kept in `logs/`.

## First divergence

28 divergent pairs (8 runs ⇒ all pairs diverge); first-divergence steps 5–24. Full evidence: `results/first-divergence-forensics.json`; per-step metadata `results/probe{1,2}-steps.jsonl`; lossless bf16 snapshots at divergence and aligned positions — release assets [forensics-50603-snapshots](https://github.com/AIwork4me/HunyuanOCR-ROCm/releases/tag/forensics-50603-snapshots) (+ INDEX.json there; uploaded as release assets because the 42 MB pack exceeds this network's git transfer limit).

Example — pair run0 vs run1, D=11, identical histories through step 10:

| quantity | run A (0) | run B (1) |
|---|---:|---:|
| selected token | 112740 | 200007 |
| logit[112740] | 69.0 | 67.0 |
| logit[200007] | 67.0 | 67.5 |
| top1−top2 margin | 2.0 | 0.5 |
| argmax-input logits equal (sha) | no | no |
| max abs logit diff across vocab | 2.6875 | (same tensor pair) |
| offline CPU argmax of saved row | 112740 ✓ | 200007 ✓ |

Compact table for all pairs (`results/near-tie-analysis.md`):

| pair | D | tok A | tok B | argmax-input equal | max abs diff | margin A | margin B | offline argmax |
|---|--:|--:|--:|---|--:|--:|--:|---|
| 0v1 | 11 | 112740 | 200007 | no | 2.69 | 2.0 | 0.5 | ✓/✓ |
| 0v2 | 5 | 200007 | 112740 | no | 3.11 | 0.5 | 0 | ✓/✓ |
| …(all 28)… | 5–24 | … | … | **no (28/28)** | 2.1–4.1 | 0–2.0 | 0–2.0 | ✓/✓ |

Sampler consistency: the probe's own argmax equals the engine's selected token at **all 512 instrumented steps**.

## Classification

**Category B — forward/logits divergence with healthy margins.**

- argmax-input logits differ at every first-divergence position (28/28), by 2.1–4.1 logit units across the vocabulary, with identical token histories (identity re-verified per pair in `first-divergence-forensics.json`).
- Margins at divergence (median 0.5, max 2.0) are *smaller than* the measured cross-run logit variation at identical histories (median 3.25, max 8.8125 over 35 aligned positions) — this is not tiny-noise-crossing-a-tie (Category A); the forward-pass variation exceeds typical decision margins outright.
- Categories C/D are excluded: no processor transforms anything here (chain is float32 cast only; bf16→fp32 injective), and the sampler/argmax is consistent with its input at every step; offline replay of every saved divergence row (CPU×20, GPU×20, fresh process) is deterministic and matches vLLM's selection (`results/offline-argmax-replay.json`).

## Upstream boundary: hidden states (Phase 9) and the W4A16 GEMM (Phase 10)

`results/hidden-state-boundary.json`: with identical token histories, **all 8 generations have 8 distinct final-hidden-state sha256s at every decode step — including step 0**, the first decode step after the shared 512-token prompt. Call-counting proves every measured generation hit the shared prompt prefix cache (520 LM-head calls = 8 warm-up + 8×64 measured, zero prefill recomputes), so the aligned steps take **identical inputs** (same weights, same cached KV, same query token). The divergence therefore originates inside the decoder forward pass, upstream of the final hidden state and the LM head. `enforce_eager` excludes torch.compile.

Micro-probe of the model's quantized GEMM (`results/w4a16-microprobe.jsonl`, `scripts/w4a16_probe.py`): `torch.ops._rocm_C.gptq_gemm_rdna3` (`RDNA3W4A16LinearKernel`) driven through vLLM's own test-recipe layer builder with the **real** layer-0 q_proj weights (K=6656, N=4096, group 128) and a fixed seeded bf16 input:

| shape | proc A | proc B | proc C |
|---|---|---|---|
| M=1 (decode) | 200 calls → **200 distinct** outputs, maxerr 0.078 | 100 → **100**, 0.047 | 100 → **100**, 0.047 |
| M=128 (prefill) | 200 → 2 distinct, 0.012 | 100 → 2, 0.008 | 100 → 2, 0.016 |

The isolated W4A16 GEMM is **not bit-repeatable on fixed inputs**: at the decode batch shape M=1 every call differs and even the first call's output differs process-to-process; at prefill shape M=128 it settles into 2 variants per process. This is consistent with every upstream observation (decode steps vary maximally; the shared cached prefill — computed at M=128-class shapes — is stable enough to be reused).

## What this rules out (evidence-backed)

- The **sampler / argmax path**: consistent at 512/512 steps; offline replay deterministic and matching (Category D excluded).
- **Logits processing** (Category C excluded: chain is cast-only by code-path mapping, and Stage-B equality ⟺ Stage-A equality by cast injectivity; equality never held anyway).
- **The LM head matmul** as the *origin* (its input already differs), and **the token-history feedback** as the cause (step 0 already differs).
- **torch.compile** (eager mode throughout).
- **Near-tie amplification as the primary story** (Category A excluded): cross-run variation exceeds the margins.
- Compatible with cadamcat's kernel probe: they cleared the *Triton paged-attention* kernel with fixed inputs — a different kernel from the W4A16 GEMM implicated here.

## What remains

The nondeterministic execution is localized to the decode-shape (M=1) invocation of `gptq_gemm_rdna3` / `RDNA3W4A16LinearKernel` on gfx1100 — present in isolation, on real weights, in three independent processes. Not yet established: the mechanism inside that HIP kernel (scheduling/atomics/reduction order), whether other M=1 GEMM paths (bf16 attn projections, o_proj) also vary, and whether the same kernel on other gfx11 hardware reproduces it. One boundary to note: the micro-probe repacks the checkpoint's packed nibbles through vLLM's test packing; a nibble-order convention difference would change which real weights are exercised, but not the fixed-input/fixed-weight repeatability character being tested.

## Limitations

- Instrumentation adds per-step GPU→CPU copies (hash + top-64 + row archive) and one identity logits processor / one identity layer wrapper; both instrumented engines reproduced 8-of-8 unique with the same competing tokens as baseline, so masking is excluded empirically, but timing-sensitive effects cannot be fully excluded in principle.
- The forensics cell is Muse/ctx=512/eager on 0.25.1 only (by design — no matrix expansion); Phase-9/10 conclusions are statements about this reproducer.
- The bf16 snapshot files are lossless roundtrips of the engine's bf16 logits (the fp32 rows the argmax saw are exactly representable in bf16); full fp32 archives remain on the reproducer box at `/workspace/vllm-50603-version-ab/forensics/results/probe{1,2}/`.
- Margins/diffs are reported in raw logit units at bf16 resolution; no rounding applied in machine-readable outputs.

## Reproduce

```bash
harness:      scripts/forensics_probe.py        # identity logits processor probe
              scripts/lmhead-hook-sitecustomize.py  # gated _get_logits hook (PYTHONPATH inject)
analyses:     scripts/analyze_forensics.py <probe-dir> <run.json> <out-dir>
              scripts/offline_replay.py <probe-dir> <forensics.json> <out>
              scripts/lmhead_analysis.py        # hidden-state boundary
              scripts/w4a16_probe.py <tag> <repeats>
baseline:     scripts/run_baseline.sh           # 3 engines, uninstrumented
```

Environment: `/workspace/vllm-50603-version-ab` (env-0.25.1, worktrees/v0.25.1); see version-topology-ab for the full build recipe. All published numbers regenerate from the JSON/JSONL/PT files in this directory (`summary.json` holds the headline set). Checksums: `SHA256SUMS`.
