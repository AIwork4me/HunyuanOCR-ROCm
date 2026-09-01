# rdna3-w4a16-upstream-cleanup — vLLM #50603 upstream-acceptance cleanup evidence

Final upstream-cleanup validation for the gfx1100 (RDNA3) W4A16 split-K determinism fix. This directory contains only the cleanup-stage evidence; full root-cause and production-fix histories live in the two prior evidence directories (see links below).

## Patch under test

`patches/upstream-final.patch` — 3 files (two `.cu` + one regression test), applied on vLLM main `07ea9350baf84e33fd696d36fec9b9f24735a733`. Design: keep the existing compute kernels and the existing upstream dispatch; replace only the nondeterministic split-K epilogue on both paths with FP32 per-split partials → fixed ascending-z FP32 reduction → one final output cast; `k_split == 1` keeps the original direct store. Per-call scratch via the PyTorch caching allocator (no `thread_local`, no env switches; `VLLM_RDNA3_W4A16_LEGACY_ATOMIC` deleted).

## Validation summary (all runs real, W7900D / gfx1100)

- Micro determinism (real Muse q_proj K=6656 N=4096, bf16 and fp16): M ∈ {1, 8, 16, 64, 128, 512} all 1 distinct / 100 repeats.
- Regression test `tests/kernels/quantization/test_rdna3_w4a16_determinism.py`: 8/8 PASS (bf16+fp16 parameterized); 4/4 FAIL on unpatched 0.25.1 (fail-before holds).
- Per-call interception (16,640 GEMM calls hashed per generation): 0 kernel-level divergences (same-input-different-output) — the patched GEMMs are bit-deterministic.
- E2E (final build, same harness as prior stages): Muse graphs ctx512/8192 = 1/8; Muse eager ctx512 = 1/8; gemma-3 ctx512/8192 = 1/8; Muse eager ctx8192 = 2/8 — root-caused to an upstream ROCm custom paged-attention defect, NOT this patch: see `muse-eager-8192-attention-rootcause.md`. With `attention_backend=TRITON_ATTN` the same build and patch give 1/8 at both depths.
- Dual-stream sanity: two-stream results bit-equal. Accuracy: max_abs_err improved vs the atomic path at every M (e.g. 0.0061 vs 0.028 at M=1); cosine 0.9999976–0.9999986.

## Contents

- `README.md` — this file
- `summary.json` — machine-readable headline results
- `performance.md` — final perf table vs same-path atomic baselines
- `validation.md` — Phase 8 matrix results (scalar/WMMA/bf16/fp16/graphs/streams/E2E)
- `self-review.md` — Phase 12 maintainer-style review (PASS/FAIL per item)
- `muse-eager-8192-attention-rootcause.md` — full diagnostic chain for the one residual E2E cell (upstream attention defect, out of scope for this patch)
- `53856-ab/` — follow-up A/B proving that PR #53856 ("Mask paged attention V cache padding") eliminates the residual: CASE A (same root cause), E2E 2/8 → 1/8 and stale-tail causal sensitivity 1/780 → 0/780
- `patches/upstream-final.patch` — the patch (byte-identical to the one under test)
- `upstream-pr-final.md` — Phase 15 PR draft
- `github-comment-final.md` — Phase 14 issue comment draft (wording per task book)
- `logs/` — final E2E outputs (muse eager ×2 engines, graphs, gemma) + perf-clean.jsonl + probe logs
- `probes/` — reproduction probes for the attention root cause
- `SHA256SUMS`

## Prior evidence (do not duplicate)

- Root cause: `../rdna3-w4a16-rootcause/` (branch `validation-50603-rdna3-w4a16-rootcause`, commit 1615542)
- Production fix: `../rdna3-w4a16-production-fix/` (branch `validation-50603-rdna3-production-fix`, commit 5bb39af)
- First-divergence logit forensics: `../first-divergence-logit-forensics/`
