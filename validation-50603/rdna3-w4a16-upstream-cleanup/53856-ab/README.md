# 53856-ab — does #53856 fix the residual Muse eager ctx8192 nondeterminism?

**Did #53856 fix Muse eager8192?** Yes — ctx8192 goes from 2/8 (first_div=27, stable) to 1/8 across repeated two-phase runs, everything else identical.
**Did #53856 remove stale-tail output sensitivity?** Yes — mutating only the V-cache slots beyond `seq_len` in the final referenced block changed the custom paged-attention output on the baseline (1/780 invocations, the exact divergence site, max |Δ| = 1.221e-04 ≈ 1.5 ULP) and never changes it on the candidate (0/780, bit-identical).
**Which CASE: A / B / C?** **CASE A — same root cause; #53856 fixes the residual.**

The stale entries in this reproducer were finite (0 NaN, 0 Inf across all 780 recorded tails; observed range −9.25 … 8.5, i.e. historical activation values), so the same V-tail masking also removes the finite stale-value manifestation of the invalid final-block V consumption on gfx1100.

A/B construction: two worktrees of vLLM main @ `07ea9350baf84e33fd696d36fec9b9f24735a733`; baseline = base + `upstream-final.patch` (W4A16 determinism patch); candidate = base + `upstream-final.patch` + PR #53856 (head `80e801cbfb7e6501f79c7ecd75aeb9b37ecf2561`, patch SHA256 `ba55bd3f…77775c5`). The W4A16 files are SHA-256-identical across both trees; the only difference is `csrc/rocm/attention.cu` (+ its test). See `environment.md`.

Routing was verified, not inferred: the custom paged-attention entry (`vllm._custom_ops.paged_attention_rocm` → `torch.ops._rocm_C.paged_attention`) was wrapped and counted — 1612 custom-path calls in both builds (780 recorded NoPE-decode invocations + 780 mutation reruns + 52 prefill-pass calls); the cross-build behavior difference in the mutation test itself proves the #53856-modified kernel executed. See `routing.log`.

Files: `environment.md` (setup + PR identity), `baseline-two-phase.log` / `with-53856-two-phase.log` (E2E A/B incl. a graphs-mode run), `stale-before.{json,log}` / `stale-after.{json,log}` (causal mutation test + finite-value stats), `routing.log`, `w4a16-regression.log` (8/8 PASS on candidate — stacking #53856 does not disturb the W4A16 fix), `summary.json`, `decision.md` (CASE A classification), `github-53856-comment-draft.md` and `github-50603-update-draft.md` (drafts only, NOT posted), `SHA256SUMS`.
