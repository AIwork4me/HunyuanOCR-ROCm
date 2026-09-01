# Decision — CASE A

## Criteria (from the task protocol)

CASE A requires BOTH:

1. baseline eager8192 = 2/8 AND #53856 eager8192 = 1/8
2. baseline stale-tail mutation changes attention output AND #53856 stale-tail mutation does NOT change attention output

## Measured

1. E2E: baseline 2/8 (3 independent processes, first_div=27 in every diverging run); candidate (#53856) 1/8 (2 independent processes); candidate graphs-mode also 1/8 at both depths.
2. Causal: V-tail mutation (slots beyond `seq_len` in the final referenced block set to +1.0, everything else identical, output compared bitwise):
   - baseline: 780 tested, 1 changed — exactly the known divergence site (s0=8199, rem=7, element [0,3,104], max |Δ| = 1.221e-04 ≈ 1.5 ULP, matching the root-cause measurement)
   - candidate: 780 tested, 0 changed

Both criteria hold → **CASE A**.

## Conclusion (narrowly stated)

The observed finite stale-slot sensitivity is not a separate mechanism: it is another observable manifestation of consuming invalid final-block V-cache slots beyond `seq_len` in the ROCm custom paged-attention kernel, and the masking added by #53856 removes it on gfx1100 — including for finite stale values, not only the NaN case (our stale entries were 0 NaN / 0 Inf, range −9.25…8.5).

## Recommended upstream action

- Do NOT open a new ROCm-attention issue.
- Prepare an independent gfx1100 validation comment for #53856 (`github-53856-comment-draft.md`; not posted).
- Keep the W4A16 determinism PR independent (branch `fix/rdna3-w4a16-determinism` @ bd5d05816b, already pushed; not opened).
- Update #50603 to link both mechanisms/fixes (`github-50603-update-draft.md`; not posted).

## Scope notes

- All runs on W7900D (gfx1100), Muse-Glimmer-30B-INT4, TP=1, eager (plus one graphs sanity run), prefix caching on, identical protocol to the validated cleanup stage.
- Routing verified by wrapping the custom-path entry point (1612 calls, identical across A/B) and by the cross-build behavioral difference itself.
- W4A16 regression on the candidate: 8/8 PASS (stacking #53856 does not disturb the W4A16 fix).
