# Draft comment for PR #53856 (NOT posted)

Independent gfx1100 validation: #53856 also fixes a finite stale-V manifestation of the same final-block padding issue.

On W7900D, Muse eager ctx8192 had a reproducible 2/8 greedy split after a ctx512 phase. At the first attention divergence, q and all valid KV entries were bit-identical, but changing only V-cache slots beyond seq_len in the final referenced block changed the ROCm custom paged-attention output (1 of 780 tested invocations, single element, ~1.5 ULP).

A/B on the same vLLM build:
- without #53856: ctx8192 = 2/8; stale-tail mutation changes output
- with #53856: ctx8192 = 1/8; stale-tail mutation no longer changes output

The stale entries here were finite (0 NaN / 0 Inf, observed range about −9.25…8.5), so this appears to be another observable consequence of consuming invalid final-block V slots rather than a separate mechanism.

Evidence: https://github.com/AIwork4me/HunyuanOCR-ROCm/tree/validation-50603-rdna3-upstream-cleanup/validation-50603/rdna3-w4a16-upstream-cleanup/53856-ab
