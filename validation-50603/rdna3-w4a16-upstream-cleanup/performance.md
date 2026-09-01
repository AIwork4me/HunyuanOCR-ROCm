# Final performance — cleaned deterministic epilogue vs atomic baselines

Protocol: real Muse q_proj weights (K=6656, N=4096, bf16), CUDA-event latency (50 warmup + 200 timed, median), determinism = distinct SHA-256 outputs / 100 repeats. Same W7900D machine for candidate and baselines.

Baseline selection is dispatch-aware: for each M the baseline is the atomic kernel that the original upstream dispatch routes that M to (scalar for M<16, WMMA otherwise), measured on the unmodified build. The old production build routed M=16..64 to scalar, so its scalar-atomic numbers are additionally shown for reference.

| M | route (upstream dispatch) | atomic baseline | clean deterministic | delta | baseline determinism |
|---|---|---|---|---|---|
| 1 | scalar | 43.1 µs | 45.1 µs | +4.6% | nondeterministic |
| 8 | scalar | 70.0 µs | 73.3 µs | +4.7% | nondeterministic |
| 16 | WMMA v1 | 97.8 µs | 81.0 µs | −17.2% | 100 distinct / 100 |
| 64 | WMMA v3/v4 | 183.0 µs | 172.4 µs | −5.8% | 100 distinct / 100 |
| 128 | WMMA head | 182.3 µs | 176.4 µs | −3.2% | 1 distinct / 100 |
| 512 | WMMA head | 535.0 µs | 554.8 µs | +3.7% | 1 distinct / 100 |

All clean-build cells: 1 distinct / 100. Reference (old production build, M=16/64 routed to scalar atomic): M=16 → 99.7 µs, M=64 → 270.2 µs.

Interpretation:

- Small-M scalar pays the per-call FP32 scratch alloc + reduce pass: +4.6–4.7% (M=1: 45.1 vs 43.1 µs). This is the M=1 decode-relevant cost.
- WMMA small/mid M is *faster* deterministic than atomic (−17.2% / −5.8%): the CAS-loop epilogue the deterministic partials replace was the dominant cost there.
- Large-M head kernels pay the partials zero-init and reduce (up to ~144 MiB row-tiled scratch, M-independent): +3.7% at M=512.

Accuracy (vs fp32 dequant reference, M=1): max_abs_err 0.0061 (clean) vs 0.028 (atomic); cosine 0.9999986. See `logs/perf-clean.jsonl` and prior-stage `correctness/` JSONLs for raw values.

Data sources: `logs/perf-clean.jsonl` (clean); prior-stage production-fix evidence for `w_stock` (original WMMA atomic) and `prodLeg` (scalar atomic) rows on the same machine.
