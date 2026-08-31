# Kernel reduction map — `csrc/rocm/q_gemm_rdna3.cu` @ vLLM 0.25.1 (`7523a5044`)

File sha256 at test time: `840fc8e76da64a85e22e41b63fad6afd7cbfd47b9ee7264d5a3df9d418e2f654`. All line references below are into that file.

## The production path (M=1 bf16 decode, the primary shape)

| item | value | source |
|---|---|---|
| `BLOCK_KN_SIZE` | **256** | `#define` line 67 |
| `THREADS_X` | 256 (8 waves, wave32) | line 68 |
| grid | `(ceil(N/1024), ceil(M/M_COUNT), ceil(K/256))` | `launch_gemm_q4_for_mcount` line 638-641 |
| **split-K** | `gridDim.z = K / BLOCK_KN_SIZE` | line 641 + header comment line 22 |
| **Z for K=6656** | **6656 / 256 = 26 split blocks** (exact) | computed |
| grid for primary shape (M=1, N=4096) | `(4, 1, 26)` = 104 blocks | computed |
| M dispatch | M=1→M_COUNT 1; 2-3→2; 4-7→4; else 8 (line ~665-690) | `launch_gemm_q4` |
| M≥16 bf16 / M≥64 fp16 | forwards to `gptq_gemm_rdna3_wmma` (separate TU) | `gptq_gemm_rdna3` lines ~725-733 |
| per-block accumulation | FP32 `block_c[m][4]`, `v_dot2_f32_bf16` fdot chains | lines 369-592 |
| output zero-init | `at::Tensor c = torch::zeros(...)` in the op | line ~758 |
| **epilogue** | cast each FP32 partial to bf16 (`__float2bfloat16`, line 611-615) then `atomic_add_pk4_bf16` — CAS-loop on a 64-bit word packing 4 bf16 lanes (`global_atomic_cmpswap_b64`, lines 174-230) | lines 594-618 |
| group handling | `groupsize = size_k/groups` (primary: 6656/52 = 128; each 256-K split spans exactly 2 groups) | lines 328-331 |

## Pseudocode of the current mechanism

```text
c[m][n] = bf16(0)                                    # host: torch::zeros
for split z in 0..25 in parallel (grid.z):           # each block owns K-slice [z*256, z*256+256)
    partial_fp32 = dot(a[m, z*256 : z*256+256],      # FP32 accumulate, v_dot2_f32_bf16
                       dequant(W[:, n, same slice]))
    partial_bf16 = bf16(partial_fp32)                # line 611-615: round to bf16
    atomic_add_pk4_bf16(&c[m][n], partial_bf16)      # CAS-loop, completion order = hardware
```

Each output element is therefore the sum of **26 separately-bf16-rounded values added in a nondeterministic completion order**, then rounded at every intermediate add (bf16 in/out of every CAS). Float addition is non-associative and the rounding happens 26 times per element in varying order — the hypothesized mechanism.

## The deterministic alternative (to be implemented)

```text
for split z in 0..25 in parallel:
    scratch[z][m][n] = partial_fp32[z][m][n]         # plain global store, no atomic, no cast

# separate kernel, one thread per (m, n):
acc_fp32 = 0.0f
for z = 0..25:                                       # compile-time fixed ascending order
    acc_fp32 += scratch[z][m][n]
c[m][n] = bf16(acc_fp32)                             # exactly one rounding at the end
```

Scratch for the primary shape: `Z × M × N × 4 B = 26 × 1 × 4096 × 4 = 426,004 B ≈ 416 KiB` (negligible). At M=128 it would be 53.1 MiB — acceptable for a prototype; production designs that avoid this are compared in Phase 10.

## What the diagnostic must show (Phase 4 targets)

1. `scratch[z]` partials themselves bit-repeatable for fixed input (if not, the nondeterminism begins *earlier* than the epilogue — major redirect).
2. Fixed-order FP32 reduction → bit-repeatable final output.
3. Different fixed orders (ascending / descending / one recorded shuffle) → different bf16 results, with the spread comparable to the spread the atomic path produces.

## M≥16 note

The bf16 WMMA path (M≥16) is a separate kernel (`q_gemm_rdna3_wmma.cu`) whose epilogue this map does not cover; the primary proof targets the scalar M=1 path used at decode. The M=128 microprobe arm showed only 2 distinct outputs/100 calls — consistent with either lower sensitivity at large M or a different epilogue in the WMMA kernel; out of scope for the primary proof.

## Gate

This is a source-level map only — per the operating rules it is NOT called root cause until the ON/OFF and split-partial experiments pass on real execution.
