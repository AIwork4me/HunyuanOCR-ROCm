# RDNA3 W4A16 Root-Cause Proof

## Finding

The gfx1100 greedy-decoding nondeterminism of vllm#50603 is caused by the split-K epilogue of `gptq_gemm_rdna3` (`RDNA3W4A16LinearKernel`): each of the Z=K/256 split-K blocks casts its FP32 partial to bf16 and CAS-atomically accumulates it directly into the bf16 output, so the low-precision reduction order — and therefore the result — varies call to call; replacing that epilogue with FP32 split scratch + fixed-order reduction makes both the isolated kernel and the full Muse greedy reproducer bit-reproducible.

## Causal chain (every link measured on the W7900D)

```text
RDNA3W4A16LinearKernel ON (atomic epilogue)      -> Muse/512 greedy: 6/8, 8/8, 8/8 unique
kernel OFF (VLLM_DISABLED_KERNELS, Triton verif) -> 1/8, 1/8, 1/8
isolated gptq_gemm_rdna3, real q_proj, M=1       -> 200/200 distinct outputs (3 procs), first-call sha differs per process
per-split FP32 partials (diagnostic op)          -> 50 calls -> 1 distinct (deterministic)
fixed-order FP32 reduction (asc/desc/shuffled)   -> three orders bitwise IDENTICAL
atomic outputs vs deterministic reduction        -> spread ±0.035, same scale as atomic-vs-atomic (0.031)
deterministic op, M=1, 200 calls x 3 procs       -> 1/200 distinct each, SAME sha across processes
Z-sweep on the UNMODIFIED production op          -> Z=1: 1/100, Z=2: 1/100, Z>=4: 100/100 distinct
deterministic prototype in-engine (all M tiled)  -> Muse greedy: 1/8, 1/8, 1/8 (both depths)
```

## Mechanism

`csrc/rocm/q_gemm_rdna3.cu` @ `752a3a5044` (see `environment/kernel-reduction-map.md`): the caller passes a zero-initialized bf16 output; `grid.z = K/256` split-K blocks each compute an FP32 partial (`v_dot2_f32_bf16` chains), cast the four per-thread partials to bf16 (`__float2bfloat16`, lines 610-616), and accumulate via a 64-bit CAS loop packing 4 bf16 lanes (`atomic_add_pk4_bf16`). For Muse q_proj (K=6656) that is Z=26 concurrent bf16 additions per output element in hardware completion order. Float addition is non-associative and every intermediate add rounds at bf16's 8-bit mantissa, so the final value varies run to run (measured spread ±0.035 logit-scale units on the real layer; E2E this becomes the 2.1–4.1-unit logit divergence measured in the forensics phase, amplified by 52 decoder layers × several W4A16 linears each).

Precise mechanism boundary: per-split FP32 partials are deterministic, and the order of *FP32* summation is irrelevant at this Z (ascending/descending/shuffled reductions are bitwise identical — FP32's 24-bit mantissa absorbs the reorder). The nondeterminism enters exclusively through the *cast-then-atomically-add* chain: rounding to bf16 before every accumulation, in unordered concurrent additions. This is why the deterministic prototype (accumulate FP32, round once) is both reproducible and more accurate.

Direct confirmation on the unmodified production op: the nondeterminism switches on between Z=2 and Z=4 (K sweep at fixed N=4096: Z=1→1/100 distinct, Z=2→1/100, Z=4→100/100, Z=8→100/100, Z=26→100/100) — single-writer (Z=1) and two-writer cases are stable, concurrent multi-writer accumulation is not.

## ON/OFF result (both kernels verified from runtime logs)

| arm | selected kernel (log line) | engine | unique/8 @512 | unique/8 @8192 |
|---|---|---:|---:|---:|
| ON | `Using RDNA3W4A16LinearKernel` | 1 | **6** | 2 |
| ON | 〃 | 2 | **8** | 3 |
| ON | 〃 | 3 | **8** | 2 |
| OFF | `Using TritonW4A16LinearKernel` | 1 | 1 | 1 |
| OFF | 〃 | 2 | 1 | 1 |
| OFF | 〃 | 3 | 1 | 1 |

ON-arm re-verified after all kernel work with the restored pristine environment: 8/8 (engine 9, `environment/on-verify-restored-env-eng9.json`).

## Split partial experiment (`microprobe/phase4-partials.jsonl`)

Real layer-0 q_proj (K=6656, N=4096, group 128), fixed seeded bf16 input, Z=26:

- partials op, 50 calls: **1 distinct** [Z,1,N] tensor (sha repeatable)
- R1 (z ascending) / R2 (descending) / R3 (recorded shuffle, seed 7): **all three bf16 outputs share one sha** (`c50b8a2b…`)
- 30 production atomic calls: 30 distinct; |atomic − R1| max 0.0352, median 0.0234 — same magnitude as atomic-vs-atomic (max 0.0313): the atomic outputs scatter around the deterministic reduction exactly as unordered bf16 rounding predicts.

## Prototype

`gptq_gemm_rdna3_deterministic` (in `patches/`): pass 1 runs the unmodified split-K compute writing FP32 partials to a scratch buffer (no atomics, no casts); pass 2 reduces in fixed ascending-z order in FP32 and rounds to bf16 once. Row-tiled at 64 rows so the scratch stays ≤ ~118 MiB for the worst Muse layer at any M (a first un-tiled version OOM'd the engine's M=8192 profiling forward at 31.7 GiB; the serving path also re-runs prefill-tail forwards at M>8 during generation — verified by an M-logger: 7 M=512 forwards + M=16/128/8192 calls — so the prototype routes ALL M through the deterministic path). Python dispatch is env-gated: `VLLM_RDNA3_W4A16_DETERMINISTIC=1` inside `RDNA3W4A16LinearKernel.apply_weights`, logged once at first use.

## Microprobe (`microprobe/deterministic-microprobe.jsonl`)

| op | proc A (200 calls) | proc B | proc C |
|---|---|---|---|
| atomic | 200 distinct | 200 | 200 (first-call sha differs per process) |
| **deterministic** | **1 distinct** | **1** | **1 — same sha `c50b8a2b…` in all three processes** |

## End-to-end (Muse/TP=1/eager, warm-up ×1 + greedy ×8, 3 engines each)

| state | engine 1 | engine 2 | engine 3 |
|---|---:|---:|---:|
| current RDNA3 (atomic) | 6/8 | 8/8 | 8/8 |
| kernel disabled (Triton, verified) | 1/8 | 1/8 | 1/8 |
| **deterministic prototype** | **1/8** | **1/8** | **1/8** |

At ctx=8192 likewise: atomic 2–3/8, prototype 1/8 ×3. First divergence in prototype engines: none (∅ across all 7 runs in every engine).

## Correctness (`correctness/correctness.jsonl`, fp32 dequant-reference matmul, real weights)

| implementation | deterministic? | max abs err | mean abs err | cosine |
|---|---|---:|---:|---:|
| current RDNA3 (atomic) | no | 0.0261 / 0.0248 (two runs) | 0.00232 / 0.00238 | 0.999981 |
| deterministic prototype | **yes** | **0.0061** | **0.00059** | **0.999999** |

The prototype is ~4× closer to the fp32 reference — expected, since it accumulates split-K partials in FP32 and rounds once instead of 26 bf16 roundings.

## Performance (`performance/perf-microbench.jsonl`, CUDA events, 50 warmup + 200 timed, real q_proj)

| path | M=1 median | M=512 median | deterministic |
|---|---:|---:|---|
| current RDNA3 (atomic) | 43.0 µs | 0.532 ms | no |
| deterministic prototype | 45.9 µs (+6.6%) | 2.107 ms (3.96×) | **yes** |

End-to-end engine wall time (both depths, eager, full harness): atomic ≈115.4 s, Triton fallback ≈188.3 s (+63%), deterministic prototype ≈171.5 s (+49% vs atomic, −9% vs the only currently-deterministic alternative). The M>8 cost comes from the prototype routing prefill-shaped calls through the tiled scalar path instead of the WMMA kernel — a production design would keep large-M on a separately-validated deterministic epilogue instead.

## Fix-design comparison (Phase 10, analysis)

| option | determinism | numerics | throughput | memory |
|---|---|---|---|---|
| 1. FP32 scratch + fixed reduce (implemented) | bitwise | best (fp32 acc, one rounding) | +6.6% @M=1; ~4× @M=512 (prot. routes all M here) | ≤118 MiB tiled scratch |
| 2. avoid split-K at M=1 (single z) | bitwise | same as fp32 acc | Z× fewer blocks per output → CU occupancy risk at decode; likely the fastest *if* occupancy holds | none |
| 3. deterministic tree reduction | bitwise | fp32 acc | same pass count as 1 | same scratch |
| 4. route M=1 to TritonW4A16 | bitwise (measured 1/8) | unchanged (Triton) | E2E +63% (measured) | none |
| 5. FP32 atomics | **not bitwise** — fp32 add is still non-associative and atomic order still varies; only the last bits improve | good | near-native | none |

Recommended upstream direction: option 1 for M≤8 decode calls (measured +6.6%) combined with a separately-validated deterministic epilogue for the WMMA path (M≥16); option 2 is worth benchmarking as the zero-scratch decode variant.

## Regression test

`tests/kernels/quantization/test_rdna3_w4a16_determinism.py` (in the patches): K=1024/N=4096 (Z=4, above the measured Z=2→4 onset), 20 repeats: deterministic op asserts `torch.equal` (PASSED); legacy atomic op documented as `xfail` (XFAILED — i.e., fails repeatability as expected on gfx1100).

## Limitations

- Only gfx1100 (W7900D) tested; the kernel is gfx11-only by construction but other RDNA3 parts are unverified.
- Primary workload is Muse-Glimmer-30B-INT4 (plus the synthetic-K Z-sweep); other W4A16 models inherit the same op but were not individually rerun here.
- The prototype does not make the M≥16 WMMA epilogue deterministic (it bypasses WMMA); prefill-heavy workloads pay ~4× on W4A16 calls for large M.
- The engine-serving discovery (prefill-tail recomputation during generation, M>8) came from an env-gated M-logger (`RDNA3_DEBUG_M_LOG`, included only in `diagnostic.patch`); the logger is excluded from the upstream candidate.
- Environment note: the dev venv (`env-rdna3`) was created as a hardlink clone of the validated `env-0.25.1`; the first two builds (before the pip-shebang was repointed) reinstalled vLLM's editable metadata into the pristine venv. It was restored and re-verified (8/8 ON-arm reproduction, engine 9); all ON/OFF arm data shown was collected before the first modified build.
- fp16 (half) activations use a different inner-dot path but the same atomic epilogue; only bf16 was exercised (all Muse linears are bf16).

## Reproduce

```bash
# environments: see ../version-topology-ab for the build recipe (identical stack)
# this experiment adds: /workspace/vllm-50603-rdna3-rootcause/vllm-rdna3
#   = vLLM 0.25.1 worktree + patches/deterministic-prototype.patch
scripts/run_phase2.sh   # ON-arm baseline (atomic) — microprobe + E2E
scripts/run_phase3.sh   # OFF arm (VLLM_DISABLED_KERNELS=RDNA3W4A16LinearKernel)
VLLM_RDNA3_W4A16_DETERMINISTIC=1 scripts/run_phase8c.sh  # prototype arm
python3 scripts/probe_rdna3.py partials|probe|correctness <tag> <n>
python3 scripts/probe_zsweep.py <tag> <n>    # Z-sweep on the unmodified op
python3 scripts/bench_perf.py                # latency microbench
```

Raw evidence: `kernel-on-off/`, `prototype/`, `microprobe/`, `correctness/`, `performance/`, `logs/` (driver + per-engine logs incl. routing lines), patches in `patches/`, checksums in `SHA256SUMS`.
