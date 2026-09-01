# Phase 8 validation matrix — results

All results from real runs on W7900D (gfx1100), env-clean venv, vllm-clean worktree at 07ea9350 + upstream-final.patch. "distinct" = unique output hashes across repeats.

## A. Scalar repeatability (bf16, K=6656 N=4096, real weights)

100 calls per M: M=1 → 1 distinct; M=8 → 1 distinct. PASS.

## B. WMMA repeatability (bf16)

M=16 → 1/100; M=64 → 1/100; M=128 → 1/100; M=512 → 1/100. PASS.

## C. fp16

Scalar z=4 / z=26 shapes and WMMA M=64/128/512: all 1/100. PASS.

## D. Per-call interception determinism proof

16,640 GEMM calls per generation hashed (input + output); two generations compared call-by-call: 0 same-input-different-output events. PASS.

## E. Dual-stream sanity

Kernel outputs from two concurrent streams: bit-identical. PASS.

## F. CUDA-graph mode E2E

Muse, graphs, ctx512 → 1/8; ctx8192 → 1/8. PASS.

## G. Eager E2E

Muse ctx512 → 1/8 (both engines); gemma-3 ctx512/8192 → 1/8; Muse ctx8192 → 2/8 (first_div=27, stable across engines and reruns).

The Muse ctx8192 cell is root-caused to an upstream ROCm custom paged-attention defect that reads KV last-block stale slots (full chain: `muse-eager-8192-attention-rootcause.md`). It is independent of this patch: the same build with `attention_backend=TRITON_ATTN` yields 1/8 at both 512 and 8192, while every W4A16 GEMM call is proven bit-deterministic (D above). Documented, not fixed here.

## H. Regression tests

`pytest tests/kernels/quantization/test_rdna3_w4a16_determinism.py` → 8/8 PASS (bf16+fp16). Unpatched vLLM 0.25.1: 4/4 FAIL (fail-before established in the prior stage). PASS.

## I. Clean rebuild from patch alone

Fresh worktree at base commit 07ea9350 (`/tmp/vllm-phase7`), `git apply patches/upstream-final.patch`, rebuild via env-clean (`--no-build-isolation --no-deps --ignore-installed`). Smoke on that tree: regression tests 8/8 PASS (`logs/phase7-pytest.log`), two-phase probe 512=1 / 8192=2 — identical to the main worktree, i.e. the 8192 residual is the documented upstream attention issue, not a build artifact (`logs/phase7-two-phase.log`). PASS.
