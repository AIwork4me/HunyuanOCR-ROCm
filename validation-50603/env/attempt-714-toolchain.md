# Attempt: building vLLM 0.25.1 with a fully-7.14 toolchain (SDK clang 23)

Goal: eliminate the issue reporter's build caveat (system 7.2.1 headers driving a
7.14-runtime-torch build) by compiling entirely with the 7.14 TheRock SDK toolchain.

Result: **blocked inside vLLM 0.25.1's own sources** — configure completes, but the
RDNA3 quantized-GEMM files do not compile under clang 23 (7.14 SDK):

- `q_gemm_rdna3_wmma.hip`, `skinny_gemms.hip`, `qdq_4_rdna3.cuh` — `__fmaf_rn` undeclared
  (clang 23 declares the HIP math intrinsics only after `<cmath>`; a `-include` compat
  prelude fixed these), then `fmaf` overload resolution failures remained in
  `moe_q_gemm_rdna3.hip` / `qdq_4_rdna3.cuh`, and torch-layer `fminf` failures in
  `csrc/libtorch_stable/activation_kernels.hip` — implicit bf16/half conversion and
  overload behavior changed with the newer toolchain, and these AMD-contributed files
  predate it.

Full log: `attempt-714-toolchain-build.log`. The shim root assembled for the attempt
(`/workspace/rocm-714-sdk-root`: FindHIP/hip-config/hip-lang from the 7.2.1 install
re-pointed at the 7.14 SDK libs via symlinks, plus minimal hipcc/hipconfig shims) is
kept for a future retry against a vLLM version that compiles under clang 23.

Consequence for the validation: the canonical builds use the system 7.2.1 hipcc
against the 7.14-runtime torch — the same build configuration the issue reporter
documented. The toolchain is identical across all three A/B states, so the
state-to-state attribution is unaffected; the runtime stack (torch 2.12.0+rocm7.14.0,
triton 3.7.1+git rocm7.14, ROCm 7.14 user-space libraries) is the community 7.14 wheel
stack in all runs.
