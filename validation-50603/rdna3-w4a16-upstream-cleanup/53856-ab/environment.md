# Environment — #53856 A/B experiment (2026-09-01)

## Question

Does vLLM PR #53856 ("[Bugfix][ROCm] Mask paged attention V cache padding") eliminate the residual Muse eager ctx8192 nondeterminism that was root-caused to stale V-cache slots after `seq_len` in the final KV block?

## Upstream PR #53856 identity (fetched fresh from GitHub at experiment time)

- URL: https://github.com/vllm-project/vllm/pull/53856
- State: OPEN; updatedAt 2026-08-30T08:49:23Z
- Head SHA: 80e801cbfb7e6501f79c7ecd75aeb9b37ecf2561 (branch `fix/rocm-fp8-v-padding`)
- Base: main @ 796822d141382ab8ce82ef6101c6d802046f94e0
- Patch file: `pr53856.patch`, 373 lines, SHA256 `ba55bd3f9ded3d233cc06cc11be4a0da9ed65522ef8e0f8f01b73015177775c5`
- Files touched: `csrc/rocm/attention.cu`, `tests/kernels/attention/test_attention.py`
- Content: adds `mask_v_cache_padding` / `mask_16b_v_cache_boundary` device helpers and, in `paged_attention_ll4mi_QKV_mfma16_kernel`'s `KV_DTYPE == kAuto` (bf16/fp16) V-accumulation loop, skips V fetches whose token range starts at/after `seq_len` and byte-masks straddling 16B fetches so only bytes below `seq_len` survive; `__launch_bounds__` 5 → 4 for the same kernel.

## Trees (worktrees of the local vLLM clone; base identical)

Both worktrees at base `07ea9350baf84e33fd696d36fec9b9f24735a733` (the validated W4A16 base).

- `/tmp/vllm-53856-ab/baseline` = base + `patches/upstream-final.patch` (W4A16 only)
- `/tmp/vllm-53856-ab/candidate` = base + `upstream-final.patch` + `pr53856.patch`

### GATE verification — only difference is #53856

SHA256 across trees:

| file | baseline | candidate |
|---|---|---|
| csrc/rocm/q_gemm_rdna3.cu | f062d868…ed58af36 | f062d868…ed58af36 (identical) |
| csrc/rocm/q_gemm_rdna3_wmma.cu | 277f8509…3707cb0 | 277f8509…3707cb0 (identical) |
| tests/…/test_rdna3_w4a16_determinism.py | 3243b835…1fa6861a | 3243b835…1fa6861a (identical) |
| csrc/rocm/attention.cu | a2d0af86…8f886a7 | 6ed9f6ca…b645779 (**only A/B difference**) |

Plus candidate-only `tests/kernels/attention/test_attention.py` (part of #53856).

## Build / runtime

- venv: `/workspace/vllm-50603-upstream-cleanup/env-clean` (python 3.12, torch 2.12.0+rocm7.14.0, ROCm 7.14 wheel stack; `HSA_OVERRIDE_GFX_VERSION=11.0.0`)
- Build: `PYTORCH_ROCM_ARCH=gfx1100 VLLM_TARGET_DEVICE=rocm MAX_JOBS=48 env-clean/bin/pip install -e . --no-build-isolation --no-deps --ignore-installed` per tree (build logs `build-baseline.log` / `build-candidate.log`; candidate log shows `attention.cu` recompilation)
- Tree selection at runtime: `PYTHONPATH=/tmp/vllm-53856-ab/<tree>` (prepended; verified `import vllm` resolves to the selected tree in every probe — each probe prints `import from: …`)
- GPU: AMD W7900D, gfx1100 (1× of 8), 48 GB; `CUDA_VISIBLE_DEVICES=0`, `AB_MM_LIMIT_ZERO=1`, `HF_HUB_OFFLINE=1`
- Model: `/workspace/vllm-50603-version-ab/models/muse` (RedHatAI/Muse-Glimmer-30B-INT4 @ f5b410ce)
- Reproducer: `probe_two_phase.py` (identical protocol to the validated cleanup stage)

## Provenance of the W4A16 patch

`patches/upstream-final.patch` = the validated cleanup-stage patch (878 lines, 3 files), identical to the one on evidence branch `validation-50603-rdna3-upstream-cleanup` @ a0277ab and to pushed branch `AIwork4me/vllm fix/rdna3-w4a16-determinism` @ bd5d05816b (not modified in this task).
