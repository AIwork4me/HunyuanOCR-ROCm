# REPRODUCE — validation-53615

Everything below was executed on one AMD Radeon PRO W7900D (gfx1100, 48 GB) with the ROCm 7.14 wheel stack. TP=1 only; TP>1/PP>1 not tested. An independent engineer needs: a gfx1100 GPU (any single ROCm 6.4+/7.x GPU with a working vLLM build should show the same host-side failures for STATE B — all three are shape/kwargs errors raised before any device-specific kernel runs — but the PASS states were only validated on gfx1100), the HunyuanOCR weights, and a vLLM dev environment.

## 0. Common setup

```bash
# vLLM source (never pushed anywhere):
git clone https://github.com/vllm-project/vllm.git /workspace/vllm-mainline-probe
cd /workspace/vllm-mainline-probe
# The validated SHAs:
#   original base:   b5707bf994cb968adfc7a29fbb80b0522f53f38d
#   latest main:     1dc464d42681d22f38caf1fdc1eb632dc4421c45
git checkout 1dc464d426

# Model: pin the exact revision used in every log in this package
export MODEL_PATH=/root/.cache/huggingface/hub/models--tencent--HunyuanOCR/snapshots/de8f10ad2f00a0cefd790b526de8a65dcfdb3205
export MODEL_REVISION=de8f10ad2f00a0cefd790b526de8a65dcfdb3205
export HF_HUB_OFFLINE=1            # keeps the bare model id pinned to the cached revision
export HF_HOME=/root/.cache/huggingface
export HIP_VISIBLE_DEVICES=0       # (CUDA_VISIBLE_DEVICES is deprecated on ROCm in vLLM ≥0.26)
export HSA_OVERRIDE_GFX_VERSION=11.0.0  # identity no-op on native gfx1100

# vLLM ROCm platform probe needs the amdsmi python package. With the TheRock
# 7.14 wheel stack it ships inside the venv:
export PYTHONPATH="$PYTHONPATH:$(python -c 'import _rocm_sdk_core, os; print(os.path.join(os.path.dirname(_rocm_sdk_core.__file__), "share", "amd_smi"))')"
# (this host-specific shim mirrors env/env.sh; if your venv already imports
#  amdsmi, skip it)

# Python stack used for all PASS evidence: torch 2.12.0+rocm7.14.0,
# transformers 5.15.1 (5.13.0 additionally studied — see SUMMARY.md Part 1),
# python 3.12. torch 2.12 specifically: the 2.13 gfx1100 wheel has no
# flash/efficient SDPA for gfx11, which OOMs the vision tower at profile time.
```

Workload: `scripts/repro_baseline.py` (synthetic 1024×960 financial-summary image, Chinese OCR extraction prompt, greedy, `max_model_len=8192`, `enforce_eager=True`, `limit_mm_per_prompt={"image": 1}`). Every script prints `[env]` lines with the exact vLLM git SHA (resolved from the imported `vllm.__file__`) and package versions — check them against the SHAs above. Override the model with `MODEL_ID`/`MODEL_PATH` env vars if your layout differs.

## STATE B — pristine vLLM current-main baseline (expected FAIL)

```bash
cd /workspace/vllm-mainline-probe
git status --short          # must be empty (pristine)
git rev-parse HEAD          # 1dc464d42681d22f38caf1fdc1eb632dc4421c45

python /workspace/HunyuanOCR-ROCm/validation-53615/scripts/repro_baseline.py
```

Expected: exit code 1, engine-core init fails during the profile/dummy run with

```
ValueError: Expected 4 multimodal RoPE channels, got position_ids with shape (3, 1, 8192).
```

raised inside `transformers/models/hunyuan_vl/modeling_hunyuan_vl.py` (forward gate `len(self.rotary_emb.mrope_section) == 4`). Reference log: `latest-main/run-baseline-latest-main.log` (latest main) / `baseline/run-baseline.log` (original base — identical failure).

Also expected on pristine main (focused tests, no GPU needed):

```bash
pip install pytest tblib   # test deps; vLLM's tests/conftest.py imports tblib
python -m pytest tests/v1/worker/test_rope_state.py \
    tests/models/transformers/test_create_attention_instances.py \
    tests/models/transformers/test_get_rope_index_kwargs.py -v
# 3 FAILED (the regression tests: assert 3 == 4; assert 192 == 128;
#           TypeError: unexpected keyword argument 'video_grid_thw')
# 3 PASSED (the backward-compat guards)
```

Reference log: `latest-main/tests-all-before-fix.log`.

## STATE C — current main + candidate patch (expected PASS)

```bash
cd /workspace/vllm-mainline-probe
git apply /workspace/HunyuanOCR-ROCm/validation-53615/diff/hunyuanocr-vllm-candidate.patch
git rev-parse HEAD          # still 1dc464d426…; the patch applies on top, uncommitted

python /workspace/HunyuanOCR-ROCm/validation-53615/scripts/repro_baseline.py
```

Expected: exit code 0 and faithful markdown OCR of the synthetic page:

```
# Quarterly Financial Summary
Revenue: 12,480,000 USD
Cost of Goods Sold: 4,210,000 USD
Gross Profit: 8,270,000 USD
```

Focus tests flip to 6/6 PASS (`latest-main/tests-focused-and-mrope-after-fix.log`).

Full E2E matrix (4 image tests incl. ×3 repeatability + alternate aspect; three modes):

```bash
V=validation-53615   # from the repo root
export VLLM_ENABLE_V1_MULTIPROCESSING=0
python $V/scripts/e2e_suite.py /tmp/e2e-eager.json                                   # enforce_eager
V53615_EAGER=0 V53615_CUDAGRAPHS=piecewise python $V/scripts/e2e_suite.py /tmp/e2e-piecewise.json
VLLM_USE_V2_MODEL_RUNNER=0 python $V/scripts/e2e_suite.py /tmp/e2e-v1.json           # legacy V1 runner
```

Expected: `"all_ok": true` in each JSON, document token hash `6cd5fba4cdb2f135` in all three modes (cross-mode determinism). Reference: `latest-main/e2e-*.json` + `latest-main/run-e2e-*.log`. Note: FULL cudagraph mode is NOT expected to pass on ROCm (pre-existing HF-rotary capture limitation — use piecewise); do not read its failure as a regression.

HF-only control (isolates vLLM plumbing from the HF model):

```bash
python $V/scripts/repro_hf_control.py     # plain transformers, sdpa, greedy
# transformers 5.15.1: same faithful markdown as vLLM STATE C
# transformers 5.13.0: garbled — and vLLM STATE C is token-identical to it
```

Root-cause probes (no GPU generation needed):

```bash
python $V/scripts/probe_boundary_a.py     # HF get_rope_index returns (4, 1, 1038)
VLLM_ENABLE_V1_MULTIPROCESSING=0 python $V/scripts/repro_instrumented.py
# monkeypatch-based instrumentation (no production-file edits): RopeState
# num_dims, buffer shapes, rotary-received position_ids (4, 1, N)
```

## Native historical reference (optional; not required for STATE B/C)

Historical evidence only — two datapoints, already captured in `state-a/`:

- Pre-removal native main (`27ec8ac626`, parent of #53272): BUILT successfully (`scripts/build_state_a.sh`; worktree + hardlink venv clone, ROCm 7.2.1 hipcc toolchain, `PYTORCH_ROCM_ARCH=gfx1100 VLLM_TARGET_DEVICE=rocm`) but FAILS at weight load for this checkpoint (`no module or parameter named 'lm_head'`) — a pre-existing native-path bug, not an environment issue. Log: `state-a/run-state-a.log`. Requires transformers ≥5.13 (that native code imports `HunYuanVLProcessor`).
- Proven-working native reference: vLLM v0.25.1 (+ validation-50603 backports) runs the same workload with faithful OCR — log `state-a/run-native-v0251-statec.log`. Rebuilding that stack is documented in `validation-50603/`; treat it as background, not part of this package's reproduction path.

## Host-specific caveats (documented, not hidden)

- `env/env.sh` / `env/env-state-a.sh` are this host's activation scripts (PATH to the two venvs, amdsmi `PYTHONPATH` shim, HIP visibility); the block above reproduces their content generically.
- `scripts/build_state_a.sh` defaults to this host's paths (`VLLM_REPO`, `STATEA_REV`, `STATEA_WORKTREE`, `STATEA_VENV` overridable) and contains the `python -m pip` note explaining the hardlink-venv shebang trap hit during the original work.
- This host has no pypi.org/huggingface.co access: pip via the Tsinghua mirror; hub-dependent vLLM tests (kernels test_mrope.py parametrizations, test_backend.py weights) cannot run here and their failures were proven identical on pristine main (see SUMMARY.md test matrices).
