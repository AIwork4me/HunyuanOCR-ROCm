# validation-50603 run environment — source me before run_state.sh.
# venv: hardlink clone of venv-torch212. torch is 2.12.0+rocm7.14.0 (NOT
# vLLM main's pinned 2.13): the 2.13 TheRock gfx1100 wheel ships no
# flash/efficient SDPA for gfx1100, so the transformers-backend ViT would
# materialize a 15k-token attention matrix (256 GiB OOM at profile). On 2.12
# the gfx11 AOTriton wheel gives flash SDPA (16k tokens, <1 GiB), matching
# the issue-era stacks. vLLM's CMake pins 2.13 but treats mismatch as
# warning-only; the extensions were rebuilt against 2.12 headers.
export PATH=/workspace/venv-vllm50603/bin:$PATH
export HF_HOME=/root/.cache/huggingface
# Model is pre-downloaded at the pinned revision (see env/model-download.log);
# offline mode pins what the reproducer's model="tencent/HunyuanOCR" resolves to.
export HF_HUB_OFFLINE=1
export HF_ENDPOINT=https://hf-mirror.com
export CUDA_VISIBLE_DEVICES=0
# AGENTS.md standard; identity no-op on a native gfx1100 (11.0.0 == 11.0.0).
export HSA_OVERRIDE_GFX_VERSION=11.0.0
# vLLM's ROCm platform probe requires the amdsmi python package; the TheRock
# SDK ships version-exact (7.14) bindings at share/amd_smi (its wrapper
# locates _rocm_sdk_core/lib/libamd_smi.so.26 relative to itself).
_SP=/workspace/venv-vllm50603/lib/python3.12/site-packages
export PYTHONPATH="$_SP/_rocm_sdk_core/share/amd_smi${PYTHONPATH:+:$PYTHONPATH}"
