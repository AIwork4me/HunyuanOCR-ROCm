# validation-53615 run environment — source me before any run.
# Mirrors validation-50603/env/env.sh conventions, pointed at the mainline
# probe venv (editable vLLM install from /workspace/vllm-mainline-probe,
# branch investigate/hunyuanocr-transformers-regression).
# torch stays 2.12.0+rocm7.14.0 (not main's pinned 2.13): the 2.13 TheRock
# gfx1100 wheel ships no flash/efficient SDPA for gfx1100, so the
# transformers-backend ViT would materialize huge attention matrices and OOM
# at profile; on 2.12 the gfx11 AOTriton wheel gives flash SDPA.
export PATH=/workspace/venv-53615-statea/bin:$PATH
export HF_HOME=/root/.cache/huggingface
export HF_HUB_OFFLINE=1
export HF_ENDPOINT=https://hf-mirror.com
export HIP_VISIBLE_DEVICES=0
# AGENTS.md standard; identity no-op on a native gfx1100 (11.0.0 == 11.0.0).
export HSA_OVERRIDE_GFX_VERSION=11.0.0
# vLLM's ROCm platform probe requires the amdsmi python package; the TheRock
# SDK ships version-exact (7.14) bindings at share/amd_smi (its wrapper
# locates _rocm_sdk_core/lib/libamd_smi.so.26 relative to itself).
_SP=/workspace/venv-53615-statea/lib/python3.12/site-packages
export PYTHONPATH="$_SP/_rocm_sdk_core/share/amd_smi${PYTHONPATH:+:$PYTHONPATH}"
