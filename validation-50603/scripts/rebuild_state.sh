#!/usr/bin/env bash
# usage: rebuild_state.sh <vllm-branch-or-sha>
# Rebuilds the compiled extensions after switching vLLM code state
# (needed for state-b: PR #53856 changes csrc/rocm/attention.cu).
# Incremental: only changed translation units recompile.
set -ex
BRANCH="$1"
VREPO=/workspace/vllm-50603
source /workspace/HunyuanOCR-ROCm/validation-50603/env/env.sh

cd "$VREPO"
git checkout -q --force "$BRANCH"
# Toolchain pinned to the system ROCm 7.2.1 (the only install shipping CMake
# configs; TheRock python SDK has none). This matches the issue reporter's
# documented build configuration: 7.2.1 headers/hipcc + 7.14 torch runtime.
ROCM_PATH=/opt/rocm HIP_PATH=/opt/rocm CMAKE_PREFIX_PATH=/opt/rocm \
PYTORCH_ROCM_ARCH=gfx1100 VLLM_TARGET_DEVICE=rocm MAX_JOBS=96 \
  pip install -e . --no-build-isolation --no-deps
python -c "import vllm; print('rebuilt:', vllm.__version__, vllm.__file__)"
