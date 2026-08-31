#!/usr/bin/env bash
# Build one isolated environment for the version A/B.
# Usage: build_env.sh <env-name> <worktree-path>
# Replicates the prior validation-50603 canonical build: AMD rocm7.14 wheel
# torch stack + system 7.2.1 hipcc compiling vLLM (PYTORCH_ROCM_ARCH=gfx1100).
set -ex

NAME=$1
WT=$2
ROOT=/workspace/vllm-50603-version-ab
VENV=$ROOT/environments/$NAME
PIP=$VENV/bin/pip
PY=$VENV/bin/python
AMD_IDX=https://repo.amd.com/rocm/whl-multi-arch/

# pypi.org/files.pythonhosted.org are unreachable from this box; use the
# Tsinghua PyPI mirror for everything not served by the AMD ROCm index.
export PIP_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple

python3 -m venv "$VENV"
$PIP install --upgrade pip -q

echo "=== stage 1: torch stack (identical wheels in both arms) ==="
$PIP install --index-url "$AMD_IDX" \
  "torch==2.12.0+rocm7.14.0" "torchvision==0.27.0+rocm7.14.0" \
  "amd-torch-device-gfx1100==2.12.0+rocm7.14.0"

echo "=== stage 2: GPU sanity ==="
$PY - <<'PYEOF'
import torch
assert torch.cuda.is_available(), "CUDA/HIP not available"
print("torch", torch.__version__, "hip", torch.version.hip)
print("device", torch.cuda.get_device_name(0), torch.cuda.get_device_capability(0))
x = torch.randn(512, 512, device="cuda", dtype=torch.bfloat16)
torch.cuda.synchronize()
print("matmul ok:", float((x @ x).float().mean()))
PYEOF

echo "=== stage 3: build tools + vLLM runtime deps at pinned commit ==="
$PIP install "setuptools>=77.0.3,<81.0.0" setuptools-rust "cmake>=3.26.1" ninja "setuptools-scm>=8.0" wheel jinja2 "packaging>=24.2"
cd "$WT"
$PIP install -r requirements/common.txt
# transformers 5.15.1 in BOTH arms: first release line with muse_glimmer
# (added 2026-08-10, PR #47867); satisfies both versions' requirement pins
# (0.23.1.dev1: >=4.56,!=5.0-5.5; 0.25.1: >=5.5.3). Pinned, not resolved,
# so the transformers layer is identical across the A/B.
$PIP install "transformers==5.15.1"

echo "=== stage 4: editable build (no build isolation; pyproject pins torch==2.11 which we deliberately override with the common 2.12 stack) ==="
cd "$WT"
PYTORCH_ROCM_ARCH=gfx1100 VLLM_TARGET_DEVICE=rocm MAX_JOBS=96 \
  $PIP install -e . --no-build-isolation --no-deps

echo "=== stage 5: import sanity ==="
$PY -c "import vllm, torch; print('vllm', vllm.__version__, '| torch', torch.__version__, '| at', vllm.__file__)"
echo "=== BUILD OK $NAME $(date -Is) ==="
