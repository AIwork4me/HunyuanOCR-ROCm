#!/usr/bin/env bash
# validation-53615 STATE A build: pre-#53272/#53615 native HunyuanVL vLLM main.
# STATE A revision = 27ec8ac626 (parent of d53b1c2efc "Remove native Hunyuan
# V1 and VL implementations (#53272)", 2026-08-21).
# Worktree from /workspace/vllm-mainline-probe + hardlink clone of
# venv-vllm50603 (torch 2.12.0+rocm7.14.0, transformers 5.10.2 — 5.11+ breaks
# the native hunyuan_vl_image.py processor import), rebuilt with the system
# ROCm 7.2.1 hipcc toolchain per validation-50603 precedent.
set -euo pipefail

VREPO="${VLLM_REPO:-/workspace/vllm-mainline-probe}"
STATEA_REV="${STATEA_REV:-27ec8ac626}"
WT="${STATEA_WORKTREE:-/workspace/vllm-53615-statea}"
VENV="${STATEA_VENV:-/workspace/venv-53615-statea}"

cd "$VREPO"
git worktree remove "$WT" --force 2>/dev/null || true
git worktree add "$WT" "$STATEA_REV"

rm -rf "$VENV"
cp -al /workspace/venv-vllm50603 "$VENV"

cd "$WT"
echo "=== state-a revision ==="
git rev-parse HEAD
git log -1 --oneline
echo "=== transformers pin at state-a ==="
grep -i transformers requirements/common.txt | head -2 || true
"$VENV/bin/python" -c "import transformers, torch; print('venv transformers', transformers.__version__, 'torch', torch.__version__)"

ROCM_PATH=/opt/rocm HIP_PATH=/opt/rocm CMAKE_PREFIX_PATH=/opt/rocm \
PYTORCH_ROCM_ARCH=gfx1100 VLLM_TARGET_DEVICE=rocm MAX_JOBS=96 \
  "$VENV/bin/python" -m pip install -e . --no-build-isolation --no-deps
# NOTE: must be `python -m pip`, NOT bin/pip: the hardlink-cloned venv's pip
# entrypoint keeps the source venv's shebang, and bin/pip would install into
# the SOURCE venv (this happened once; venv-vllm50603 was restored with
# `python -m pip install -e /workspace/vllm-50603`).

"$VENV/bin/python" -c "import vllm; print('state-a built:', vllm.__version__, vllm.__file__)"
