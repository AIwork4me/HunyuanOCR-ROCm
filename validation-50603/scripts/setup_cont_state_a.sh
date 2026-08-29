#!/usr/bin/env bash
# Continuation of setup_and_build_state_a.sh (torch 2.13 stack already in
# place and GPU-sanitied): runtime deps -> model download -> State A build.
set -ex

export HF_HOME=/root/.cache/huggingface
export HF_ENDPOINT=https://hf-mirror.com
export VLLM_PYTHON_EXECUTABLE=/workspace/venv-vllm50603/bin/python
VPY=/workspace/venv-vllm50603/bin/python
PIP=/workspace/venv-vllm50603/bin/pip
PIPFLAGS="--retries 10 --timeout 60"
# pypi.org is intermittently unreachable from this box; aliyun mirror is up
# (probed 2026-08-29) and used as the primary index for PyPI-hosted wheels.
PYPIDX="https://mirrors.aliyun.com/pypi/simple/"

cd /workspace/vllm-50603
git checkout -q state-a

echo "=== stage 3 (cont): vLLM runtime deps + build tools ==="
$PIP install $PIPFLAGS -i "$PYPIDX" -r requirements/common.txt
$PIP install $PIPFLAGS -i "$PYPIDX" "setuptools>=77.0.3,<81.0.0" setuptools-rust "cmake>=3.26.1" ninja "setuptools-scm>=8.0" wheel jinja2 "packaging>=24.2"

echo "=== stage 4 (cont): model download (tencent/HunyuanOCR @ de8f10ad) ==="
dl() {
  if [[ -x /workspace/venv-vllm50603/bin/hf ]]; then
    /workspace/venv-vllm50603/bin/hf download tencent/HunyuanOCR \
      --revision de8f10ad2f00a0cefd790b526de8a65dcfdb3205
  else
    $VPY -c "from huggingface_hub import snapshot_download; snapshot_download('tencent/HunyuanOCR', revision='de8f10ad2f00a0cefd790b526de8a65dcfdb3205')"
  fi
}
for i in 1 2 3; do
  if dl > /workspace/HunyuanOCR-ROCm/validation-50603/env/model-download.log 2>&1; then
    echo "model download ok (attempt $i)"; break
  fi
  echo "model download attempt $i failed; retrying"; sleep 30
  [[ $i == 3 ]] && exit 1
done

echo "=== stage 5 (cont): editable build of vLLM state-a ==="
PYTORCH_ROCM_ARCH=gfx1100 VLLM_TARGET_DEVICE=rocm MAX_JOBS=96 \
  $PIP install -e . --no-build-isolation --no-deps

echo "=== stage 6 (cont): vLLM import sanity ==="
$VPY -c "import vllm; print('vllm', vllm.__version__, 'at', vllm.__file__)"
echo "=== ALL DONE ==="
