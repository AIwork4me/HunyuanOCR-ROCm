#!/usr/bin/env bash
# validation-50603: one-shot environment setup + State A (baseline) build.
# Chained so each stage gates the next: torch 2.13 upgrade -> runtime deps ->
# model download (pinned revision) -> GPU sanity -> editable build of vLLM
# state-a. Log: validation-50603/env/build-state-a.log
set -ex

export HF_HOME=/root/.cache/huggingface
export HF_ENDPOINT=https://hf-mirror.com
export VLLM_PYTHON_EXECUTABLE=/workspace/venv-vllm50603/bin/python
VPY=/workspace/venv-vllm50603/bin/python
PIP=/workspace/venv-vllm50603/bin/pip
AMD_IDX=https://repo.amd.com/rocm/whl-multi-arch/

cd /workspace/vllm-50603
git checkout -q state-a

echo "=== stage 1: torch 2.13.0+rocm7.14.0 (leaf gfx1100 2.13; family gfx11 wheel pins torch==2.12 and is intentionally omitted — this workload routes through Triton/CK attention, not aten-SDPA fused backends; pytorch#194498 context) ==="
$PIP install --index-url "$AMD_IDX" \
  "torch==2.13.0+rocm7.14.0" "torchvision==0.28.0+rocm7.14.0" \
  "amd-torch-device-gfx1100==2.13.0+rocm7.14.0"

echo "=== stage 2: GPU sanity on the new torch ==="
$VPY - <<'PY'
import torch
assert torch.cuda.is_available(), "CUDA/HIP not available"
print("torch", torch.__version__, "hip", torch.version.hip)
print("device", torch.cuda.get_device_name(0))
x = torch.randn(512, 512, device="cuda", dtype=torch.bfloat16)
y = (x @ x).float().mean()
torch.cuda.synchronize()
print("matmul mean ok:", float(y))
q = torch.randn(1, 8, 128, device="cuda", dtype=torch.bfloat16)
k = torch.randn(1, 4, 128, device="cuda", dtype=torch.bfloat16)
v = torch.randn(1, 4, 128, device="cuda", dtype=torch.bfloat16)
o = torch.nn.functional.scaled_dot_product_attention(q, k, v)
print("sdpa ok:", tuple(o.shape))
PY

echo "=== stage 3: vLLM runtime deps (requirements/common.txt at state-a) + build tools ==="
$PIP install "setuptools>=77.0.3,<81.0.0" setuptools-rust "cmake>=3.26.1" ninja "setuptools-scm>=8.0" wheel jinja2 "packaging>=24.2"
$PIP install -r requirements/common.txt

echo "=== stage 4: model download (tencent/HunyuanOCR @ de8f10ad) ==="
$VPY -m huggingface_hub.commands.huggingface_cli download tencent/HunyuanOCR \
  --revision de8f10ad2f00a0cefd790b526de8a65dcfdb3205 \
  > /workspace/HunyuanOCR-ROCm/validation-50603/env/model-download.log 2>&1 || \
$VPY -c "from huggingface_hub import snapshot_download; snapshot_download('tencent/HunyuanOCR', revision='de8f10ad2f00a0cefd790b526de8a65dcfdb3205')"

echo "=== stage 5: editable build of vLLM state-a ==="
PYTORCH_ROCM_ARCH=gfx1100 VLLM_TARGET_DEVICE=rocm MAX_JOBS=96 \
  $PIP install -e . --no-build-isolation --no-deps

echo "=== stage 6: vLLM import sanity ==="
$VPY -c "import vllm; print('vllm', vllm.__version__, 'at', vllm.__file__)"
echo "=== ALL DONE ==="
