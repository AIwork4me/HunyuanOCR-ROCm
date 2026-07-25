---
model_id: hunyuan-ocr
backend: vllm
hardware:
  gpu: "AMD gfx1100"
  vram_min_gb: 48
environment:
  type: docker
  rocm: "7.2"
command: |
  # 1. Start vLLM server:
  vllm serve tencent/HunyuanOCR --host 0.0.0.0 --port 10000 --max-model-len 32768

  # 2. Run adapter:
  python adapter/run_adapter.py \
    --platform linux-rocm --backend vllm \
    --server-url http://127.0.0.1:10000/v1 \
    --model tencent/HunyuanOCR \
    --img-dir /root/datasets/OmniDocBench_data/images \
    --out-dir /tmp/hunyuanocr-predictions

  # 3. Score + publish (via omnidocbench-rocm):
  omnidocbench-rocm run --stage score --platform linux-rocm --cdm \
    --predictions-dir /tmp/hunyuanocr-predictions \
    --run-stats /tmp/hunyuanocr-predictions/_run_stats.json --version v16
expected_overall:
  value: 93.64
  tolerance: 0.5
---

# Reproduce HunyuanOCR 93.64 on AMD ROCm

## Prerequisites

```bash
rocminfo | grep -E "Name:|VRAM"    # must show gfx1100 + ≥48 GB
ls -la /dev/kfd                     # must exist
```

## Quickstart

Full 1,651-page evaluation with CDM formula scoring.

```bash
# Start vLLM server on AMD GPU
vllm serve tencent/HunyuanOCR --host 0.0.0.0 --port 10000 --max-model-len 32768

# In another terminal, run adapter
cd /path/to/HunyuanOCR-ROCm
python adapter/run_adapter.py \
  --platform linux-rocm --backend vllm \
  --server-url http://127.0.0.1:10000/v1 \
  --model tencent/HunyuanOCR \
  --img-dir /root/datasets/OmniDocBench_data/images \
  --out-dir /tmp/hunyuanocr-predictions

# Score
omnidocbench-rocm run --stage score --platform linux-rocm --cdm \
  --predictions-dir /tmp/hunyuanocr-predictions \
  --run-stats /tmp/hunyuanocr-predictions/_run_stats.json --version v16
```

## Expected output

Overall **93.64** (±0.5). Text 95.48, Table TEDS 92.97%, Formula CDM 92.46%.

## If it fails

See [OmniDocBench-ROCm pitfalls](https://github.com/AIwork4me/OmniDocBench-ROCm/docs/pitfalls.md).
