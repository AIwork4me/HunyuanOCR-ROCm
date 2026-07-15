#!/bin/bash
# Start the HunyuanOCR-1.5 vLLM server (OpenAI-compatible) on gfx1100.
# Uses the box's vLLM ROCm build (/opt/venv, vllm 0.16.1, native HunYuanVL).
#
# Usage:  MODEL_PATH=/root/models/HunyuanOCR GPU=0 PORT=8000 bash scripts/serve_vllm.sh
# Run one instance per GPU (ports 8000/8001/8002) for multi-GPU throughput.
set -e
MODEL_PATH=${MODEL_PATH:-/root/models/HunyuanOCR}
GPU=${GPU:-0}
PORT=${PORT:-8000}
GPU_MEM_UTIL=${GPU_MEM_UTIL:-0.9}
MAX_MODEL_LEN=${MAX_MODEL_LEN:-131072}
SERVED_NAME=${SERVED_NAME:-tencent/HunyuanOCR}
LOG=${LOG:-/root/hunyuanocr-results/vllm_${PORT}.log}
mkdir -p "$(dirname "$LOG")"

echo "[serve] model=${MODEL_PATH} served-as=${SERVED_NAME} gpu=${GPU} port=${PORT} log=${LOG}"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1

CUDA_VISIBLE_DEVICES=${GPU} /opt/venv/bin/vllm serve "${MODEL_PATH}" \
    --served-model-name "${SERVED_NAME}" -tp 1 \
    --limit-mm-per-prompt '{"image":4,"video":0}' \
    --trust-remote-code \
    --port ${PORT} \
    --gpu-memory-utilization ${GPU_MEM_UTIL} \
    --max-model-len ${MAX_MODEL_LEN} \
    --max-num-batched-tokens ${MAX_MODEL_LEN} \
    > "${LOG}" 2>&1 &

echo "[started] pid=$!  Readiness: curl -sf http://127.0.0.1:${PORT}/v1/models"
