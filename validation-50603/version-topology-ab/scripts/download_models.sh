#!/usr/bin/env bash
# Download pinned model snapshots for the TP=1 A/B (via hf-mirror; huggingface.co unreachable from this box).
set -x
export HF_HOME=/root/.cache/huggingface
export HF_ENDPOINT=https://hf-mirror.com
PY=/workspace/venv-vllm50603/bin/python
cd /workspace/vllm-50603-version-ab
$PY - <<'PY1'
from huggingface_hub import snapshot_download
p = snapshot_download(
    "RedHatAI/Muse-Glimmer-30B-INT4",
    revision="f5b410ce4234fad70eef8be99b4680ee4e30b418")
print("MUSE_SNAPSHOT", p)
PY1
echo "MUSE_RC=$?"
$PY - <<'PY2'
from huggingface_hub import snapshot_download
p = snapshot_download(
    "RedHatAI/gemma-3-27b-it-quantized.w4a16",
    revision="2b537554d6c6f6368945e8df4e5fb7bbbb5d56c9")
print("GEMMA_SNAPSHOT", p)
PY2
echo "GEMMA_RC=$?"
echo "=== DOWNLOADS COMPLETE $(date -Is) ==="
