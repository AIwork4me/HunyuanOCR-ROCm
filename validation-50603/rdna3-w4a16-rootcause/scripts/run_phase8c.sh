#!/usr/bin/env bash
# Phase 8 State C: deterministic prototype E2E (env-rdna3, VLLM_RDNA3_W4A16_DETERMINISTIC=1)
set -u
ROOT=/workspace/vllm-50603-rdna3-rootcause
VENV=$ROOT/env-rdna3
export HF_HOME=/root/.cache/huggingface HF_HUB_OFFLINE=1 CUDA_VISIBLE_DEVICES=0
export HSA_OVERRIDE_GFX_VERSION=11.0.0 AB_MM_LIMIT_ZERO=1
export VLLM_RDNA3_W4A16_DETERMINISTIC=1
export PYTHONPATH="$VENV/lib/python3.12/site-packages/_rocm_sdk_core/share/amd_smi${PYTHONPATH:+:$PYTHONPATH}"
mkdir -p $ROOT/e2e/prototype $ROOT/logs
for eng in 1 2 3; do
  echo "--- prototype eng$eng $(date -Is) ---"
  timeout 60m "$VENV/bin/python" /workspace/vllm-50603-version-ab/harness/nondet_eager_tp1.py muse "$eng" 1 \
    "$ROOT/e2e/prototype/eng$eng.json" > "$ROOT/logs/e2e-proto-eng$eng.log" 2>&1
  grep -aE '^\[within\]' "$ROOT/logs/e2e-proto-eng$eng.log" | sed 's/^/  /'
done
echo "=== routing evidence ==="
grep -ahE 'Using .*LinearKernel|deterministic split-K' $ROOT/logs/e2e-proto-eng*.log | sort -u
echo "=== PHASE8C DONE $(date -Is) ==="
