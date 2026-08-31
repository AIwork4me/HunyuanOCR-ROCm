#!/usr/bin/env bash
# Phase 3: kernel ON/OFF E2E causal A/B. ON arm = phase2 baseline (reuse);
# this script runs the OFF arm and extracts routing evidence for both.
set -u
ROOT=/workspace/vllm-50603-rdna3-rootcause
AB=/workspace/vllm-50603-version-ab
VENV=$AB/environments/env-0.25.1
export HF_HOME=/root/.cache/huggingface HF_HUB_OFFLINE=1 CUDA_VISIBLE_DEVICES=0
export HSA_OVERRIDE_GFX_VERSION=11.0.0 AB_MM_LIMIT_ZERO=1
export VLLM_DISABLED_KERNELS=RDNA3W4A16LinearKernel
export PYTHONPATH="$VENV/lib/python3.12/site-packages/_rocm_sdk_core/share/amd_smi${PYTHONPATH:+:$PYTHONPATH}"
mkdir -p $ROOT/e2e/kernel-off $ROOT/logs $ROOT/kernel-off

for eng in 1 2 3; do
  echo "--- OFF eng$eng $(date -Is) ---"
  timeout 40m "$VENV/bin/python" $AB/harness/nondet_eager_tp1.py muse "$eng" 1 \
    "$ROOT/e2e/kernel-off/eng$eng.json" > "$ROOT/logs/e2e-off-eng$eng.log" 2>&1
  grep -aE '^\[within\]' "$ROOT/logs/e2e-off-eng$eng.log" | sed 's/^/  /'
done
# routing evidence for both arms
for arm in off; do
  echo "=== routing ($arm) ==="
  grep -ahE 'Selected .* for |Using .*LinearKernel|disabled by environment' $ROOT/logs/e2e-$arm-eng*.log | sort -u
done
echo "=== routing (on, from phase2 logs) ==="
grep -ahE 'Selected .* for |Using .*LinearKernel' $ROOT/logs/e2e-baseline-eng*.log | sort -u
echo "=== PHASE3 DONE $(date -Is) ==="
