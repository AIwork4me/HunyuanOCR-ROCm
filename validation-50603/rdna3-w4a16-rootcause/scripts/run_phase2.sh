#!/usr/bin/env bash
# Phase 2 baseline: (A) isolated microprobe 3x200, (B) Muse E2E 3 engines.
set -u
ROOT=/workspace/vllm-50603-rdna3-rootcause
AB=/workspace/vllm-50603-version-ab
VENV=$AB/environments/env-0.25.1
export HF_HOME=/root/.cache/huggingface HF_HUB_OFFLINE=1 CUDA_VISIBLE_DEVICES=0
export HSA_OVERRIDE_GFX_VERSION=11.0.0 AB_MM_LIMIT_ZERO=1
export PYTHONPATH="$VENV/lib/python3.12/site-packages/_rocm_sdk_core/share/amd_smi${PYTHONPATH:+:$PYTHONPATH}"

echo "=== A. microprobe 3 procs x200 ==="
cd $AB/worktrees/v0.25.1
for TAG in baseA baseB baseC; do
  timeout 15m "$VENV/bin/python" $AB/forensics/harness/w4a16_probe.py $TAG 200 2>&1 | grep '"tag"' | tee -a $ROOT/microprobe/baseline-microprobe.jsonl
done

echo "=== B. Muse E2E 3 engines ==="
mkdir -p $ROOT/e2e/baseline $ROOT/logs
for eng in 1 2 3; do
  echo "--- baseline eng$eng $(date -Is) ---"
  timeout 40m "$VENV/bin/python" $AB/harness/nondet_eager_tp1.py muse "$eng" 1 \
    "$ROOT/e2e/baseline/eng$eng.json" > "$ROOT/logs/e2e-baseline-eng$eng.log" 2>&1
  grep -aE '^\[within\]' "$ROOT/logs/e2e-baseline-eng$eng.log" | sed 's/^/  /'
done
echo "=== PHASE2 DONE $(date -Is) ==="
