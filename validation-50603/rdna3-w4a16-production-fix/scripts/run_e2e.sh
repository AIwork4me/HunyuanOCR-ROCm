#!/usr/bin/env bash
# Phase 12-13 E2E: production build (deterministic by default).
# A: Muse eager x3 engines; B: Muse graphs x1; C: gemma eager x1; D: Muse legacy-atomic control x1.
set -u
ROOT=/workspace/vllm-50603-rdna3-production-fix
VENV=$ROOT/env-prod
BASE=$(dirname $VENV)
export HF_HOME=/root/.cache/huggingface HF_HUB_OFFLINE=1 CUDA_VISIBLE_DEVICES=0
export HSA_OVERRIDE_GFX_VERSION=11.0.0 AB_MM_LIMIT_ZERO=1
export PYTHONPATH="$VENV/lib/python3.12/site-packages/_rocm_sdk_core/share/amd_smi${PYTHONPATH:+:$PYTHONPATH}"
mkdir -p $ROOT/e2e/det-eager $ROOT/e2e/det-graphs $ROOT/e2e/gemma $ROOT/e2e/legacy-control $ROOT/logs

run() { # outdir eng eager model legenv
  local OUT=$1 ENG=$2 EAGER=$3 MODEL=$4 LEG=$5
  local EXTRA=""
  [ "$LEG" = "1" ] && EXTRA="VLLM_RDNA3_W4A16_LEGACY_ATOMIC=1"
  echo "--- $OUT eng$ENG eager=$EAGER leg=$LEG $(date -Is) ---"
  env $EXTRA timeout 60m "$VENV/bin/python" /workspace/vllm-50603-version-ab/harness/nondet_eager_tp1.py \
    "$MODEL" "$ENG" "$EAGER" "$ROOT/e2e/$OUT/eng$ENG.json" > "$ROOT/logs/e2e-$OUT-eng$ENG.log" 2>&1
  grep -aE '^\[within\]' "$ROOT/logs/e2e-$OUT-eng$ENG.log" | sed 's/^/  /'
}

for eng in 1 2 3; do run det-eager $eng 1 muse 0; done
run det-graphs 1 0 muse 0
run gemma 1 1 gemma3 0
run legacy-control 1 1 muse 1
echo "=== routing ==="
grep -ahE 'Using .*LinearKernel' $ROOT/logs/e2e-det-eager-eng1.log $ROOT/logs/e2e-legacy-control-eng1.log | sort -u
echo "=== E2E DONE $(date -Is) ==="
