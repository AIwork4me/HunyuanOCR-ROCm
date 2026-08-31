#!/usr/bin/env bash
# Secondary model matrix: gemma3, 1 engine per (version, eager) — mirrors cadamcat's one-engine-per-cell.
set -u
ROOT=/workspace/vllm-50603-version-ab
run() { # env tag
  local NAME=$1 VTAG=$2
  local VENV=$ROOT/environments/$NAME
  export HF_HOME=/root/.cache/huggingface HF_HUB_OFFLINE=1 CUDA_VISIBLE_DEVICES=0
  export HSA_OVERRIDE_GFX_VERSION=11.0.0 AB_MM_LIMIT_ZERO=1
  export PYTHONPATH="$VENV/lib/python3.12/site-packages/_rocm_sdk_core/share/amd_smi${PYTHONPATH:+:$PYTHONPATH}"
  for eager in 0 1; do
    echo "--- $VTAG gemma3-e$eager-eng1 $(date -Is) ---"
    timeout 75m "$VENV/bin/python" "$ROOT/harness/nondet_eager_tp1.py" gemma3 1 "$eager" "$ROOT/results/$VTAG/gemma3-e$eager-eng1.json" \
      > "$ROOT/logs/$VTAG/gemma3-e$eager-eng1.log" 2>&1
    rc=$?
    grep -aE '^\[within\]|NONDET DONE' "$ROOT/logs/$VTAG/gemma3-e$eager-eng1.log" | sed 's/^/  /'
    [ $rc -ne 0 ] && { echo "  FAILED rc=$rc"; tail -5 "$ROOT/logs/$VTAG/gemma3-e$eager-eng1.log" | sed 's/^/    /'; }
    sleep 10
  done
}
run env-0.25.1 vllm-0.25.1
run env-0.23.1.dev1 vllm-0.23.1.dev1
echo "=== GEMMA MATRICES COMPLETE $(date -Is) ==="
