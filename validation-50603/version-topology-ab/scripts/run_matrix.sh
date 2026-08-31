#!/usr/bin/env bash
# Run the TP=1 measurement matrix in one isolated environment.
# Usage: run_matrix.sh <env-name> <version-tag> [muse-only|muse+gemma3]
# One engine (process) per invocation of nondet_eager_tp1.py — matching
# cadamcat's harness shape. Engine index = independent process repetition.
set -u
ROOT=/workspace/vllm-50603-version-ab
NAME=$1; VTAG=$2; SCOPE=${3:-muse-only}
VENV=$ROOT/environments/$NAME
PY=$VENV/bin/python
HARNESS=$ROOT/harness/nondet_eager_tp1.py
RES=$ROOT/results/$VTAG
LOGS=$ROOT/logs/$VTAG
mkdir -p "$RES" "$LOGS"

# Run environment (identical for both arms).
export HF_HOME=/root/.cache/huggingface
export HF_HUB_OFFLINE=1
export CUDA_VISIBLE_DEVICES=0
export HSA_OVERRIDE_GFX_VERSION=11.0.0
export PYTHONPATH="$VENV/lib/python3.12/site-packages/_rocm_sdk_core/share/amd_smi${PYTHONPATH:+:$PYTHONPATH}"
export AB_MM_LIMIT_ZERO=1  # symmetric both arms: 0.23 arm OOMs at ViT profiling without AMD flash_attn (see harness.diff)

wait_idle() {
  for i in $(seq 1 40); do
    v=$(rocm-smi --showmeminfo vram --csv 2>/dev/null | awk -F, '/GPU_0/ {gsub(/[^0-9]/,"",$3); print $3}' | head -1)
    [ -n "$v" ] && [ "$v" -lt 3000000000 ] && return 0
    sleep 5
  done
  echo "  WARNING vram did not return to idle"
}

run_cell() { # model engine eager
  local which=$1 eng=$2 eager=$3
  local tag="$which-e$eager-eng$eng"
  echo "--- $VTAG $tag  $(date -Is) ---"
  wait_idle
  timeout 75m "$PY" "$HARNESS" "$which" "$eng" "$eager" "$RES/$tag.json" \
    > "$LOGS/$tag.log" 2>&1
  local rc=$?
  grep -aE '^\[env\]|^\[sampling\]|^\[within\]|NONDET DONE' "$LOGS/$tag.log" | sed 's/^/  /'
  echo "  rc=$rc  vram_peak_marker: $(grep -ac 'GPU KV cache size' "$LOGS/$tag.log")"
  if [ $rc -ne 0 ]; then
    echo "  RUN FAILED rc=$rc; tail:"; tail -8 "$LOGS/$tag.log" | sed 's/^/    /'
  fi
}

MODELS="muse"; ENGINES="1 2 3"
if [ "$SCOPE" = "muse+gemma3" ]; then MODELS="muse gemma3"; fi

for which in $MODELS; do
  for eager in 0 1; do
    for eng in $ENGINES; do
      # gemma3 is secondary: one engine per cell mirrors cadamcat's published cells
      if [ "$which" = "gemma3" ] && [ "$eng" != "1" ]; then continue; fi
      run_cell "$which" "$eng" "$eager"
    done
  done
done
echo "=== MATRIX $VTAG COMPLETE $(date -Is) ==="
