#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 AIwork4me
#
# Real one-page ROCm smoke for HunyuanOCR-ROCm (llama.cpp backend).
#
# Runs on a TRUSTED self-hosted gfx1100 runner. The operator provides the inputs
# via environment variables (the public repo never ships weights or OmniDocBench
# pages):
#   HUNYUANOCR_GGUF_DIR    dir with HunyuanOCR-bf16.gguf + mmproj-HunyuanOCR-bf16.gguf
#   HUNYUANOCR_SMOKE_GT    a 1-page (non-sensitive) OmniDocBench-format GT json
#   HUNYUANOCR_SMOKE_IMAGES  dir holding the image referenced by the GT
#   HUNYUANOCR_LLAMA_SERVER (optional) path to llama-server; default: on PATH
#   HUNYUANOCR_SMOKE_PORT  (optional) port; default 8081
#   HUNYUANOCR_SMOKE_OUT   (optional) artifacts dir; default ./smoke-artifacts
#
# Pipeline: verify ROCm/GPU/weights/server -> start a local llama-server (trap'd)
# -> wait for /v1/models -> predict one page -> assert non-empty, non-ERROR: markdown
# -> validate -> manifest verify -> write run_manifest + env summary. Fails fast on
# any missing prereq or bad output. No secrets are printed.
set -euo pipefail

REPO="$(cd "$(dirname "$0")/.." && pwd)"
GGUF_DIR="${HUNYUANOCR_GGUF_DIR:?HUNYUANOCR_GGUF_DIR must point to the GGUF weights dir}"
SMOKE_GT="${HUNYUANOCR_SMOKE_GT:?HUNYUANOCR_SMOKE_GT must be a 1-page OmniDocBench GT json (provided by the runner)}"
SMOKE_IMAGES="${HUNYUANOCR_SMOKE_IMAGES:?HUNYUANOCR_SMOKE_IMAGES must hold the smoke image}"
LLAMA_SERVER="${HUNYUANOCR_LLAMA_SERVER:-llama-server}"
PORT="${HUNYUANOCR_SMOKE_PORT:-8081}"
OUT="${HUNYUANOCR_SMOKE_OUT:-$REPO/smoke-artifacts}"
HOST=127.0.0.1
SERVER_PID=""

log() { echo "[smoke] $*"; }
die() { echo "[smoke][fatal] $*" >&2; exit 1; }

mkdir -p "$OUT"
: > "$OUT/smoke.stdout"
: > "$OUT/smoke.stderr"

cleanup() {
  rc=$?
  if [[ -n "$SERVER_PID" ]] && kill -0 "$SERVER_PID" 2>/dev/null; then
    log "stopping llama-server (pid $SERVER_PID)"
    kill "$SERVER_PID" 2>/dev/null || true
    wait "$SERVER_PID" 2>/dev/null || true
  fi
  exit "$rc"
}
trap cleanup EXIT INT TERM

# --- 1. verify prerequisites (ROCm, GPU, weights, server binary) -------------
[[ -e "$GGUF_DIR/HunyuanOCR-bf16.gguf" ]]      || die "missing $GGUF_DIR/HunyuanOCR-bf16.gguf"
[[ -e "$GGUF_DIR/mmproj-HunyuanOCR-bf16.gguf" ]] || die "missing $GGUF_DIR/mmproj-HunyuanOCR-bf16.gguf"
[[ -f "$SMOKE_GT" ]]                            || die "smoke GT not found: $SMOKE_GT"
[[ -d "$SMOKE_IMAGES" ]]                        || die "smoke images dir not found: $SMOKE_IMAGES"
command -v rocm-smi >/dev/null 2>&1 || [[ -d /opt/rocm ]] || die "ROCm not detected (no rocm-smi, no /opt/rocm)"
command -v "$LLAMA_SERVER" >/dev/null 2>&1      || die "llama-server not found ($LLAMA_SERVER)"
log "prerequisites OK (ROCm + weights + server present)"

# --- 2. start a local llama-server (loopback only) ---------------------------
log "starting llama-server on $HOST:$PORT"
"$LLAMA_SERVER" \
  --model "$GGUF_DIR/HunyuanOCR-bf16.gguf" \
  --mmproj "$GGUF_DIR/mmproj-HunyuanOCR-bf16.gguf" \
  --host "$HOST" --port "$PORT" --alias HYVL \
  -ngl 999 -c 65536 -n 32768 >>"$OUT/smoke.stdout" 2>>"$OUT/smoke.stderr" &
SERVER_PID=$!

# --- 3. wait for /v1/models health -------------------------------------------
healthy=0
for _ in $(seq 1 60); do
  if curl -fsS "http://$HOST:$PORT/v1/models" >/dev/null 2>&1; then
    healthy=1
    break
  fi
  kill -0 "$SERVER_PID" 2>/dev/null || die "llama-server exited before becoming healthy (see $OUT/smoke.stderr)"
  sleep 5
done
[[ "$healthy" == "1" ]] || die "llama-server did not become healthy on :$PORT within timeout"

# --- 4. predict one page ------------------------------------------------------
PRED_DIR="$OUT/predictions"
rm -rf "$PRED_DIR"
mkdir -p "$PRED_DIR"
log "predicting one page -> $PRED_DIR"
python "$REPO/scripts/run_phase2_vllm.py" \
  --backend-name llamacpp --server-alias HYVL \
  --gt-json "$SMOKE_GT" --images-dir "$SMOKE_IMAGES" \
  --pred-dir "$PRED_DIR" --host "$HOST" --ports "$PORT" \
  --model HYVL --concurrency 1 --max-retries 2 \
  >>"$OUT/smoke.stdout" 2>>"$OUT/smoke.stderr"

# --- 5. assert the output is real markdown (non-empty, not ERROR:) -----------
MD="$(find "$PRED_DIR" -maxdepth 1 -name '*.md' | head -n1)"
[[ -n "$MD" ]] || die "no .md prediction produced in $PRED_DIR"
SIZE=$(wc -c <"$MD")
[[ "$SIZE" -gt 0 ]] || die "prediction $MD is empty"
if head -n1 "$MD" | grep -q '^ERROR:'; then
  die "prediction $MD starts with 'ERROR:' (server returned an error)"
fi
log "prediction OK ($(basename "$MD"), ${SIZE} bytes)"

# --- 6. validate + manifest verify -------------------------------------------
python "$REPO/scripts/validate_predictions.py" --gt-json "$SMOKE_GT" --pred-dir "$PRED_DIR" \
  >>"$OUT/smoke.stdout" 2>>"$OUT/smoke.stderr"
PYTHONPATH="$REPO/src" python - "$PRED_DIR" <<'PY' >>"$OUT/smoke.stdout" 2>>"$OUT/smoke.stderr"
import json, sys
from hunyuan_ocr import runner
m = json.load(open(sys.argv[1] + "/run_manifest.json"))
errs = runner.validate_manifest(m)
print("[smoke] manifest status=%s run_counts=%s final_state=%s" % (m.get("status"), m.get("run_counts"), m.get("final_state")))
sys.exit(1 if errs else 0)
PY

# --- 7. write an environment summary (no secrets) ----------------------------
{
  echo "# ROCm smoke summary"
  echo "- backend: llamacpp"
  echo "- server: $HOST:$PORT (pid $SERVER_PID)"
  echo "- prediction: $(basename "$MD") ($SIZE bytes)"
  echo "- rocm-smi top device:"
  rocm-smi --showproductname --showmeminfo vram 2>/dev/null | head -n 8 | sed 's/^/  /' || echo "  (rocm-smi unavailable)"
} >"$OUT/env_summary.md"

log "SMOKE PASS -> $OUT (run_manifest.json, smoke.stdout/stderr, env_summary.md)"
