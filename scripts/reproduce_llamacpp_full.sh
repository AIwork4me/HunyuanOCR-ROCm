#!/usr/bin/env bash
# Reproduce the published llama.cpp full-set (1651-page) result on OmniDocBench v1.6.
# Pipeline: predict -> validate -> score. See reproduce_llamacpp_canary.sh for server start.
set -euo pipefail

LLAMA_DIR="${LLAMA_DIR:?LLAMA_DIR must point to a llama.cpp checkout at the locked commit}"
GGUF_DIR="${GGUF_DIR:?GGUF_DIR must contain HunyuanOCR-bf16.gguf + mmproj-HunyuanOCR-bf16.gguf}"
DATA_DIR="${DATA_DIR:?DATA_DIR must be the OmniDocBench data dir}"
GT_JSON="${GT_JSON:-$DATA_DIR/OmniDocBench.json}"
OUT_DIR="${OUT_DIR:?OUT_DIR must be set (output predictions dir)}"
PORTS="${PORTS:-8081,8082,8083,8084}"
HOST="${HOST:-127.0.0.1}"
CONCURRENCY="${CONCURRENCY:-16}"
LOCKED_LLAMA_COMMIT="a320cbfcb7056b7b81fb854d97fe01d0ea77c4b5"

echo "[repro] full 1651 pages, llama.cpp @ $LOCKED_LLAMA_COMMIT"

for f in "$GT_JSON" "$DATA_DIR/images" "$GGUF_DIR/HunyuanOCR-bf16.gguf" \
         "$GGUF_DIR/mmproj-HunyuanOCR-bf16.gguf" "$LLAMA_DIR/build/bin/llama-server"; do
  [[ -e "$f" ]] || { echo "[fatal] missing: $f" >&2; exit 1; }
done

ACTUAL="$(git -C "$LLAMA_DIR" rev-parse HEAD)"
[[ "$ACTUAL" == "$LOCKED_LLAMA_COMMIT" ]] || { echo "[fatal] llama.cpp HEAD=$ACTUAL != $LOCKED_LLAMA_COMMIT" >&2; exit 1; }

RESUME="${RESUME:-0}"
OVERWRITE="${OVERWRITE:-0}"
if [[ "$RESUME" = "1" && "$OVERWRITE" = "1" ]]; then
  echo "[fatal] set RESUME=1 OR OVERWRITE=1, not both" >&2; exit 1
fi
if compgen -G "$OUT_DIR/*.md" > /dev/null; then
  if [[ "$RESUME" != "1" && "$OVERWRITE" != "1" ]]; then
    echo "[fatal] $OUT_DIR already has predictions; set RESUME=1 to continue or OVERWRITE=1 to redo" >&2
    exit 1
  fi
fi
mkdir -p "$OUT_DIR"

REPO="$(cd "$(dirname "$0")/.." && pwd)"

MODE_FLAG=""
[[ "$OVERWRITE" = "1" ]] && MODE_FLAG="--overwrite"
MODE_DESC="fresh"
[[ "$RESUME" = "1" ]] && MODE_DESC="resume"
[[ "$OVERWRITE" = "1" ]] && MODE_DESC="overwrite"

echo "[repro] step 1/4: predict ($MODE_DESC)"
python "$REPO/scripts/run_inference.py" \
  --backend-name llamacpp --server-alias HYVL \
  --gt-json "$GT_JSON" --images-dir "$DATA_DIR/images" \
  --pred-dir "$OUT_DIR" --host "$HOST" --ports "$PORTS" \
  --model HYVL --concurrency "$CONCURRENCY" $MODE_FLAG

echo "[repro] step 2/4: validate"
python "$REPO/scripts/validate_predictions.py" --gt-json "$GT_JSON" --pred-dir "$OUT_DIR"

echo "[repro] step 3/4: score"
python "$REPO/scripts/score_predictions.py" --pred-dir "$OUT_DIR" --gt-json "$GT_JSON" \
  --label llamacpp-full-1651

echo "[repro] step 4/4: verify manifest conservation laws"
PYTHONPATH="$REPO/src" python -c "import json,sys; from hunyuan_ocr import runner; \
m=json.load(open('$OUT_DIR/run_manifest.json')); \
errs=runner.validate_manifest(m); \
print('[manifest] backend=%s run=%s final=%s' % (m.get('backend'), m.get('run_counts'), m.get('final_state'))); \
sys.exit(1 if errs else 0)" \
  || { echo "[fatal] run_manifest.json violates conservation laws: see above" >&2; exit 1; }

echo "[repro] done."
