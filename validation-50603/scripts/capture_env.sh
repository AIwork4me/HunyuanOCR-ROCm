#!/usr/bin/env bash
# usage: capture_env.sh <state-name>  (call with the validation venv on PATH)
# Prints the full provenance of the vLLM code state + python stack to stdout.
set -euo pipefail
STATE="$1"
VREPO=/workspace/vllm-50603

echo "state:            $STATE"
echo "captured:         $(date -Is)"
python - <<'PY'
import torch
print("host gpu:         " + torch.cuda.get_device_name(0))
PY
cd "$VREPO"
echo "vllm branch:      $(git branch --show-current)"
echo "vllm HEAD:        $(git rev-parse HEAD)"
echo "vllm subject:     $(git log -1 --format=%s)"
echo "diff vs state-a:  $(git diff state-a --stat | tail -n1)"
echo "dirty files:      $(git status --porcelain | wc -l)"
echo "gate line (rocm.py):"
grep -n 'gqa_ratio >= ' vllm/platforms/rocm.py | sed 's/^/  /'
echo "python stack:"
python - <<'PY'
import torch, transformers, triton
import vllm
print("  torch        :", torch.__version__, "| hip:", torch.version.hip)
print("  vllm         :", vllm.__version__, "|", vllm.__file__)
print("  transformers :", transformers.__version__)
print("  triton       :", triton.__version__)
PY
echo "pip freeze:"
pip freeze | sort | sed 's/^/  /'
