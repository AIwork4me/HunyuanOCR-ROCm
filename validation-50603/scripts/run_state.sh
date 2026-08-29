#!/usr/bin/env bash
# usage: run_state.sh <state-dir-name> <vllm-branch-or-sha>
# Runs the three #50603 reproducer scripts against the given vLLM code state.
# The vLLM build must already match the branch: state-a and state-b need
# `rebuild_state.sh` after checkout (csrc change); state-c is python-only.
set -uo pipefail
HERE="$(cd "$(dirname "$0")/.." && pwd)"
STATE_NAME="$1"; BRANCH="$2"
VREPO=/workspace/vllm-50603
OUT="$HERE/$STATE_NAME"

source "$HERE/env/env.sh"

cd "$VREPO"
git checkout -q --force "$BRANCH"
if [[ -n "$(git status --porcelain)" ]]; then
  echo "FATAL: vllm-50603 working tree dirty after checkout of $BRANCH" >&2
  exit 1
fi
git rev-parse HEAD > "$OUT/vllm-head.txt"
"$HERE/scripts/capture_env.sh" "$STATE_NAME" > "$OUT/env-capture.txt"

mkdir -p "$OUT"
rc_total=0
for s in repro_determinism repro_warmup repro_control_short; do
  echo "[$STATE_NAME] running $s.py ..."
  if V50603_EVIDENCE="$OUT/$s.evidence.json" timeout 110m \
      python "$HERE/scripts/$s.py" > "$OUT/$s.log" 2>&1; then
    echo "[$STATE_NAME] $s.py ok"
  else
    echo "[$STATE_NAME] $s.py FAILED rc=$? (log kept)"
    rc_total=1
  fi
done

# Routing evidence: State A/B must show the Triton-fallback warning
# (gqa_ratio=2 excluded from the CK kernel); State C must NOT show it.
grep -hE "Cannot use ROCm custom paged attention|kernel_paged_attention_2d|JIT compilation during inference" \
  "$OUT"/*.log | sort -u > "$OUT/routing-markers.txt" || true
echo "[$STATE_NAME] routing markers:"; sed 's/^/    /' "$OUT/routing-markers.txt" 2>/dev/null
exit $rc_total
