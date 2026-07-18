# HunyuanOCR-ROCm Makefile.
#
# Defaults are repo-relative (under $(CURDIR)/artifacts) — NEVER /root or
# /workspace. The dataset and model directories are machine-specific, so targets
# that need them REQUIRE DATA_DIR / MODEL_DIR to be set explicitly on the command
# line or via the environment, e.g.:
#
#   make eval-canary-transformers DATA_DIR=/data/OmniDocBench_data MODEL_DIR=/models/HunyuanOCR
#
# Canary naming is canonical: OmniDocBench_canary_148.json (materialized from
# the full GT). The legacy OmniDocBench_150.json name is no longer used here.

PYTHON ?= python

# Output predictions default to a repo-local artifacts dir (overridable).
PRED_DIR ?= $(CURDIR)/artifacts/predictions
PRED_CANARY ?= $(CURDIR)/artifacts/canary-transformers/preds

# Dataset + model are machine-specific: required by the targets that need them.
DATA_DIR ?=
MODEL_DIR ?=
GPUS ?= 0,1,2

GT_FULL ?= $(DATA_DIR)/OmniDocBench.json
GT_CANARY ?= $(DATA_DIR)/OmniDocBench_canary_148.json
CANARY_MANIFEST ?= $(CURDIR)/eval/canary_148.manifest.json

# Fail fast with a clear message if a required variable is empty.
# Usage inside a recipe:  $(call require_var,DATA_DIR)
define require_var
@bash -c 'test -n "$$$(1)" || { echo "[fatal] $(1) is not set — pass $(1)=... on the command line or export it" >&2; exit 1; }'
endef

.PHONY: install-dev check test lint build doctor canary-materialize eval-canary-transformers score-canary ci-local

install-dev:
	pip install -e ".[client,download,dev]"

check:
	python scripts/check_repo.py

test:
	pytest -q -m "not gpu"

lint:
	ruff check . && ruff format --check .

build:
	python -m build

doctor:
	hunyuan-ocr doctor

# Rebuild the 148-page canary GT byte-identically from the full GT + manifest.
canary-materialize:
	$(call require_var,DATA_DIR)
	hunyuan-ocr canary materialize \
	  --full-gt "$(GT_FULL)" \
	  --manifest "$(CANARY_MANIFEST)" \
	  --out "$(GT_CANARY)"

# Phase-1 transformers driver over the 148-page canary (needs ROCm torch + model).
eval-canary-transformers:
	$(call require_var,DATA_DIR)
	$(call require_var,MODEL_DIR)
	python scripts/run_phase1_transformers.py \
	  --gt-json "$(GT_CANARY)" --images-dir "$(DATA_DIR)/images" \
	  --pred-dir "$(PRED_CANARY)" --model "$(MODEL_DIR)" --gpu-ids $(GPUS)

# Score the canary predictions produced by eval-canary-transformers.
score-canary:
	$(call require_var,DATA_DIR)
	python scripts/score_predictions.py \
	  --pred-dir "$(PRED_CANARY)" --gt-json "$(GT_CANARY)" --label transformers-canary-148

# Local approximation of CI (CPU, no torch).
ci-local: check lint test
