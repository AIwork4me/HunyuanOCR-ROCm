MODEL ?= /root/models/HunyuanOCR
GPUS ?= 0,1,2
DATA ?= /workspace/OmniDocBench_data
GT_FULL ?= $(DATA)/OmniDocBench.json
GT_CANARY ?= $(DATA)/OmniDocBench_150.json
PRED_FULL ?= /root/hunyuanocr-results/phase1-transformers/preds
PRED_CANARY ?= /root/hunyuanocr-results/canary-transformers/preds

.PHONY: demo eval-canary eval-full score score-canary oracle-check
demo:
	python scripts/run_phase1_transformers.py --gt-json $(DATA)/OmniDocBench_30.json --images-dir $(DATA)/images --pred-dir /root/hunyuanocr-results/demo/preds --model $(MODEL) --gpu-ids 0 --limit 3
eval-canary:
	python scripts/run_phase1_transformers.py --gt-json $(GT_CANARY) --images-dir $(DATA)/images --pred-dir $(PRED_CANARY) --model $(MODEL) --gpu-ids $(GPUS)
	python scripts/score_predictions.py --pred-dir $(PRED_CANARY) --gt-json $(GT_CANARY) --label transformers-canary-150
eval-full:
	python scripts/run_phase1_transformers.py --gt-json $(GT_FULL) --images-dir $(DATA)/images --pred-dir $(PRED_FULL) --model $(MODEL) --gpu-ids $(GPUS)
score:
	python scripts/score_predictions.py --pred-dir $(PRED_FULL) --gt-json $(GT_FULL) --label transformers
score-canary:
	python scripts/score_predictions.py --pred-dir $(PRED_CANARY) --gt-json $(GT_CANARY) --label transformers-canary-150
oracle-check:
	@echo "oracle = transformers canary (150). Re-run: make score-canary"
