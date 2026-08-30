# validation-53615 — HunyuanOCR regression after the vLLM Transformers-backend migration (#53615)

**What this is:** a complete, reproducible evidence package for the HunyuanOCR runtime regression on current [vllm-project/vllm](https://github.com/vllm-project/vllm) main, introduced when [PR #53615](https://github.com/vllm-project/vllm/pull/53615) migrated `HunYuanVLForConditionalGeneration` from the (deleted, #53272) native implementation to `TransformersMultiModalForCausalLM`, together with a locally validated candidate fix.

**Hardware / scope:** one AMD Radeon PRO W7900D (gfx1100, RDNA3, 48 GB), ROCm 7.14 wheel stack, TP=1 only. **TP>1 and PP>1 are NOT TESTED** — do not read anything here as multi-GPU validation.

**Main conclusion:** HunyuanOCR is broken on the post-#53615 Transformers backend because three migration gaps stack on top of each other: (1) the generic m-RoPE position path hardcodes 3 axes while transformers normalizes this model into a 4-section `mrope_section`; (2) the backend's `kv_lora_rank`-presence heuristic misreads HunYuanVL's vestigial DeepSeek-lineage config fields as MLA and poisons the attention head size (192 instead of the GQA 128); (3) the generic `get_mrope_input_positions` wrapper passes `video_grid_thw=None` into `get_rope_index` signatures that don't accept it. A 9-file candidate patch fixes all three; after it, greedy image→OCR generation is byte-identical to the plain-HF reference on transformers 5.13.0 and faithful on 5.15.1, deterministically, on the latest validated upstream SHA `1dc464d426` (2026-08-30).

**Status:** candidate fix validated locally and rebased onto latest upstream main; **upstream PR NOT YET SUBMITTED** — intentionally held for human review (recommended split: two PRs, see [SUMMARY.md](SUMMARY.md)).

## Navigate

| What | Where |
|---|---|
| Full findings report + latest-main addendum | [SUMMARY.md](SUMMARY.md) |
| How to reproduce STATE B (pristine main, expected FAIL) and STATE C (patched, expected PASS) | [REPRODUCE.md](REPRODUCE.md) |
| Baseline failure on pristine main (original base `b5707bf994`) | [baseline/run-baseline.log](baseline/run-baseline.log) |
| Baseline failure + root-cause reconfirmation on latest main (`1dc464d426`) | [latest-main/](latest-main/) |
| Root-cause instrumentation (positions, configs, vestigial MLA fields) | [instrumentation/](instrumentation/) |
| Focused regression tests (3 fail-on-main / 3 guards) | [latest-main/tests-all-before-fix.log](latest-main/tests-all-before-fix.log), [latest-main/tests-focused-and-mrope-after-fix.log](latest-main/tests-focused-and-mrope-after-fix.log) |
| W7900 E2E matrices (original base and latest main) | [e2e/](e2e/), [latest-main/e2e-*.json](latest-main/) |
| Candidate patch (latest main) + historical base patch | [diff/hunyuanocr-vllm-candidate.patch](diff/hunyuanocr-vllm-candidate.patch), [diff/hunyuanocr-vllm-candidate-b5707bf994.patch](diff/hunyuanocr-vllm-candidate-b5707bf994.patch) |
| Reproducer / instrumentation / E2E / build scripts | [scripts/](scripts/) |
| Native-implementation historical reference (incl. pre-removal native bug) | [state-a/](state-a/) |
| Run environment (env.sh is host-specific by design) | [env/](env/) |
| File inventory + integrity checksums | [ARTIFACTS.txt](ARTIFACTS.txt), [SHA256SUMS](SHA256SUMS) |

The vLLM candidate lives uncommitted in `/workspace/vllm-mainline-probe` (branch `investigate/hunyuanocr-transformers-regression`, no upstream, never pushed). Model: `tencent/HunyuanOCR` pinned at revision `de8f10ad2f00a0cefd790b526de8a65dcfdb3205`, run with `HF_HUB_OFFLINE=1`.
