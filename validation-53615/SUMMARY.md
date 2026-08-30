# validation-53615 — complete evidence report

Part 1 preserves the full investigation report from the original validation run (candidate base `b5707bf994`, 2026-08-30). Part 2 is the latest-main addendum (rebase to `1dc464d426`, duplicate-work search, two additional focused regression tests, full revalidation).

## Part 1 — original investigation report (base `b5707bf994`)

### Executive result

READY FOR HUMAN PR REVIEW (held for human decision; nothing has been submitted upstream):

- Baseline reproduced on pristine `origin/main` (`b5707bf994`): `ValueError: Expected 4 multimodal RoPE channels, got position_ids with shape (3, 1, 8192)` at engine init.
- Root cause proven with live runtime values: HunyuanOCR configures 4 multimodal RoPE axes (transformers normalizes `xdrope_section=[16,16,16,16]` into a 4-section `mrope_section`), but vLLM allocates/propagates 3 (hardcoded in both the V2 `RopeState` and the legacy V1 runner buffer).
- Two additional #53615-migration blockers were found, root-caused and fixed: a vestigial-`kv_lora_rank` heuristic that poisoned `head_size` to 192 (HunYuanVL attention is plain GQA `head_dim=128`), and a Qwen-centric `get_rope_index(**kwargs)` call passing `video_grid_thw=None` that HunYuanVL's signature rejects.
- After the candidate diff: real greedy image→OCR generation completes on the W7900, byte-identical to the HF reference implementation on transformers 5.13.0 and producing faithful OCR on 5.15.1, deterministic across eager/compiled/prefix-cache-off/V1-runner modes (identical token hashes).

### Exact base revision

upstream/main `b5707bf994cb968adfc7a29fbb80b0582f53f38d` (2026-08-29 19:18:42 -0700, "[Bugfix][MoE] Enable cuBLAS out_dtype router GEMM on all CUDA archs (#54048)"), vLLM version string `v0.13.0rc1-8455-gb5707bf994`, local branch `investigate/hunyuanocr-transformers-regression` (no upstream, never pushed).

### Environment

1× AMD Radeon PRO W7900D-class gfx1100 (RDNA3), 51,522,830,336 B ≈ 48 GB VRAM; ROCm 7.14 wheel stack (`torch.version.hip` 7.14.60850, TheRock SDK; build toolchain system hipcc 7.2.1); torch 2.12.0+rocm7.14.0; transformers 5.15.1 for final E2E and 5.13.0 for the parity study; Python 3.12.3; model `tencent/HunyuanOCR` pinned at `de8f10ad2f00a0cefd790b526de8a65dcfdb3205` with `HF_HUB_OFFLINE=1`. pypi.org and huggingface.co are unreachable from this host; pip used the Tsinghua mirror and hub-dependent vLLM tests cannot run (see test matrix).

### Root cause 1 — 4-axis M-RoPE vs hardcoded 3

The checkpoint's raw `text_config.rope_scaling` is `{"type": "xdrope", "xdrope_section": [16,16,16,16], alpha 1000, …}`. Transformers (5.13+) `HunYuanVLTextConfig` normalization (`_normalize_mrope_section_alias` + `convert_rope_params_to_dict`, `configuration_hunyuan_vl.py:170-215`) renames `xdrope_section`→`mrope_section`, converts type `xdrope`→`dynamic`, validates the sections sum to `head_dim//2` (64 = 128//2), and removes the xdrope fields. Post-normalization probes: `uses_mrope = True`, `uses_xdrope_dim = 0`, `text_config.rope_parameters["mrope_section"] = [16,16,16,16]`, `type(hf_config)=HunYuanVLConfig`, `type(hf_text_config)=HunYuanVLTextConfig`. `get_rope_state()` (`vllm/v1/worker/gpu/mm/rope.py`) checks `uses_mrope` first and returned `RopeState(num_dims=3)` hardcoded — instrumented proof `positions_buffer=(3, 8193)`; the legacy V1 runner hardcodes `(3, max_num_tokens+1)` the same way. HF `HunYuanVLModel.get_rope_index` returns `(4, 1, 1038)` for the exact workload (boundary A); the profile-run dummy batch hands `(3, 1, 8192)` into `HunYuanVLTextModel.forward`, whose gate `num_mrope_axes = len(self.rotary_emb.mrope_section) = 4` raises the ValueError. Config geometry: `head_dim=128`, `num_attention_heads=16`, `num_key_value_heads=8`, `hidden_size=1024`, `vocab_size=120818`. Why #53615 exposed it: the native implementation used `SupportsXDRoPE` with config-derived 4-axis positions; the migration moved the model onto the generic 3-axis m-RoPE path. Fix: derive the axis count from the normalized config (`len(mrope_section)` via new `num_mrope_axes()` in `vllm/transformers_utils/config.py`, exposed as `ModelConfig.num_mrope_axes`, used by both runners; 3-axis fallback preserves legacy behavior). Qwen-style 3-section models are unchanged, asserted by test.

### Root cause 2 — false MLA geometry (head_size 192)

After fixing RC1, engine init fails with `RuntimeError: shape '[-1, 16, 192]' is invalid for input of size 16777216` at `vllm/model_executor/layers/attention/attention.py:524` (`query.view(-1, num_heads, head_size)`), reached via HF `HunYuanVLDenseV1Attention.forward` → `vllm_attention_forward`. The query arrives correct for GQA (16,777,216 elements = 8192 tokens × 16 heads × 128; `k`/`v` = 8 heads × 128). Cause: `create_attention_instances` (`vllm/model_executor/models/transformers/base.py`) treats any `kv_lora_rank` in the text config as MLA and, when `MLAAttention` isn't used, overwrites `head_size = qk_nope_head_dim + qk_rope_head_dim = 128 + 64 = 192`. But HunYuanVL's text attention is plain GQA over `head_dim=128` (`q_proj = num_attention_heads × head_dim` etc.); its checkpoint carries vestigial DeepSeek-lineage MLA fields (`kv_lora_rank=512`, `qk_nope_head_dim=128`, `qk_rope_head_dim=64`, `q_lora_rank=1536`, `v_head_dim=128`) that the implementation ignores. Category: vLLM generic adapter heuristic (A) triggered by misleading config fields (C). Fix: apply the override only when `MLAFuser` modules actually matched (`elif mla_fusers:`) — a structural gate, no model names.

### Root cause 3 — incompatible None kwargs

After RC2, the first real request fails with `TypeError: HunYuanVLModel.get_rope_index() got an unexpected keyword argument 'video_grid_thw'`: the generic `get_mrope_input_positions` wrapper (`vllm/model_executor/models/transformers/multimodal.py`) always passed `image_grid_thw=` and `video_grid_thw=` even when `None`, and HunYuanVL's signature has no `video_grid_thw`. Fix: only pass grid kwargs that are actually present (every HF `get_rope_index` defaults them to `None`, so absent-modality omission is safe for Qwen-style signatures too).

### Regression test (RC1)

`tests/v1/worker/test_rope_state.py` — `test_rope_state_qwen_style_mrope_keeps_3_axes` (guard, passes on main) and `test_rope_state_hunyuan_style_xdrope_gets_4_axes` (fails on pristine main at `AssertionError: assert 3 == 4`, passes with the fix). CPU-only, tmp-dir config fixtures through the full transformers normalization path.

### Three-state regression table

| State | Revision | Backend | Result | Position shape |
|---|---|---|---|---|
| pre-#53615 (immediate parent of native removal #53272) | `27ec8ac626` (2026-08-21 main) | native | FAIL at weight load: `no module or parameter named 'lm_head'` — pre-existing native-path bug (weight-tying refactors #51665/#52147 in that window); the checkpoint does carry top-level `lm_head.weight` under its raw `model.*`/`vit.*` layout | never reached |
| pre-#53615-era native reference (proven working) | `01a3fe7d2f` (v0.25.1 tag + validation-50603 backports, state-c) | native | PASS — same workload, faithful markdown OCR, exit 0 | not captured (native internal 4-axis XD-RoPE path) |
| current main | `b5707bf994` | Transformers | FAIL — `Expected 4 multimodal RoPE channels, got position_ids with shape (3, 1, 8192)` | `(3, 8193)` buffer → `(3, 1, 8192)` at HF forward |
| candidate fix | local diff on `b5707bf994` | Transformers | PASS — faithful OCR on transformers 5.15.1; token-identical to HF reference on 5.13.0 | `(4, 8193)` buffer → `(4, 1, N)` into HF rotary |

The pre-removal native failure is a product-level bug (full traceback in `state-a/run-state-a.log`), evidence the native path was already broken before the migration.

### W7900 E2E (original base)

Greedy (`temperature=0, top_p=1.0, top_k=-1, max_tokens=128`), TP=1, transformers 5.15.1, pinned model revision. Token hash = sha256 of token IDs (first 16 hex):

| Test | Image | Result | Deterministic | Hash |
|---|---|---|---|---|
| 1 simple | 640×400 "W7900-OK" | PASS | — | 379b06d5b4cc4dca |
| 2 document | 1024×960 financial page | PASS | — | 6cd5fba4cdb2f135 |
| 3 repeatability ×3 | same document request | PASS | YES (3/3 identical) | 6cd5fba4cdb2f135 ×3 |
| 4 alternate aspect | portrait 768×1280 log | PASS | — | eee9091f4a671a11 |

Mode variants (all 4 tests each): `enforce_eager` PASS · default mode + `cudagraph_mode="PIECEWISE"` PASS (identical doc hash — cross-mode determinism) · prefix caching disabled PASS · V1 model runner (`VLLM_USE_V2_MODEL_RUNNER=0`) PASS (exercises the V1 buffer fix). Instrumented rotary log confirms 4-axis positions reach the model: `position_ids (4, 1, 8192) … mrope_section=[16,16,16,16]`, then `(4, 1, 1)` per decode step.

### Correctness oracle (transformers version sensitivity)

vLLM+fixes on transformers 5.13.0 produces output byte-for-byte identical (same token-ID list) to plain HF transformers 5.13.0 `sdpa` greedy on the same GPU — both produce the same garbled text for this workload; on transformers 5.15.1 both produce the faithful markdown. Conclusion: the garble at 5.13.0 is an HF-side issue (fixed upstream by 5.15.1), not a vLLM bug; the vLLM plumbing reproduces the reference implementation exactly. Logs: `instrumentation/run-hf-control.log`, `instrumentation/run-hf-control-5151.log`, `instrumentation/run-vllm-5151.log`.

### Known limitations (original base)

Default mode with FULL CUDA graphs fails during capture on ROCm inside the HF rotary's lazy cos/sin cache growth (`hipErrorStreamCaptureUnsupported`) — pre-existing, unrelated to this diff; piecewise graphs and `enforce_eager` both work. This package's E2E does not claim FULL cudagraph support.

### Files changed (original base)

```
 tests/v1/worker/test_rope_state.py                    | 182 ++++++++++++
 vllm/config/model.py                                  |   6 +
 vllm/model_executor/models/transformers/base.py       |   6 +-
 vllm/model_executor/models/transformers/multimodal.py |  10 +-
 vllm/transformers_utils/config.py                     |  45 ++++
 vllm/v1/worker/gpu/mm/rope.py                         |   7 +-
 vllm/v1/worker/gpu_model_runner.py                    |  14 +-
```

`git diff --check` clean; ruff check/format clean (pinned ruff 0.14.0, pre-commit not installable offline).

### Test matrix (original base)

| Test | Result |
|---|---|
| focused regression test | FAIL on main (`assert 3 == 4`) → PASS with fix |
| mrope executor suites (keye, keye_vl1_5, paddleocr_vl) | 12/12 PASS |
| kernels test_mrope.py | 6 pass / 48 fail — all hub-config downloads blocked offline; identical on pristine main (control) → environmental |
| transformers backend tests (test_backend.py, test_layer_registry.py) | 8 pass / 16 fail / 1 skip — all hub-weight downloads offline; `test_mla` control fails identically on pristine main → environmental |
| Hunyuan parser suites | not runnable offline (hub fetch at collection) |
| ruff on changed files | clean |
| git diff --check | clean |
| real W7900 E2E | PASS (4/4 × 4 mode variants) |

### Multi-GPU status

TP=1 tested (PASS). TP>1 NOT TESTED LOCALLY. PP>1 NOT TESTED LOCALLY.

## Part 2 — latest-main addendum (rebase to `1dc464d426`, 2026-08-30)

### Rebase

`OLD_BASE_SHA = b5707bf994cb968adfc7a29fbb80b0522f53f38d` → `NEW_UPSTREAM_MAIN_SHA = 1dc464d42681d22f38caf1fdc1eb632dc4421c45` (5 commits; none touch the six candidate production files — #54346 touches only native Qwen2.5-VL/Qwen3-VL model files). The candidate reapplied cleanly (`git apply --check` clean) and stays uncommitted on the local branch. No compiled-code drift in the 5 commits, so the prebuilt kernels remain valid. Updated patch: [diff/hunyuanocr-vllm-candidate.patch](diff/hunyuanocr-vllm-candidate.patch) (9 files, 510+/9−, now including two new focused tests); the original-base patch is preserved as [diff/hunyuanocr-vllm-candidate-b5707bf994.patch](diff/hunyuanocr-vllm-candidate-b5707bf994.patch).

### Duplicate-work search

GitHub PR/issue search (read-only) for HunyuanOCR, HunYuanVL, mrope_section, num_mrope_axes, xdrope_section, MLAFuser/head_size, transformers backend MLA, get_rope_index video_grid_thw, "4 multimodal RoPE channels": **no open or merged PR implements any of the three fixes**. Nearest neighbors, all disjoint: #48725 (spec-decode headroom for mrope/xdrope buffers), #49744 (fused QK-norm + mRoPE), #12067 (2025 DeepSeek head_size hardcode removal — different subsystem), Hunyuan A13B tool/reasoning parsers (#52658, #52133). Old HunyuanOCR issues #29598/#40165 are unrelated failure modes. No STOP condition.

### Root-cause reconfirmation on `1dc464d426`

RC1: pristine latest main fails identically — `ValueError: Expected 4 multimodal RoPE channels, got position_ids with shape (3, 1, 8192)` ([latest-main/run-baseline-latest-main.log](latest-main/run-baseline-latest-main.log)); live config probe: `mrope_section=[16,16,16,16]` (4 axes), sections sum 64 = `head_dim//2`, `uses_mrope=True`, `uses_xdrope_dim=0`, vestigial `kv_lora_rank=512/qk_nope=128/qk_rope=64` present ([latest-main/config-probe-latest-main.log](latest-main/config-probe-latest-main.log)). RC2/RC3: pristine `base.py` still gates the head-size override on bare `kv_lora_rank is not None`; pristine `multimodal.py` still passes both grid kwargs unconditionally (verified via `git show HEAD:…`). No root cause was invalidated by the rebase.

### Two additional focused regression tests (mandatory set now complete)

- `tests/models/transformers/test_create_attention_instances.py` — RC2: `test_vestigial_kv_lora_rank_keeps_gqa_head_size` (real `HunYuanVLTextConfig` with vestigial MLA fields, no `MLAFuser`; unbound `Base.create_attention_instances` over a duck-typed backend) **FAILs on pristine main (`assert 192 == 128`), PASSes with the fix**; `test_matched_mla_modules_still_override_head_size` guards that genuine MLA modules keep the `qk_nope+qk_rope` override on both.
- `tests/models/transformers/test_get_rope_index_kwargs.py` — RC3: `test_hunyuan_style_get_rope_index_gets_no_absent_grid_kwargs` (stub `get_rope_index` with HunYuanVL's signature) **FAILs on pristine main (`TypeError: … unexpected keyword argument 'video_grid_thw'`), PASSes with the fix**; `test_qwen_style_model_receives_present_grids_only` guards present-grid forwarding for Qwen-style signatures.

Full before/after matrix on latest main ([latest-main/tests-all-before-fix.log](latest-main/tests-all-before-fix.log), [latest-main/tests-focused-and-mrope-after-fix.log](latest-main/tests-focused-and-mrope-after-fix.log)): the three regression tests fail on pristine / pass with the candidate; the three guard tests pass on both.

### Latest-main test matrix

| Test | Result |
|---|---|
| focused: rope_state (2) + create_attention_instances (2) + get_rope_index kwargs (2) | 6/6 PASS with candidate |
| surrounding mrope executor suites incl. new upstream `test_qwen3_asr_mrope.py` | 17/17 PASS |
| kernels test_mrope.py + transformers backend tests | failure set byte-identical between candidate and pristine control (64 failed / 14 passed / 1 skipped / 1 error each) → all pre-existing environmental (hub downloads offline); diff of FAILED lines: identical |
| ruff check + format (9 files) | clean |
| git diff --check | clean |

### Latest-main W7900 E2E (transformers 5.15.1, TP=1, pinned revision)

| Mode | 1 simple | 2 document | 3 ×3 determinism | 4 portrait | doc hash |
|---|---|---|---|---|---|
| enforce_eager | PASS | PASS | PASS | PASS | 6cd5fba4cdb2f135 |
| default + piecewise cudagraphs | PASS | PASS | PASS | PASS | 6cd5fba4cdb2f135 |
| V1 runner | PASS | PASS | PASS | PASS | 6cd5fba4cdb2f135 |

The document token hash is identical to the original-base runs — cross-base determinism. None of the three error signatures (3-vs-4 RoPE, `[-1,16,192]` reshape, `video_grid_thw` TypeError) occurs. FULL cudagraphs remain unsupported (same pre-existing ROCm/HF rotary capture limitation; not claimed).

### Recommended upstream split

**Two PRs.** PR-A: dynamic M-RoPE axis count + `get_rope_index` grid-kwargs compatibility (7 files, +386/−8) — one narrative (the positions path for XD-RoPE models on the Transformers backend), lands first. PR-B: Transformers-backend MLA head-size false-positive gate (2 files, +124/−1) — self-contained attention-instantiation fix with its own test. Rationale: separate subsystem ownership under CODEOWNERS (@njhill v1/worker vs @hmellor transformers backend vs /vllm/config owners), independent landability with no intermediate regression (each PR strictly reduces failures; with only PR-A the model still fails at the reshape — same as today, never worse), isolated reviewer risk (PR-B touches MLA-adjacent logic that DeepSeek-lineage reviewers will want to scrutinize separately), clean per-PR test boundaries. HunyuanOCR is fully restored only when both land; each PR should reference the other.

### Proposed upstream PR titles (text only)

1. `[Bugfix] Derive multimodal RoPE axis count from config for the Transformers backend (4-axis XD-RoPE, e.g. HunyuanOCR)`
2. `[Bugfix] Transformers backend: don't apply the MLA head-size override without MLA attention modules`

### Remaining known limitations

FULL-CUDA-graph capture on ROCm fails inside the HF rotary's lazy cos/sin growth (pre-existing; piecewise + eager work). transformers 5.13.0 itself garbles this workload (vLLM matches it token-for-token; 5.15.1 correct) — HF-side. Hub-dependent vLLM tests cannot run on this offline host (proven identical on pristine main). TP>1/PP>1 NOT TESTED LOCALLY.
