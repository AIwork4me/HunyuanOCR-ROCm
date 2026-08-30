# RC3 upstream PR draft — TEXT ONLY, nothing submitted

**Title:** `[Bugfix][Multimodal] Forward only present RoPE grid kwargs`

## Purpose

The Transformers multimodal backend currently forwards `image_grid_thw` and `video_grid_thw` to HF `get_rope_index` even when the corresponding modality is absent.

Some Transformers implementations do not accept every grid keyword. For example, HunYuanVL does not accept `video_grid_thw`, so an image-only HunyuanOCR request fails when vLLM explicitly passes `video_grid_thw=None`:

```
TypeError: HunYuanVLModel.get_rope_index() got an unexpected keyword argument 'video_grid_thw'
```

No existing PR addressing this was found.

## Fix

Only forward grid kwargs for modalities that are actually present (+7/−2 lines in `vllm/model_executor/models/transformers/multimodal.py`). Generic and introduces no model- or hardware-specific path.

## Tests

`tests/models/transformers/test_get_rope_index_kwargs.py` (CPU-only, no weights, 95 lines):

- Focused regression `test_omits_absent_grid_kwargs` — HunYuanVL-like signature without `video_grid_thw`, image-only request: **pristine main FAIL** (`TypeError: … unexpected keyword argument 'video_grid_thw'`) → **candidate PASS**; also asserts `image_grid_thw` and `mm_token_type_ids` are forwarded with correct values and positions/delta come back well-formed.
- Positive guard `test_forwards_present_image_grid` — Qwen-style signature with sentinel defaults: the present image grid is passed with the real tensor; the absent video grid is not passed at all (distinguishing omitted vs passed-None).

Command: `pytest tests/models/transformers/test_get_rope_index_kwargs.py` — pristine `2c7d7dd64a`: 2 failed (regression TypeError + sentinel baseline) → candidate: 2 passed. Logs: `rc3-final-before.log` / `rc3-final-after.log`.

Surrounding runnable M-RoPE suites: `test_keye_mrope`, `test_keye_vl1_5_mrope`, `test_paddleocr_vl_mrope`, `test_qwen3_asr_mrope` — 11/11 PASS. Hub-dependent suites (e.g. `test_backend.py`) cannot run on the validation host (offline) and fail identically on pristine main — environmental.

W7900/HunyuanOCR boundary (1× Radeon PRO W7900D, gfx1100, TP=1, pinned `tencent/HunyuanOCR` revision): with two other known #53615 migration gaps applied as LOCAL prerequisite patches (m-RoPE axis count, MLA head-size heuristic — separate upcoming fixes, NOT part of this PR), the real workload fails at exactly this boundary before the change (`TypeError: … 'video_grid_thw'`, `rc3-boundary-before.log`) and passes it after the change, completing correct greedy OCR (`rc3-boundary-after.log`). The patch itself is generic and hardware-independent; the bug was found while validating HunyuanOCR after #53615.

## Evidence

Full reproduction package: `validation-53615/` in https://github.com/AIwork4me/HunyuanOCR-ROCm/tree/validation-53615-evidence (see the immutable commit URL recorded in the final report when submitting).

## AI assistance

ZCode Agent assisted with implementation and testing; the human author reviewed the diff and validation evidence and is fully responsible for the change.
