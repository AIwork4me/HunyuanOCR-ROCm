# PATCHES — Upstream-derived Code & Dependency Provenance

HunyuanOCR-ROCm ports algorithm code from Tencent HunyuanOCR and depends on a
specific `llama.cpp` build. This file records the provenance of every
upstream-derived file and pinned upstream dependency so the diff against upstream
stays auditable. There are **no diff patches against a vendored runtime** here
(unlike repos that carry `patches/<runtime>/`); the items below are
upstream-derived source files kept verbatim where possible.

## Upstream-derived source files (ported from Tencent HunyuanOCR)

Upstream: https://github.com/Tencent-Hunyuan/HunyuanOCR
Upstream revision (REPRO.yaml): `de8f10ad2f00a0cefd790b526de8a65dcfdb3205` (main)

| file | upstream origin | license | lint policy |
|---|---|---|---|
| `src/hunyuan_ocr/postprocess.py` | upstream doc_parse / postprocess (clean_repeated_substrings, process_one) | Tencent Hunyuan Community License | verbatim: lint-only, never reformatted (pyproject `per-file-ignores` + `format.exclude`) |
| `src/hunyuan_ocr/tasks.py` | upstream task definitions | Tencent Hunyuan Community License | verbatim (lint-only) |
| `src/hunyuan_ocr/contract.py` | upstream inference contract/prompt | Tencent Hunyuan Community License | verbatim (lint-only) |
| `src/hunyuan_ocr/backends/*` | upstream inference paths (adapted to ROCm/OpenAI client) | Tencent Hunyuan Community License | normal lint |

Original packaging/tooling (everything else in `src/hunyuan_ocr/`) is Apache-2.0
(AIwork4me). See `NOTICE` and `LICENSES/`.

## Pinned upstream runtime dependency

| dependency | upstream repo | pinned commit | purpose |
|---|---|---|---|
| llama.cpp | https://github.com/ggml-org/llama.cpp | `a320cbfcb7056b7b81fb854d97fe01d0ea77c4b5` (2026-07-16, GitHub API) | HIP server for HunyuanOCR GGUF; contains `tools/mtmd/models/hunyuanvl.cpp` + `src/models/hunyuan-vl.cpp` |

## Weight artifacts (cross-checked byte-for-byte against official repos)

From `REPRO.yaml` (sha256 recomputed in-repo, cross-checked via hf-mirror.com):

| artifact | sha256 | upstream match |
|---|---|---|
| HunyuanOCR-bf16.gguf | `a160215620dbd0ab43ec6faa28259654fd24c929953aa97c765176f7c0363217` | == ggml-org/HunyuanOCR-GGUF LFS oid |
| mmproj-HunyuanOCR-bf16.gguf | `46401739a91d0778d86369bb952db685b215512d61a941c3b859f337f6014fcd` | == ggml-org/HunyuanOCR-GGUF LFS oid |
| HunyuanOCR safetensors | `632a1e082c4dd5a3284cf1ffcdba2fdaa06f435762c58c2f34aff0f3bd6c0249` | == tencent/HunyuanOCR LFS oid |
| config.json | `cc34ab90d0b873a1832c06e0f3fe127b47d7f390e8fda19445e4144068ed2af9` | == tencent/HunyuanOCR content sha |

## Removal condition

The upstream-derived files are required for correct HunyuanOCR post-processing
and cannot be removed while the model is supported. If upstream merges equivalent
ROCm support and this repo switches to consuming upstream directly, these files
are deleted and this section updated.
