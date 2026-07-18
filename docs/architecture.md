# Architecture

> A 5-minute orientation for contributors. This project is **benchmark
> infrastructure** for evaluating HunyuanOCR-1.5 on AMD ROCm — not a model port.

## Pipeline

```
                OmniDocBench v1.6 (GT + images)
                              |
                              v
                   Evaluation Driver (run_inference.py
                     / hunyuan-ocr predict / transformers driver)
                              |
          +-------------------+-------------------+
          |                   |                   |
      llama.cpp            vLLM           Transformers
   (llama-server,         (OpenAI          (ROCm torch,
    GGUF, HIP)            API, HIP)        SDPA ViT)
          |                   |                   |
          +-------------------+-------------------+
                              |
                              v
                  Prediction artifacts (<stem>.md per page,
                     atomic, resumable, .run.lock'd)
                              |
                              v
                        Validator  (hunyuan-ocr validate)
                              |
                              v
                          Scorer   (hunyuan-ocr score;
                       OmniDocBench v1.6 scorer venv)
                              |
                              v
                Reproducibility Manifest (run_manifest.json +
                     reproducibility.lock.yaml — single source of truth)
```

Three backends produce predictions for the **same** page set; the downstream
validate → score → manifest path is shared and backend-agnostic.

## Module responsibilities

Everything lives in the importable `hunyuan_ocr` package (`src/`); `scripts/` are
thin wrappers so the CLI works from a checkout, and the package itself works from
a wheel install.

| Module | Responsibility | Depends on |
|---|---|---|
| `contract.py` | The **frozen decoding contract** — prompt, sampling, image config. Never changes without a re-baseline. | — |
| `tasks.py`, `postprocess.py` | Task prompts + `clean_repeated_substrings`/`process_one (**verbatim port** from upstream HunyuanOCR). | `contract` |
| `backends/vllm_client.py` | OpenAI-compatible per-image inference (llama-server / vLLM / any OAI). | `contract` |
| `backends/transformers.py` | ROCm-torch backend (Phase-1 oracle; SDPA ViT, pixel cap). | torch |
| `preflight.py` | Fail-fast input validation + sharding, **before** model load. | — |
| `endpoint_pool.py` | Circuit-breaking OpenAI endpoint pool (health, half-open probes). | — |
| `driver.py` | The predict orchestration: preflight → pool → dispatch → **crash-safe** manifest. Package-resident, wheel-runnable. | `preflight`, `endpoint_pool`, `runner`, `contract` |
| `runner.py` | Prediction integrity: atomic writes, resumability, structured errors, **run-manifest schema** (conservation laws), writer lock. | — |
| `validation.py` | Pre-score validation of a prediction dir (missing/empty/ERROR/partial/unresolved). | `runner` |
| `scoring.py` | OmniDocBench eval-config writer + scorer runner + result parser; private temp config dir. | OmniDocBench venv |
| `omnidocbench.py` | Dataset iteration + prediction filename mapping. | — |
| `canary.py` | Rebuild the 148-page canary byte-identically from the full GT + manifest. | — |
| `cli.py` | Unified `hunyuan-ocr` CLI (doctor / validate / manifest / canary / predict / score). | all |
| `ci/` | GPU-CI bridge: `poller.py` runs the real gfx1100 smoke for a commit status; `github.py` is the `gh api` boundary. | `gh` |

## Design principles

1. **CPU-installable core.** `pip install hunyuan-ocr` works on a plain CPU; GPU
   deps (torch, vLLM) are opt-in extras. The CLI, validation, scoring-wrapper, and
   CI poller never import torch.
2. **Atomic + resumable.** One `<stem>.md` per page written atomically; resumability
   skips only genuinely-complete pages. An exclusive `.run.lock` prevents two writers.
3. **One source of truth.** `reproducibility.lock.yaml` pins every input (commits,
   SHA256, env, metric formula); README results are cross-checked against it by
   `scripts/check_repo.py` (CI fails on drift).
4. **Evaluation-backed, not precision-aligned.** No same-page-set CUDA control
   exists; results are never described as a ROCm-vs-CUDA delta. See
   [benchmark-methodology.md](benchmark-methodology.md).

## Contributing a change

- Behavior change → add a CPU unit test (no torch, no GPU).
- Anything affecting scores (prompt/sampling/post-processing/scorer/metric) →
  re-baseline per [CONTRIBUTING.md](../CONTRIBUTING.md); never edit a formal
  number by hand.
- `pytest -q -m "not gpu"`, `ruff check .`, `python scripts/check_repo.py`,
  `reuse lint` must all pass before merge.
