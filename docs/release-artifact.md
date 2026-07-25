# Benchmark release artifact

Every benchmark release ships a **self-contained evidence bundle** so a result is
not just a number — it is reproducible from recorded inputs. This document defines
the standard layout and where each field comes from.

## Layout

```
benchmark-artifacts/
├── README.md                  # what this bundle is + how to reproduce
├── run_manifest.json          # the run's terminal state (from the prediction dir)
├── metrics.json               # parsed OmniDocBench metrics for the run
├── environment.json           # ROCm / torch / transformers / vLLM / gpu versions
├── REPRO.yaml  # the pinned inputs (copied from the repo at release time)
├── commands.txt               # the exact predict / validate / score commands used
└── checksums.sha256           # sha256 of the above files
```

## Provenance rules (no hand-typed numbers)

- **`metrics.json`** — produced by `hunyuan_ocr.scoring.parse_run_summary` from
  the scorer's `run_summary.json`. Never edited by hand.
- **`environment.json`** — the `env`/`platform` block written by
  `hunyuan_ocr.runner.write_run_manifest` (best-effort ROCm/torch/transformers/vLLM
  versions).
- **`REPRO.yaml`** — the repo's lock **at the release commit**
  (copied verbatim). It pins: the repo + llama.cpp + OmniDocBench commits, the
  GT/model SHA256 + LFS oids, the eval-config + canary-manifest hashes, and the
  Overall metric formula. This is the single source of truth for "which inputs
  produced this number."
- **`commands.txt`** — the redacted argv (`runner.safe_argv`) from the manifest;
  secret flags are redacted.
- **`checksums.sha256`** — `sha256sum` of every other file in the bundle, so the
  bundle is tamper-evident.

## Which inputs each field ties down

| Field | Ties down | Source |
|---|---|---|
| `REPRO.yaml` → `hunyuanocr_rocm.commit` | the benchmark-harness code | repo |
| `→ llama_cpp.commit` | the inference engine | ggml-org/llama.cpp |
| `→ omnidocbench.scorer_commit` | the scorer | opendatalab/OmniDocBench |
| `→ model.benchmark_artifact` / `current_remote_artifact` | the model weights (GGUF + safetensors + config) | tencent/HunyuanOCR, ggml-org/HunyuanOCR-GGUF |
| `→ omnidocbench.gt_json_*_sha256` | the dataset subset | sha256sum |
| `→ environment` | the runtime stack | sys / torch / rocm-smi |
| `run_manifest.json` | the run's terminal state (complete/failed/pending, conservation laws) | runner |

## Reproducing a bundle

```bash
# 1. check out the pinned harness + llama.cpp + scorer commits from the lock
# 2. verify weights against model.benchmark_artifact (sha256) in the lock
# 3. rebuild the canary byte-identically:
hunyuan-ocr canary materialize --full-gt <full GT> \
  --manifest eval/canary_148.manifest.json --out <canary GT>
# 4. run the commands in commands.txt
# 5. compare your metrics.json / run_manifest.json to the bundle's
```

> A formal result only changes when a new bundle exists (new prediction dir + new
> manifest + new metrics). Editing a number by hand is not a release.
