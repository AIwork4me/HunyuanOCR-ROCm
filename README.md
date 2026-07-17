# HunyuanOCR-ROCm

> Precision-aligned AMD ROCm port of [HunyuanOCR-1.5](https://github.com/Tencent-Hunyuan/HunyuanOCR) — evaluated on OmniDocBench v1.6, across three inference backends, on AMD gfx1100 (RDNA3).

[![OmniDocBench v1.6](https://img.shields.io/badge/OmniDocBench-v1.6-blue)](https://github.com/opendatalab/OmniDocBench)
[![vLLM Overall](https://img.shields.io/badge/vLLM%20canary-94.81-green)](reports/canary-baseline.md)
[![llama.cpp Full](https://img.shields.io/badge/llama.cpp%20full%201651-92.09-yellow)](reports/project-stage-summary.md)
[![License](https://img.shields.io/badge/license-mixed%20(see%20NOTICE)-blue)](NOTICE)
[![Weights: Hunyuan Community License](https://img.shields.io/badge/weights-Hunyuan%20Community%20License-orange)](NOTICE)

## Results

Three inference backends on AMD gfx1100 (RDNA3, 48 GB ×4, ROCm 7.2), bf16, OmniDocBench v1.6:

| Backend | Canary (148 pages) | Full (1651 pages) | formula CDM | >14k ViT stability | Speed |
|---|---|---|---|---|---|
| **vLLM** 0.16.1 (Flash-Attn) | **94.81** | — | 96.48 | ✅ (capped 3.4M) | ~6 s/page |
| transformers 5.13 (SDPA) | 94.11 | — | 94.25 | ❌ NaN above 14.2k | ~180 s/page |
| **llama.cpp** (C++ GGML, BF16 GGUF) | 93.33 | **92.09** | 89.64 | ✅ **(uncapped)** | **~1.4 s/page** |
| Upstream (reported) | 94.74 | 94.74 | — | — | — |

**Key findings:**
- **llama.cpp is the fastest and most stable backend on gfx1100**, and the only one that runs at **full resolution** (no pixel cap). Its C++ ViT is deterministic at >14k vision tokens — the >14k NaN/non-determinism only affects the transformers SDPA path ([ROCm issue #6416](https://github.com/ROCm/ROCm/issues/6416)).
- The **formula CDM gap** (~5.65 pts on the canary) is from **inference-engine-level generation differences**, not resolution, streaming, or post-processing — confirmed by systematic ablation ([analysis](docs/tencent-114-followup3-draft.md)).
- The **>14k ViT instability** is a sharp threshold (~14,200 patches) in the ROCm PyTorch SDPA kernel; it does **not** affect vLLM's Flash-Attention or llama.cpp's C++ GGML paths.

## Quick start (llama.cpp, recommended)

### Build llama.cpp with HIP on gfx1100

```bash
git clone https://github.com/ggml-org/llama.cpp.git
cd llama.cpp
HIPCXX=/opt/rocm/llvm/bin/clang HIP_PATH=/opt/rocm \
cmake -S . -B build -DGGML_HIP=ON -DGPU_TARGETS=gfx1100 \
  -DGGML_HIP_ROCWMMA_FATTN=ON -DCMAKE_BUILD_TYPE=Release -DLLAMA_CURL=ON
cmake --build build --config Release -j$(nproc) --target llama-server
```

### Download the BF16 GGUF

```bash
huggingface-cli download ggml-org/HunyuanOCR-GGUF \
  HunyuanOCR-bf16.gguf mmproj-HunyuanOCR-bf16.gguf \
  --local-dir ./HunyuanOCR-GGUF
```

### Run inference

```bash
./build/bin/llama-server \
  --model ./HunyuanOCR-GGUF/HunyuanOCR-bf16.gguf \
  --mmproj ./HunyuanOCR-GGUF/mmproj-HunyuanOCR-bf16.gguf \
  --host 0.0.0.0 --port 8080 --alias HYVL \
  -ngl 999 -c 65536 -n 32768
```

### Evaluate on OmniDocBench v1.6

```bash
git clone https://github.com/AIwork4me/HunyuanOCR-ROCm.git
cd HunyuanOCR-ROCm
pip install -e . --no-deps && pip install pillow tqdm pyyaml requests openai

# run predictions (one server per GPU for throughput)
python scripts/run_phase2_vllm.py \
  --gt-json /path/to/OmniDocBench.json \
  --images-dir /path/to/images \
  --pred-dir ./predictions \
  --ports 8080 --model HYVL --concurrency 8

# score (requires OmniDocBench scorer)
python scripts/score_predictions.py \
  --pred-dir ./predictions \
  --gt-json /path/to/OmniDocBench.json
```

## Architecture

```
HunyuanOCR-ROCm/
├── src/hunyuan_ocr/            # frozen decoding contract + shared post-processors
│   ├── contract.py             # prompt, sampling, image config (FROZEN)
│   ├── tasks.py                # 12 task prompts (verbatim port from upstream)
│   ├── postprocess.py          # clean_repeated_substrings + process_one (verbatim port)
│   ├── omnidocbench.py         # dataset iteration + prediction filename mapping
│   ├── scoring.py              # OmniDocBench config writer + scorer + result parser
│   └── backends/
│       ├── transformers.py     # Phase 1: transformers backend (oracle)
│       ├── vllm_client.py      # Phase 2: vLLM OpenAI-compatible client
│       └── (llamacpp reuses vllm_client — llama-server is OAI-compatible)
├── scripts/
│   ├── run_phase1_transformers.py   # multi-GPU transformers driver
│   ├── run_phase2_vllm.py          # multi-server vLLM/llama.cpp driver (resumable)
│   ├── serve_vllm.sh               # start a vLLM server (HIP, compiled, tuned)
│   ├── score_predictions.py        # OmniDocBench scoring wrapper
│   └── regression_canary.py        # 150-page canary regression oracle
├── eval/configs/               # OmniDocBench eval config template
├── reports/                    # canary BASELINE + project stage summary
├── docs/superpowers/           # design spec + implementation plans
└── Makefile                    # demo, eval-linux, eval-canary, score targets
```

## gfx1100 adaptations

| Adaptation | Reason | File |
|---|---|---|
| ViT pixel cap (3.4M, `GFX1100_VIT_MAX_PIXELS`) | ROCm SDPA ViT NaN above ~14.2k tokens ([#6416](https://github.com/ROCm/ROCm/issues/6416)) | `backends/transformers.py` |
| SDPA attention (vs eager) | ~1.4× faster on RDNA3 | `backends/transformers.py` |
| torch.compile for vLLM | ~28× decode speedup (2→150 tok/s) | `scripts/serve_vllm.sh` |
| `-c 65536` for llama-server | large pages overflow 32768 ctx at full res | `scripts/serve_vllm.sh` |

## Issues filed

- **[ROCm/ROCm#6416](https://github.com/ROCm/ROCm/issues/6416)** — bf16 ViT forward non-determinism + NaN above ~14.3k tokens on gfx1100.
- **[Tencent-Hunyuan/HunyuanOCR#114](https://github.com/Tencent-Hunyuan/HunyuanOCR/issues/114)** — recommended max resolution / vision-token budget; three-backend comparison data; formula CDM gap analysis.

## License

This repository is **mixed-licensed** (see [NOTICE](NOTICE)):

1. **Original packaging/tooling** (drivers, `runner.py`, `validation.py`, `scoring.py`, `omnidocbench.py`): **Apache-2.0** ([LICENSE](LICENSE), [LICENSES/Apache-2.0.txt](LICENSES/Apache-2.0.txt)).
2. **Code ported from HunyuanOCR** (`contract.py`, `tasks.py`, `postprocess.py`, `backends/*`): upstream-derived portions under the **Tencent Hunyuan Community License** ([LICENSES/Tencent-Hunyuan-Community-License.txt](LICENSES/Tencent-Hunyuan-Community-License.txt)) — *not* Apache.
3. **HunyuanOCR model weights** (`tencent/HunyuanOCR`, `ggml-org/HunyuanOCR-GGUF`): **Tencent Hunyuan Community License** — **not OSI Open Source**; excludes EU/UK/KR; "Powered by Tencent Hunyuan" is *encouraged*, not required.
4. **llama.cpp**: MIT. **vLLM**: Apache-2.0.

Tencent is not affiliated with, sponsoring, or endorsing this project.

## Acknowledgements

- [Tencent HunyuanOCR](https://github.com/Tencent-Hunyuan/HunyuanOCR) — the model.
- [ggml-org/llama.cpp](https://github.com/ggml-org/llama.cpp) — the inference engine.
- [vLLM](https://github.com/vllm-project/vllm) — the serving backend.
- [OmniDocBench](https://github.com/opendatalab/OmniDocBench) — the evaluation benchmark.
- [ROCm](https://github.com/ROCm/ROCm) — the AMD compute platform.
