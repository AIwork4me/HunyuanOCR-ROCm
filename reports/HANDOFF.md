# HunyuanOCR-ROCm — Comprehensive Handoff Document

> **Historical — 2026-07-16.** Retained as experimental evidence; `README.md` is
> the single source of current status. Some conclusions in this file read
> stronger than the evidence now supports (see README). Machine-local paths
> (`/root/...`, `/workspace/...`) are factual cross-session evidence, not user
> repro paths — use `scripts/reproduce_*.sh` + `reproducibility.lock.yaml`.

**Date:** 2026-07-16 · **Repo:** https://github.com/AIwork4me/HunyuanOCR-ROCm · **State:** snapshot 2026-07-16; see git history for current state.

> Every conclusion below has a specific evidence reference (file path, command, or data point) and reproducible steps. If you pick this up in a new session, start here.

---

## 1. Project Goal

Run Tencent **HunyuanOCR-1.5** (~1B VLM) on **AMD gfx1100 (RDNA3, ROCm 7.2)**, precision-aligned with the original on **OmniDocBench v1.6** (1651 pages), across three inference backends: transformers → vLLM → llama.cpp. Standalone first; integrate into OmniDocBench-AMD platform later (Phase 4).

## 2. Environment (exact, reproducible)

| Component | Value | How to verify |
|---|---|---|
| GPU | 4× AMD gfx1100 (RDNA3, 48 GB each) | `rocm-smi --showproductname` |
| ROCm | hip 7.2.53211-e1a6bc5663; ROCk kernel 6.14.14 | `rocm-smi` / `rocminfo` |
| torch | 2.9.1+gitff65f5b (ROCm build, from /opt/venv) | `python -c "import torch; print(torch.__version__, torch.version.hip)"` |
| transformers | 5.13.0 (isolated venv at /root/hunyuanocr-venvs/transformers) | `pip show transformers` |
| vLLM | 0.16.1.dev0+g89a77b108 (/opt/venv) | `pip show vllm` |
| llama.cpp | master, ggml 0.16.0, commit a320cbf (built at /root/llama.cpp) | `/root/llama.cpp/build/bin/llama-server --version` |
| GGUF weights | ggml-org/HunyuanOCR-GGUF BF16 (at /root/models/HunyuanOCR-gguf/) | `ls /root/models/HunyuanOCR-gguf/*.gguf` |
| HF weights | tencent/HunyuanOCR (at /root/models/HunyuanOCR) | `ls /root/models/HunyuanOCR/model.safetensors` |
| OmniDocBench | v1.6, 1651 pages (at /workspace/OmniDocBench_data/) | `ls /workspace/OmniDocBench_data/OmniDocBench.json` |
| OmniDocBench scorer | /root/ocr-eval/OmniDocBench, venv 3.11 | `ls /root/ocr-eval/OmniDocBench/.venv/bin/python` |
| Upstream source | /root/HunyuanOCR-src (cloned from Tencent-Hunyuan/HunyuanOCR) | `ls /root/HunyuanOCR-src/inference/` |

## 3. Evaluation Results (with evidence + repro)

### 3.1 Three-backend canary-148

The `OmniDocBench_150.json` subset has 148 pages (despite the name). All three backends completed 148/148, 0 errors.

| Backend | Overall | text EditDist | formula CDM | table TEDS | order EditDist | Resolution |
|---|---|---|---|---|---|---|
| **vLLM** 0.16.1 | **94.81** | 0.0514 | 0.9648 | 0.9308 | 0.1135 | capped 3.4M |
| transformers 5.13 | 94.11 | 0.0437 | 0.9425 | 0.9246 | 0.1184 | capped 3.4M |
| **llama.cpp** BF16 | 93.33 | 0.0512 | 0.9083 | 0.9429 | 0.1270 | uncapped (full) |
| llama.cpp BF16 (capped 3.4M) | 92.50 | 0.0540 | 0.8897 | 0.9392 | 0.1342 | capped 3.4M |

**Evidence:** prediction dirs at `/root/hunyuanocr-results/{canary-150,vllm-canary-150,llamacpp-canary-150,llamacpp-canary-capped-150}/`. Score results at `/root/ocr-eval/OmniDocBench/result/{canary-150,vllm-canary-150,llamacpp-canary-150,llamacpp-canary-capped-150}_quick_match_run_summary.json`.

**Reproduce scoring:**
```bash
cd /workspace/HunyuanOCR-ROCm
/root/hunyuanocr-venvs/transformers/bin/python scripts/score_predictions.py \
  --pred-dir /root/hunyuanocr-results/<DIR> \
  --gt-json /workspace/OmniDocBench_data/OmniDocBench_150.json \
  --label <LABEL>
```

### 3.2 llama.cpp full-set (1651 pages)

**Overall: 92.09** (text 0.0467 / CDM 0.8964 / TEDS 0.9130 / order 0.1375). 1651/1651 pages, 0 errors.

**Evidence:** `/root/hunyuanocr-results/llamacpp-full-1651/` (1651 .md files). Score at `/root/ocr-eval/OmniDocBench/result/llamacpp-full-1651_quick_match_run_summary.json`. Two independent scoring runs confirmed 92.09.

**Reproduce:**
```bash
# Start 4 llama-servers (one/GPU, ctx=65536):
for g in 0 1 2 3; do
  CUDA_VISIBLE_DEVICES=$g /root/llama.cpp/build/bin/llama-server \
    --model /root/models/HunyuanOCR-gguf/HunyuanOCR-bf16.gguf \
    --mmproj /root/models/HunyuanOCR-gguf/mmproj-HunyuanOCR-bf16.gguf \
    --host 127.0.0.1 --port $((8081+g)) --alias HYVL -ngl 999 -c 65536 -n 32768 &
done
# Run predictions:
cd /workspace/HunyuanOCR-ROCm
python scripts/run_phase2_vllm.py --gt-json /workspace/OmniDocBench_data/OmniDocBench.json \
  --images-dir /workspace/OmniDocBench_data/images \
  --pred-dir /root/hunyuanocr-results/llamacpp-full-1651 \
  --ports 8081,8082,8083,8084 --model HYVL --concurrency 16
# Score:
python scripts/score_predictions.py --pred-dir /root/hunyuanocr-results/llamacpp-full-1651 \
  --gt-json /workspace/OmniDocBench_data/OmniDocBench.json --label llamacpp-full-1651
```

### 3.3 Note on vLLM full-set

vLLM full-set was attempted but **servers crashed** (compiled-mode instability: 3 of 4 servers died mid-run, producing 780 ERROR pages → false score of 46.31). Cleaned + re-run also failed (servers died again). The vLLM canary (94.81) is the reliable vLLM number. The full-set vLLM number is **not available**.

---

## 4. Key Findings (each with evidence + repro)

### Finding 1: ROCm ViT >14k instability (transformers SDPA path)

**Claim:** The Hunyuan-ViT bf16 forward becomes non-deterministic + NaN above a sharp threshold of ~14,200–14,688 vision tokens, ONLY in the PyTorch/ROCm SDPA path.

**Evidence:**
- Per-layer `max()` grows: `[8, 15, 20, …, 376, …, 400, 11456]` — two spikes (layer 9: 20→376, layer 26: 400→11456).
- 3× identical `get_image_features` runs: ≤14,200 patches → `max|Δ|=0.000` (deterministic); 14,688 patches → `max|Δ|=9312` + intermittent NaN.
- End-to-end greedy on same image, 3 runs: `"I"`, `"(1,0),(1000,999)"`, coherent-but-wrong paragraph.
- Isolation: bare matmul, standalone SDPA, LayerNorm, single ViT block (random input) ALL deterministic at 14,688. LLM text-only at 4k tokens deterministic.
- fp32 forward: does NOT NaN (but ViT still grows to 13,465 max — growth is real, not precision).

**Reproduce:**
```python
# See ROCm issue #6416 for full repro script.
# Key: load tencent/HunyuanOCR bf16 sdpa, thumbnail image to 2304px (~14.7k patches),
# run get_image_features 3×, compare.
```

**Filed:** https://github.com/ROCm/ROCm/issues/6416 (OPEN)

### Finding 2: llama.cpp C++ ViT stable at full resolution

**Claim:** llama.cpp's C++ GGML ViT is fully deterministic at >14k tokens (same page that fails on transformers).

**Evidence:**
- Same sample page (1653×2339, ~15k tokens): 3× identical output (ALL IDENTICAL: True), 0 NaN.
- Output is correct OCR (matrix LaTeX matches transformers/vLLM).

**Reproduce:**
```bash
# Start llama-server, send the page 3× via the adapter:
cd /workspace/HunyuanOCR-ROCm
/root/hunyuanocr-venvs/transformers/bin/python -c "
import base64; from openai import OpenAI
c=OpenAI(api_key='EMPTY',base_url='http://127.0.0.1:8081/v1',timeout=600)
img='/workspace/OmniDocBench_data/images/page-d1561665-5359-42fe-920c-d6e3bff81953.png'
b64=base64.b64encode(open(img,'rb').read()).decode()
outs=[]
for _ in range(3):
  r=c.chat.completions.create(model='HYVL',max_tokens=256,temperature=0,messages=[{'role':'user','content':[{'type':'image_url','image_url':{'url':f'data:image/jpeg;base64,{b64}'}},{'type':'text','text':'提取文档图片中正文的所有信息用markdown格式表示，其中页眉、页脚部分忽略，表格用html格式表达，文档中公式用latex格式表示，按照阅读顺序组织进行解析。'}]}])
  outs.append(r.choices[0].message.content)
print('IDENTICAL:', len(set(outs))==1)
"
```

### Finding 3: formula CDM gap is inference-engine-level

**Claim:** The formula CDM gap (llama.cpp 90.83% vs vLLM 96.48% on canary) is NOT from resolution, streaming, post-processing, or systematic formula loss. It's from inference-engine kernel differences.

**Evidence (4 hypotheses ruled out):**
1. ❌ Streaming parsing: stream vs non-stream output identical (same content).
2. ❌ Systematic formula loss: ~60% of formula pages have same `$$` count; ~40% have mixed deltas (not one-sided).
3. ❌ `\dots` vs `\ldots`: both backends use `\ldots` dominantly; difference <5 occurrences.
4. ❌ Resolution cap: **REFUTED** — capped llama.cpp is WORSE on ALL metrics (CDM 88.97% capped vs 90.83% uncapped). Full resolution is strictly better.

**Reproduce the cap test:**
```bash
python scripts/run_phase2_vllm.py --gt-json .../OmniDocBench_150.json --images-dir .../images \
  --pred-dir /root/hunyuanocr-results/llamacpp-canary-capped-150 \
  --ports 8081,8082,8083,8084 --model HYVL --concurrency 16 --max-pixels 3400000
python scripts/score_predictions.py --pred-dir /root/hunyuanocr-results/llamacpp-canary-capped-150 ...
# Result: Overall 92.50 (vs uncapped 93.33). CDM 88.97 (vs uncapped 90.83). Cap HURTS.
```

### Finding 4: Throughput characteristics

| Backend | Mode | Decode speed | Notes |
|---|---|---|---|
| transformers | SDPA bf16 | ~5.5 tok/s | full set ~40h, impractical |
| vLLM | eager | ~2 tok/s | unfused kernels; too slow |
| vLLM | torch.compile | ~150 tok/s/server | 28× faster; compiles ~140s; servers unstable under sustained load |
| llama.cpp | BF16 GGUF HIP | ~22 tok/s/slot (4 slots) | 1.4s/page warm; most stable |

**Key tuning findings:**
- vLLM `--max-model-len` must be ≥ input(~13k) + max_tokens(32768) ≈ 46k. Setting 32768 silently 400-rejects all requests. Use 65536.
- vLLM `--enforce-eager` avoids torch.compile stalls (earlier "stall" was uncapped 65k-patch profiling, not compile).
- llama-server `-c 32768` overflows on large uncapped images; use `-c 65536`.

---

## 5. Issues Filed (all OPEN)

| Issue | URL | Content |
|---|---|---|
| ROCm #6416 | https://github.com/ROCm/ROCm/issues/6416 | bf16 ViT forward non-determinism + NaN above ~14.3k tokens on gfx1100; full repro + threshold table + isolation |
| Tencent #114 | https://github.com/Tencent-Hunyuan/HunyuanOCR/issues/114 | 3 comments: (1) canary dual-backend data, (2) three-backend canary + llama.cpp full 1651, (3) formula CDM gap systematic analysis |

---

## 6. gfx1100 Adaptations (in code)

| Adaptation | File | Env var | Default |
|---|---|---|---|
| ViT pixel cap (3.4M) | `src/hunyuan_ocr/backends/transformers.py` | `HUNYUANOCR_VIT_MAX_PIXELS` | 3400000 |
| SDPA attention (vs eager) | `src/hunyuan_ocr/backends/transformers.py` | `HUNYUANOCR_ATTN` | sdpa |
| vLLM torch.compile | `scripts/serve_vllm.sh` | `ENFORCE_EAGER` | 0 (compile on) |
| vLLM max-model-len | `scripts/serve_vllm.sh` | `MAX_MODEL_LEN` | 65536 |
| Capped preprocessor dir | `/root/models/HunyuanOCR-vllm` (symlinks + capped preprocessor_config.json) | — | max_pixels=3400000 |

---

## 7. Known Issues / Limitations

1. **vLLM servers crash under sustained load** (compiled mode, 3+ hours) — the vLLM full-set is NOT available. Canary (94.81) is the reliable number.
2. **transformers full-set impractical** (~5.5 tok/s → ~40h for 1651 pages). Canary (94.11) is the reliable number.
3. **llama.cpp formula CDM gap** (~5.65 pts vs vLLM on canary) — inference-engine-level, not a single bug. Consistent with Tencent's "not yet aligned" note.
4. **ROCm leaked VRAM** after killing GPU processes — requires thorough cleanup (kill all by PID via script, not inline pkill which self-kills the bash).
5. **llama-server ctx overflow** — large uncapped images need `-c 65536` (not 32768).

---

## 8. Next Steps

### Phase 4: Integrate into OmniDocBench-AMD

1. Wrap the backends behind `contracts/adapter.md` (`run_adapter(img_dir, out_dir, *, platform, config)`).
2. Add `hub/registry.yaml` entry in the platform repo (`/workspace/omnidocbench-amd`):
   ```yaml
   - model_id: hunyuanocr-1.5
     repo: AIwork4me/HunyuanOCR-ROCm
     platforms:
       linux-rocm: {badge: community, overall: 92.09}
   ```
3. Run `scripts/check_conformance.py` on the adapter.
4. Set badge to `community` (license posture unresolved).

### Other possible work

- **vLLM full-set**: retry with more stable server config (lower mem-util, fewer servers, shorter run). The vLLM canary (94.81) suggests full-set would be ~93-94.
- **formula CDM investigation**: diff the per-page CDM scores between llama.cpp and vLLM to find the specific pages/formula types where they diverge most. Might reveal a fixable preprocessing issue.
- **transformers full-set**: impractical via transformers; if needed, run via vLLM as the "transformers-equivalent" reference.

---

## 9. Key File Locations

| What | Path |
|---|---|
| **GitHub repo** | https://github.com/AIwork4me/HunyuanOCR-ROCm |
| **Platform repo** | /workspace/omnidocbench-amd |
| **Code** | /workspace/HunyuanOCR-ROCm/src/hunyuan_ocr/ |
| **Design spec** | docs/superpowers/specs/2026-07-15-hunyuanocr-rocm-design.md |
| **Phase 1 plan** | docs/superpowers/plans/2026-07-15-hunyuanocr-rocm-phase1-transformers.md |
| **Stage summary** | reports/project-stage-summary.md |
| **This handoff** | reports/HANDOFF.md |
| **Predictions** | /root/hunyuanocr-results/{canary-150,vllm-canary-150,llamacpp-canary-150,llamacpp-canary-capped-150,llamacpp-full-1651}/ |
| **Score results** | /root/ocr-eval/OmniDocBench/result/*_run_summary.json |
| **llama.cpp build** | /root/llama.cpp/build/bin/llama-server |
| **GGUF weights** | /root/models/HunyuanOCR-gguf/ (BF16: 1.08GB + mmproj 1.00GB) |
| **HF weights** | /root/models/HunyuanOCR/ (original) |
| **Upstream source** | /root/HunyuanOCR-src/ (cloned, for reference) |
| **Capped vLLM dir** | /root/models/HunyuanOCR-vllm/ (symlinks + capped preprocessor) |
| **Issue drafts** | docs/{rocm-issue-draft,tencent-114-followup*-draft}.md |

---

## 10. Memory (cross-session pointers)

- `hunyuanocr-rocm-vit-fullres-explosion.md` — the >14k ViT instability finding (memory).
- `amd-doc-parsing-zone-program.md` — the platform program (memory).
- `harness-background-vllm.md` — how to run GPU servers as background tasks (memory).
- `github-push-from-env.md` — how to push to GitHub from this env (memory).

---

*Powered by Tencent Hunyuan.*
