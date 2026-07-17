**Update 3 — Formula CDM gap analysis: systematic debugging rules out resolution, streaming, and post-processing; inference-engine-level generation differences are the leading explanation**

Following up on the formula CDM gap we reported (llama.cpp 89.6% vs vLLM 96.5% on the full set), we conducted a systematic investigation to localize it. Here's what we found.

## The gap (apples-to-apples, same 148-page canary)

| Backend | Overall | text | formula CDM | table TEDS |
|---|---|---|---|---|
| vLLM (capped 3.4M) | 94.81 | 0.0514 | **0.9648** | 0.9308 |
| llama.cpp (uncapped) | 93.33 | 0.0512 | **0.9083** | 0.9429 |
| llama.cpp (capped 3.4M) | 92.50 | 0.0540 | **0.8897** | 0.9392 |

The formula CDM gap is **~5.65 points** (vLLM vs llama.cpp uncapped, same pages).

## What we ruled out

1. **Streaming parsing** — not the cause. llama-server's OpenAI SSE streaming format parses identically to vLLM's; streaming and non-streaming requests produce the same output.
2. **Systematic formula omission** — not the cause. Per-page formula block counts (`$$` pairs) match exactly on ~60% of formula-heavy pages; the remaining ~40% have mixed deltas (sometimes llama.cpp has more, sometimes fewer). Not one-sided.
3. **LaTeX token differences (`\dots` vs `\ldots`)** — negligible. Both backends use `\ldots` as the dominant form; the `\dots` frequency difference is <5 occurrences across 148 pages.
4. **Resolution cap** — **refuted (and the opposite is true)**. We ran the canary with llama.cpp CAPPED to 3.4M pixels (matching vLLM's resolution). The capped version scored **WORSE on every metric**, including formula CDM (88.97% vs 90.83% uncapped). Full resolution is strictly better — the cap removes fine-grained detail that helps formula recognition.

## Where the gap actually is

After ruling out the above, the leading explanation is **inference-engine-level generation differences**:

- llama.cpp (C++ GGML ViT + GGML LLM on HIP) and vLLM (Flash-Attn-Triton ViT + PyTorch LLM on ROCm) run the **same BF16 weights** through **different kernel implementations**. These produce slightly different intermediate values, which — for formula-heavy pages where the model must transcribe fine mathematical notation — accumulate into measurably different LaTeX output (e.g., spacing, delimiter choices, alignment formatting). These differences are invisible to a text comparison but affect the **CDM pixel-level render comparison**.

- This is consistent with your "accuracy not yet aligned" note: the gap is not a single bug but a distributed, subtle inference-engine difference. The text (95.3%) and table (91.3%) metrics are much closer to parity because they are less sensitive to pixel-level precision.

## Summary table — complete three-backend evaluation

| Backend | canary-148 | full-1651 | formula CDM (canary) | >14k ViT | speed | resolution |
|---|---|---|---|---|---|---|
| vLLM 0.16.1 (Flash-Attn) | **94.81** | — | **96.48** | ✅ (capped) | ~6 s/page | capped 3.4M |
| transformers 5.13 (SDPA) | 94.11 | — | 94.25 | ❌ NaN >14k | ~180 s/page | capped 3.4M |
| llama.cpp (C++ GGML) | 93.33 | **92.09** | 90.83 | ✅ **(uncapped)** | **~1.4 s/page** | **full res** |
| llama.cpp (C++ GGML, capped) | 92.50 | — | 88.97 | ✅ | ~1.4 s/page | capped 3.4M |

## What we're taking away

- **llama.cpp is the only backend that can run at full resolution on ROCm** (the >14k ViT instability doesn't affect its C++ path). Full resolution is strictly better than capping — confirming the cap was always a workaround, not an intended operating point.
- The formula CDM gap is best explained as an inference-engine artifact, not a model or preprocessing bug (the leading explanation, not a singly-proven root cause). Closing it would require either (a) kernel-level numerical alignment between GGML and PyTorch, or (b) accepting it as a known backend characteristic.
- We're sharing these findings in the hope they're useful if your team ever evaluates the llama.cpp path for production. The full per-page outputs and scoring artifacts are available on request.

Thanks! 🙏
