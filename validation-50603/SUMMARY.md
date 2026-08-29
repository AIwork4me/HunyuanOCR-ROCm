# validation-50603 — SUMMARY

Three-state A/B validation of vllm-project/vllm#50603 (HunyuanOCR on gfx1100/Radeon Pro W7900D, Triton paged-attention fallback) against PR #53856 (CK kernel V-padding mask) and PR #54210 (widen the gfx11 gqa_ratio gate to 1). Protocol, provenance, environment and deviations: see [README.md](README.md). All runs on this box, 2026-08-29, single Radeon Pro W7900D (gfx1100, 48 GB), vLLM 0.25.1 base, torch 2.12.0+rocm7.14.0, transformers 5.10.2, model `tencent/HunyuanOCR@de8f10ad` (sha256-verified against the parent repo's REPRO.yaml artifact).

## Verdict

**The #50603 symptoms do not reproduce on this validation stack, in any of the three states.** On the issue's own reproducer, vLLM 0.25.1 (state A) already produces byte-identical, fully correct output across three identical greedy generations at ~15360 vision tokens — with and without warmup — and at 3840 tokens; the fallback warning and the `kernel_paged_attention_2d` first-call JIT warning that anchor the issue's log evidence appear exactly as reported, yet run 0 does not diverge and nothing is garbled. Consequently:

- **PR #53856 alone (state B) changes nothing** — byte-identical results to state A in all nine runs. This is expected and confirms the PR's own scope: its fix lives in the ROCm CK custom paged-attention kernel, which a gqa_ratio=2 model never reaches without the gate widening. #53856 cannot fix #50603 by itself.
- **PR #53856 + #54210 (state C) reroutes decode to the CK kernel and stays deterministic and correct** — the fallback warning and the `kernel_paged_attention_2d` JIT disappear from state C's logs (the Triton chunked-prefill `_fwd_kernel` remains, which is expected on the CK path), and the outputs are again byte-identical and fully correct.
- The observed outputs match across all three states at the same sha8 (`d2bbe837` full-res, `a2fe683f` short control), i.e. on this stack the Triton fallback and the CK custom kernel are output-equivalent for this workload under greedy decoding.

For the open PRs this reads as supportive of #54210's position (its description already argued against the garbled-output theory and measured the Triton kernel's relative error as flat from 1K to 32K): on the one box where the reported symptoms were produced, with the reported version, the reporter's two symptoms are not observable in 2026-08 with either attention kernel, while #54210's gate widening demonstrably routes this model to the CK kernel and yields identical, correct results. It also strengthens #54210's ordering argument that #53856 should land first — state B shows #53856 is inert for this workload until the gate admits gqa_ratio 2.

## What could not be tested here

- **The reporter's exact runtime (torch 2.11.0+rocm7.14.0).** An attempted 2.11 probe on this container failed inside vLLM engine init with `hipErrorInvalidValue` (in `torch.nn.functional.linear` / blas) in the two wheel configurations tried (with the gfx11 family wheel installed, and with it removed); logs kept in `baseline-torch211-unrunnable/` and `env/run-state-a-torch211.log` + `env/build-state-a-torch211.log`. The reported symptoms may still be live on the reporter's original stack — 0.25.1 predates the 7.14 toolchain and the reporter's own build caveat flags exactly this class of environment sensitivity.
- **Current upstream main.** On the PRs' own base (`d1922cb5a7`, 2026-08-28) the reproducer cannot run at all: upstream removed the native Hunyuan V1/VL implementation (#53272, #53615) after the issue was filed, and the generic Transformers backend the model now routes through fails before any paged-attention code is reached — first by truncating the model's 4-axis XD-RoPE positions to 3 axes (`ValueError: Expected 4 multimodal RoPE channels, got position_ids with shape (3, 1, 8192)`; vLLM's m-RoPE state is hardcoded to 3 dims while transformers normalizes this model to a 4-section m-RoPE), then, after a minimal position-buffer patch, on an attention reshape for HunYuan's MLA-variant q/k (`RuntimeError: shape '[-1, 16, 192]' is invalid for input of size 16777216`). A dedicated rerun on a clean worktree produced fresh logs for both walls (`env/mainline-probe/`); the position-buffer patch is preserved (`env/attempt-mainline-enabling-patch.diff`) and the second wall is genuine model-support work, out of scope here. This is an upstream integration finding worth knowing independently of the PRs: **the #50603 workload is currently unservable on vLLM main.**

## Results table (from scripts/parse_results.py)

Deterministic = the three greedy generations in one process are byte-identical (symptom A detector; expect NO on the issue's stack). GT lines = ground-truth text lines recovered from the rendered page (full-res page: 9 lines; short control page: 6 lines; expect heavy misses on the issue's stack). The issue's observed values on the reporter's stack are shown for contrast; sha8s are stack-dependent and not comparable across environments.

Reproducibility of the no-repro: the entire state A suite was re-run once after the audit (`baseline-rerun/`) and all nine generations came out byte-identical to the first pass, including the same per-engine routing markers — the headline result is itself reproducible on this stack, not a one-off.

| protocol (issue-observed on 0.25.1/torch 2.11) | 3× sha8 | deterministic | correct |
|---|---|---|---|
| full-res, no warmup | ba29b5a1, fd0cc624, fd0cc624 | no | garbled |
| full-res, warmup + ×3 | fd0cc624 ×3 | yes | garbled |
| short control, no warmup | f00c87f2, f31888d4, f31888d4 | no | (too small to judge) |

| state | script | 3× sha8 | deterministic | GT title in any run | GT lines best | warmup sha8 |
|---|---|---|---|---|---|---|
| baseline (`752a3a504`) | repro_determinism | d2bbe837, d2bbe837, d2bbe837 | yes | yes | 9/9 | — |
| baseline (`752a3a504`) | repro_warmup | d2bbe837, d2bbe837, d2bbe837 | yes | yes | 9/9 | d2bbe837 |
| baseline (`752a3a504`) | repro_control_short | a2fe683f, a2fe683f, a2fe683f | yes | yes | 6/6 | — |
| pr53856 (`c2a2677a1`) | repro_determinism | d2bbe837, d2bbe837, d2bbe837 | yes | yes | 9/9 | — |
| pr53856 (`c2a2677a1`) | repro_warmup | d2bbe837, d2bbe837, d2bbe837 | yes | yes | 9/9 | d2bbe837 |
| pr53856 (`c2a2677a1`) | repro_control_short | a2fe683f, a2fe683f, a2fe683f | yes | yes | 6/6 | — |
| pr53856-pr54210 (`01a3fe7d2`) | repro_determinism | d2bbe837, d2bbe837, d2bbe837 | yes | yes | 9/9 | — |
| pr53856-pr54210 (`01a3fe7d2`) | repro_warmup | d2bbe837, d2bbe837, d2bbe837 | yes | yes | 9/9 | d2bbe837 |
| pr53856-pr54210 (`01a3fe7d2`) | repro_control_short | a2fe683f, a2fe683f, a2fe683f | yes | yes | 6/6 | — |

## Routing evidence (kernel actually executed per state)

Extracted by `run_state.sh` into `<state>/routing-markers.txt` (one set per engine process; three processes per state, one per script).

| state | gate (vllm/platforms/rocm.py) | "Cannot use ROCm custom paged attention kernel, falling back to Triton" | `kernel_paged_attention_2d` JIT (Triton decode) | `_fwd_kernel` JIT (Triton chunked prefill) | effective decode kernel |
|---|---|---|---|---|---|
| A baseline | gqa_ratio ∈ [3,16] | present | present | present | Triton fallback |
| B +#53856 | gqa_ratio ∈ [3,16] | present | present | present | Triton fallback (#53856 inert: fix is in the CK kernel this model cannot reach) |
| C +#53856+#54210 | gqa_ratio ∈ [1,16] | **absent** | **absent** | present (expected: prefill stays Triton on the CK path) | ROCm CK custom kernel |

## Observed output (identical in all three states, all runs)

Full-res 2048×1920 page, greedy, `repro_determinism.py` run 0 (runs 0-2 share sha8 `d2bbe837`, 260 chars):

```text
Quarterly Financial Summary

Revenue: 12,480,000 USD
Cost of Goods Sold: 4,210,000 USD
Gross Profit: 8,270,000 USD
Operating Expense: 3,150,000 USD
Net Income: 5,120,000 USD
Earnings Per Share: 2.84 USD
Fiscal Year: 2026 Q2 Report
Prepared by Finance Committee
```

Engine-side markers in state A, matching the issue's quoted log lines verbatim (line numbers included): `chunked_prefill_paged_decode.py:419` fallback warning and `jit_monitor.py:129` "Triton kernel JIT compilation during inference: kernel_paged_attention_2d" — the JIT warning fires on the first generate, yet run 0 equals runs 1 and 2, which is the direct non-repro of symptom A on this stack.

## Notes and deviations

Full list in [README.md](README.md) (§ Environment, § Why the base is v0.25.1). Short form: the runtime stack is the community rocm7.14 wheel stack in every run (torch 2.12.0+rocm7.14.0, triton 3.7.1+git rocm7.14); torch 2.11.0 (the reporter's exact runtime) could not be tested because that stack does not initialize on this container (`hipErrorInvalidValue` in engine init, two wheel configurations tried — see `baseline-torch211-unrunnable/`); transformers is 5.10.2 instead of 5.13.0 (5.11+ breaks vLLM 0.25.1's HunYuanVL image-processor import; transformers only supplies the tokenizer/processor); the base is v0.25.1 because current main cannot serve this model at all (native implementation removed post-issue); #53856 is applied as its full production diff rather than a head cherry-pick (the PR head is a test-only commit). The vLLM build is driven by the box's system 7.2.1 hipcc (the rocm7.14 wheel SDK ships no build toolchain) — identical across all three states, so it cannot confound the A→B→C attribution; a fully-7.14 (SDK clang) build was attempted and is blocked inside vLLM 0.25.1's own RDNA3 sources (`env/attempt-714-toolchain.md`).

## Files

```text
validation-50603/
├── README.md                  # protocol, provenance, environment, how to run
├── SUMMARY.md                 # this file
├── baseline/                  # state A (v0.25.1)      logs, evidence JSON, env-capture, routing markers
├── baseline-rerun/            # state A repeat run (reproducibility of the no-repro)
├── pr53856/                   # state B (+ #53856 backport)
├── pr53856-pr54210/           # state C (+ #53856 + #54210 backports)
├── baseline-torch211-unrunnable/  # failed torch 2.11 probe (kept as evidence)
├── env/                       # state-commits.txt, per-state diffs, upstream PR diffs, build/run logs,
│                              # mainline-probe/ (current-main failure logs), enabling-patch attempt,
│                              # model-download.log, env.sh
├── scripts/                   # verbatim gist reproducer scripts + evidence dump, run_state.sh,
│                              # rebuild_state.sh, capture_env.sh, parse_results.py
└── draft-comment-50603.md     # proposed follow-up comment for the issue (for review before posting)
```
