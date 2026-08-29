# validation-50603 — Issue vllm-project/vllm#50603 three-state A/B validation

Independent, issue-specific validation of the #50603 HunyuanOCR reproducer on gfx1100, isolating the effect of two upstream PRs. This directory is self-contained on purpose: the benchmark defaults of the parent repository (pixel cap 3.4 Mpx, `--max-model-len 65536`, torch.compile, the frozen decoding-contract prompt) are deliberately NOT used here, because the issue's authoritative reproducer pins its own parameters and the pixel cap would downscale the 2048×1920 page (3.93 Mpx) and change the vision-token count that defines the long-sequence symptom.

## What is being tested

Issue vllm-project/vllm#50603 (2026-07-31) reports on gfx1100 (RDNA3, Radeon Pro W7900D), vLLM 0.25.1 with tencent/HunyuanOCR (decoder GQA, gqa_ratio=2): (A) the first `generate()` call of a process is non-deterministic under greedy decoding (the ROCm CK custom paged-attention kernel is gated out at gqa_ratio<3, so vLLM falls back to the Triton `kernel_paged_attention_2d`, which is JIT-compiled on first use), and (B) at ~15360 vision tokens the warmed, deterministic output is garbled OCR, while transformers with `attn_implementation="sdpa"` on the same stack is deterministic and correct.

Two open upstream PRs are relevant: #53856 ("[Bugfix][ROCm] Mask paged attention V cache padding", changes `csrc/rocm/attention.cu`, fixes `0 * NaN` propagation from unwritten FP8 V-cache padding in the CK kernel) and #54210 ("[ROCm] Enable Navi custom paged attention at gqa_ratio 1 and 2", changes one line in `vllm/platforms/rocm.py`, which routes gqa_ratio=2 models such as HunyuanOCR to the CK kernel instead of the Triton fallback; its description explicitly says #53856 should land first and that it does not address the garbled-output symptom).

## Result (short version)

The bug does not reproduce on this box on any of the three states: all nine runs (three scripts × three states) are byte-identical across their three greedy generations and extract all ground-truth lines correctly. Full detail in [SUMMARY.md](SUMMARY.md).

## States (vLLM code under test)

The issue was filed against vLLM 0.25.1, and upstream removed the native Hunyuan V1/VL model implementations (`vllm/model_executor/models/hunyuan_vision.py`) only AFTER that (#53272, #53615) — on the current main that the PRs are proposed against, the model no longer loads at all (see "Why the base is v0.25.1" below). The three states therefore use the issue's own version as the common base, with each PR applied as an auditable backport:

| state | dir | #53856 | #54210 | vLLM commit |
|---|---|---|---|---|
| A baseline | `baseline/` | no | no | `752a3a5044` (= v0.25.1 tag) |
| B | `pr53856/` | yes | no | `c2a2677a13` (state-a + full #53856 `csrc/rocm/attention.cu` diff, +104 lines) |
| C | `pr53856-pr54210/` | yes | yes | `01a3fe7d2f` (state-b + the #54210 one-line gfx11 gate widening in `vllm/platforms/rocm.py`) |

Applied diffs are recorded verbatim: `env/state-b_vs_state-a.diff`, `env/state-c_vs_state-b.diff`, `env/state-c_vs_state-a.diff`, plus the pristine upstream PR diffs `env/pr53856.full.diff` and `env/pr54210.full.diff`. Backport notes: #53856 is a multi-commit PR and its head commit is a test-only follow-up — the backport therefore carries the full production diff (`796822d1..d9a29233`, `csrc/rocm/attention.cu` only) and not the PR's test-file hunk, which targets the current test harness; #54210 is a single commit and applies with its one line unchanged.

### Why the base is v0.25.1 (and what happened on current main)

An initial attempt based this validation on current upstream main (`d1922cb5a7`, the base of #54210, 2026-08-28) with the PRs cherry-picked. On that main, HunyuanOCR is served through the generic Transformers modeling backend because the native implementation was removed, and it fails before any paged-attention code is exercised, in two independent places: (1) the model is a 4-axis XD-RoPE model, transformers normalizes it into a 4-section m-RoPE, but vLLM's m-RoPE position buffer is hardcoded to 3 dims, so the 4th position channel is silently dropped and the HF forward rejects the (3, seq) position_ids; (2) after patching that, vLLM's unified attention layer fails to reshape HunYuan's MLA-variant q/k (head 192 = 128 nope + 64 rope). A minimal enabling patch for the first wall (`env/attempt-mainline-enabling-patch.diff`, committed identically on all three mainline states) was built and verified, but the second wall is genuine model-support work, out of scope for this validation. Since the PRs' own base cannot run the #50603 workload at all, the issue's version is the only base on which the authoritative reproducer actually runs; the mainline state branches were deleted, their enabling patch and build/run logs kept under `env/` (`build-state-a*.log`, `run-state-a.log` at the time of the attempt).

## Reproducer protocol (identical across all three states)

The three scripts in `scripts/repro_*.py` are copied verbatim from the gist referenced by the issue (gist `030fc6a872de36197178c9e8217949d1`): `repro_determinism.py` (2048×1920 page, 15360 ViT patches, three identical greedy `generate()` calls, no warmup — symptom A detector), `repro_warmup.py` (same page, one discarded warmup then three measured calls — isolates symptom B), `repro_control_short.py` (1024×960 page, 3840 patches, no warmup — shows symptom A is not length-gated). The only addition is a guarded block at the end of each script that dumps the already-printed output texts and sha8 hashes to a JSON file when `V50603_EVIDENCE` is set; it does not touch the workload.

Pinned parameters (from the issue): model `tencent/HunyuanOCR`, dtype `bfloat16`, `max_model_len=32768`, `gpu_memory_utilization=0.90`, `enforce_eager=True`, `limit_mm_per_prompt={"image": 1}`, `trust_remote_code=True`; sampling `temperature=0.0, top_p=1.0, top_k=-1, max_tokens=1024` (512 in the short-seq control), `repetition_penalty=1.08`; greedy, no seed, prompt = the issue's Chinese document-parsing prompt, one image per call. Expected outcome per the issue (observed on the reporter's stack): `run0 != run1 == run2` with sha8 `ba29b5a1, fd0cc624, fd0cc624` and the GT title absent (garbled), warmup restoring determinism but not correctness. sha8 values are stack-dependent, so they are recorded as qualitative evidence; the acceptance signals are the determinism pattern and the ground-truth line hit count computed in `scripts/parse_results.py` (the full-res page carries 9 GT lines, the short control page 6).

Routing evidence is taken from the run logs: states A and B must log "Cannot use ROCm custom paged attention kernel, falling back to Triton implementation" plus the `kernel_paged_attention_2d` JIT warning (gqa_ratio=2 excluded); state C must not — and does not — show either, which is what attributes any state-C change to the gate widening rather than to unrelated drift.

## Environment

The runtime stack is the community ROCm 7.14 wheel stack end to end: torch 2.12.0+rocm7.14.0, torchvision 0.27.0+rocm7.14.0, triton 3.7.1+git rocm7.14, and the 7.14 ROCm user-space libraries (TheRock python SDK: `amd-torch-device-gfx1100`/`gfx11`, `rocm-sdk-core` 7.14) from the AMD multi-arch wheel index (`https://repo.amd.com/rocm/whl-multi-arch/`), on 1× AMD Radeon Pro W7900D (gfx1100, 48 GB, RDNA3), python 3.12.3, transformers 5.10.2 — all installed in `/workspace/venv-vllm50603` (hardlink clone of the workspace `venv-torch212`).

vLLM is installed editable from `/workspace/vllm-50603` (branches `state-a`/`state-b`/`state-c`) built with `PYTORCH_ROCM_ARCH=gfx1100 VLLM_TARGET_DEVICE=rocm`; switching states A→B (and B→C) requires a rebuild via `scripts/rebuild_state.sh` (A→B changes `attention.cu`; the B→C change is Python-only but a rebuild is harmless and keeps every state's `env-capture.txt` authoritative).

One build detail, stated once: the HIP compiler driving the vLLM build is the box's system 7.2.1 hipcc (`/opt/rocm`), because the rocm7.14 wheel SDK ships no CMake toolchain. This matches the issue reporter's documented build, and the toolchain is identical across all three states, so the A→B→C attribution is unaffected — the kernels that execute (Triton runtime-JIT, and the CK custom kernel compiled from identical sources) run on the same 7.14 runtime in every run. An attempt to build with a fully-7.14 toolchain instead was blocked inside vLLM 0.25.1's own RDNA3 quantized-GEMM sources under the newer clang (see `env/attempt-714-toolchain.md` + `env/attempt-714-toolchain-build.log`) — 0.25.1 predates that toolchain.

Deviations from the reporter's exact stack, with reasons: torch 2.12.0 instead of 2.11.0+rocm7.14.0 (same rocm7.14 runtime family — an attempted 2.11.0 probe on this container failed with `hipErrorInvalidValue` inside engine init in the two wheel configurations tried, see `baseline-torch211-unrunnable/` and `env/run-state-a-torch211.log`; the reporter's own build caveat flags exactly this class of stack sensitivity), and transformers 5.10.2 instead of 5.13.0 (the only mid-5.x version where vLLM 0.25.1's `hunyuan_vl_image.py` string-key `AutoImageProcessor.register` import survives — 5.11+ added a strict `key.__module__` check that turns the import into an AttributeError; transformers only supplies the tokenizer/processor here). The model is pre-downloaded at the pinned revision `de8f10ad2f00a0cefd790b526de8a65dcfdb3205` — whose `model.safetensors` sha256 equals the benchmark artifact recorded in the parent repo's REPRO.yaml — and the runs execute with `HF_HUB_OFFLINE=1` so the reproducer's bare `model="tencent/HunyuanOCR"` resolves deterministically.

## How to run

```bash
bash validation-50603/scripts/run_state.sh baseline state-a
bash validation-50603/scripts/rebuild_state.sh state-b
bash validation-50603/scripts/run_state.sh pr53856 state-b
bash validation-50603/scripts/rebuild_state.sh state-c
bash validation-50603/scripts/run_state.sh pr53856-pr54210 state-c
python validation-50603/scripts/parse_results.py
```

Each `run_state.sh` invocation checks out the branch, captures `env-capture.txt` (git + pip + gate-line provenance), runs the three scripts (each in a fresh process, 110 min timeout), and extracts the routing markers. Results land in `<state-dir>/` as logs, evidence JSON and routing markers; `SUMMARY.md` aggregates them.
