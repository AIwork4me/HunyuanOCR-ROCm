# cadamcat harness analysis — benchmarks/gfx1100-greedy-eager-ab

Upstream repo: https://github.com/cadamcat/dual-radeon-vllm
Frozen commit: `782c7d431ab8e821242f2a717bf5d03b0be3301d` (cloned 2026-08-30T22:53:59+00:00; `git status --short` empty at freeze). Working tree preserved untouched under `upstream/dual-radeon-vllm/`.

## What the harness measures

`nondet_eager.py <muse|gemma3> <proc-tag> <0|1 eager>` creates ONE vLLM `LLM` engine, then for each context depth in `[512, 8192]`: builds a synthetic token prompt, does one warm-up generation (`max_tokens=8`), then 8 identical measured greedy generations (`max_tokens=64`), records all 8 token-ID sequences, the count of distinct sequences, and each run's first-divergence index versus run 0.

Measurement constants (verbatim from `benchmarks/gfx1100-greedy-eager-ab/nondet_eager.py` at the frozen commit):

| Constant | Value |
|---|---|
| Models | `muse` = `/models/Muse-Glimmer-30B-INT4`, `gemma3` = `/models/gemma-3-27b-it-w4a16` (local paths on cadamcat's box) |
| Context depths | `DEPTHS = [512, 8192]`, both run in one engine, 512 first |
| Prompt construction | `TokensPrompt(prompt_token_ids=[1000 + (i % 20000) for i in range(d)])` — synthetic IDs, deterministic, no tokenizer involvement in prompt content |
| Warm-up | 1 × `llm.generate([p], SamplingParams(max_tokens=8, temperature=0.0, ignore_eos=True))` per depth, immediately before the 8 measured generations, same prompt |
| Measured generations | `REPEATS = 8` × `llm.generate([p], SamplingParams(max_tokens=NGEN=64, temperature=0.0, ignore_eos=True))` |
| Greedy semantics | `temperature=0.0` explicit in `SamplingParams` (greedy argmax path); `ignore_eos=True`; no top_k/top_p overrides; seed not passed to `LLM` (engine default `seed=0` recorded in cadamcat's logs) |
| Engine kwargs | `max_model_len = max(DEPTHS) + 512 = 8704`, `max_num_seqs = 128`, `gpu_memory_utilization = 0.92`, `disable_log_stats = True`, `enforce_eager = <arg>`, `tensor_parallel_size = 2` |
| Environment | `VLLM_ROCM_CLONE_MMAP_WEIGHTS=1` (setdefault), `VLLM_CLONE_MMAP` popped |
| Comparison method | distinct = `{tuple(seq)}` over the 8 runs; first divergence = first index where run_i differs from run_0 |

## cadamcat's published result being reproduced (the target)

From `eager-ab.json` (vLLM 0.23.1.dev1+g9ddef7117.d20260715, ROCm 7.14 container, TP=2, one engine per cell):

| model | ctx | graphs-on distinct/8 | eager distinct/8 | varied (graphs → eager) |
|---|---:|---:|---:|---|
| muse | 512 | 5 | 5 | varies both ways |
| muse | 8192 | 1 | 3 | was stable → unstable |
| gemma3 | 512 | 1 | 1 | stable both ways |
| gemma3 | 8192 | 5 | 2 | varies both ways |

Qualitative pattern to check for at muse/512 eager: cadamcat's 8 generations split 3-then-5 — first three begin one way, last five another, identical from token 1 onward within each group ("settles after three generations"), despite the preceding warm-up.

## cadamcat's environment (from run logs and repo metadata)

- vLLM `v0.23.1.dev1+g9ddef7117.d20260715` — official `rocm/vllm` **7.14.0** container image, **PyTorch 2.11.0** (per `sliding-window-block-skip.json` userspace string), safetensors 0.8.0, nccl 2.27.7, Python 3.14.
- 2× RX 7900 XT (gfx1100, 19.98 GiB each) → forced TP=2; engine log records `disable_custom_all_reduce=True`, seed=0, prefix caching on, chunked prefill on, quantization=compressed-tensors for both models, async scheduling on.
- Their site-packages carry local kernel patches (window block-skip, 3 `first_block` sites in `chunked_prefill_paged_decode.py`). Their README states the nondeterminism is symmetric between kernel patch states, so the patch is not treated as a variable by them; our runs are stock vLLM either way. Disclosed as an environment difference.

## Model provenance resolution

cadamcat's harness uses local paths; the repo does not record HF IDs. Resolution by name + config evidence (both engines report `quantization=compressed-tensors`):

| harness key | local path (cadamcat) | HF repo (resolved) | Pinned revision (main branch SHA, 2026-08-30) | On-disk size (per cadamcat) |
|---|---|---|---|---|
| muse | `/models/Muse-Glimmer-30B-INT4` | `RedHatAI/Muse-Glimmer-30B-INT4` | `f5b410ce4234fad70eef8be99b4680ee4e30b418` | 21 GB |
| gemma3 | `/models/gemma-3-27b-it-w4a16` | `RedHatAI/gemma-3-27b-it-quantized.w4a16` | `2b537554d6c6f6368945e8df4e5fb7bbbb5d56c9` | 19 GB |

Supporting evidence: dir name matches repo name exactly for Muse; for gemma3 the resolved repo is compressed-tensors w4a16 of gemma-3-27b-it at ≈19.8 GB, matching cadamcat's "19 GB on disk" note; no other candidate matches name+quant+size. Residual risk that cadamcat's gemma3 snapshot differs from `main` of that repo (their copy has no recorded revision) — accepted, and Muse (the primary) is name-exact.

## What WE preserve exactly vs what must change

Preserved byte-for-byte in our `harness/nondet_eager_tp1.py` (no redesign):

1. `DEPTHS = [512, 8192]`, `NGEN = 64`, `REPEATS = 8`
2. Prompt construction `1000 + (i % 20000)` per depth
3. Warm-up semantics: exactly one `max_tokens=8` warm-up per depth, same prompt, before the measured 8
4. `SamplingParams(max_tokens=NGEN, temperature=0.0, ignore_eos=True)` for measured runs
5. Engine kwargs other than TP: `max_model_len=8704`, `max_num_seqs=128`, `gpu_memory_utilization=0.92`, `disable_log_stats=True`, `enforce_eager=<arg>`
6. Same model repositories, pinned to explicit revisions (cadamcat left revision=None against a local snapshot)
7. Environment handling: `VLLM_ROCM_CLONE_MMAP_WEIGHTS` setdefault, `VLLM_CLONE_MMAP` popped
8. Distinct-count and first-divergence-vs-run-0 computation

Changed, with reasons (each appears in `artifacts/harness.diff`):

1. **`tensor_parallel_size=2` → `1`** — the entire point of this experiment: our W7900D has 48 GB and cadamcat explicitly cannot run TP=1 ("the topology is not testable from this box"). This is the variable under test.
2. **Model paths** → local HF snapshot dirs on this machine (same repos as above). Paths were machine-specific in the original too.
3. **Output file + extended JSON** — cadamcat writes one JSON per (model, eager, proc) with seqs; we keep every seq and add, per the task's evidence requirements: SHA256 per generation, decoded text, per-pair first-divergence matrix, engine/vllm/torch/ROCm metadata, and an `environment` block. Generation semantics untouched.
4. **Engine teardown** — explicit `del llm` + `gc.collect()` + `torch.cuda.empty_cache()` at process end (upstream relies on process exit); each engine remains one process, one engine per invocation, exactly as upstream.

Everything else is untouched. The upstream files remain pristine under `upstream/`.

## Version arms and dependency control (plan of record)

- Arm "0.25.1": clean upstream tag `v0.25.1` = `752a3a504485790a2e8491cacbb35c137339ad34` (same commit as prior validation's state A). Prior validation already proved this builds and runs on this box with torch 2.12.0+rocm7.14.0 (the state-c backports do not enter our build — fresh worktree at the tag).
- Arm "0.23.1.dev1": upstream commit `9ddef7117` (the `+g9ddef7117` in cadamcat's version string; exact commit is recoverable, `.d20260715` marks their container's dirty-tree date which we cannot reproduce and do not need to — it reflects their local kernel patches).
- Common torch: **torch 2.12.0+rocm7.14.0** on both arms (single-variable A/B on vLLM version). Known unavoidable deviation from cadamcat: their exact runtime is torch 2.11.0, which does not initialize on this container (prior validation evidence: `hipErrorInvalidValue` in engine init with vLLM 0.25.1, two wheel configurations tried — `validation-50603/baseline-torch211-unrunnable/`). This joint vLLM+torch difference versus cadamcat is disclosed prominently wherever results are stated.
- Both builds: system 7.2.1 hipcc toolchain against 7.14-runtime torch — the same build configuration the prior validation used for all its states (toolchain identical across arms here, so it cannot confound the A/B).
