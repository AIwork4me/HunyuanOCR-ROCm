# Interpretation (Phase 7)

## Decision logic applied

| case | reading of our data |
|---|---|
| A: 0.23 nondet / 0.25 det | not observed — 0.25.1 varies in 10 of 12 primary cells (11 of 12 control) |
| B: both deterministic | not observed |
| **C: both TP=1 nondeterministic** | **observed** — 11 of 12 (0.23.1.dev1) and 10 of 12 (0.25.1) primary cells vary |
| **D: 0.25.1 unexpectedly nondeterministic** | **observed as a facet of C** — our earlier HunyuanOCR non-reproduction does not generalize to this model/harness |
| E: 0.23 cannot run | not the case after the mm-limit fix (initial OOM diagnosed and resolved, `../artifacts/mm-zero-decision.md`) |

**Primary reading (Case C):** the greedy nondeterminism reproduces on the W7900D at TP=1 and is **not explained by the vLLM version axis** (0.23.1.dev1 vs 0.25.1, everything else pinned identical) and **not explained by TP topology** (cadamcat's TP=2 and our TP=1 both vary, in the same regime, with the same eager-insensitivity and the same engine-to-engine spread).

## What this rules out / strengthens

Ruled out by this experiment (within its scope):

- "TP=2 distributed/all-reduce is required for the effect" — varies at TP=1 with no inter-GPU communication at all (single rank, world_size=1).
- "Fixed between 0.23.x and 0.25.1" — both source builds, built identically on the same torch/triton/transformers, vary.
- "CUDA graph capture/replay is the cause" — `--enforce-eager` varies on both versions at TP=1 (consistent with cadamcat's eager A/B at TP=2).
- "our earlier #50603 no-repro on HunyuanOCR generalizes" — the same box, same torch stack, and the clean v0.25.1 build that produced byte-stable HunyuanOCR output produces 6–8-unique-of-8 here. The non-repro was model/workload-specific, not stack-wide.

Strengthened:

- The phenomenon tracks the **model + prompt regime**, not the version or topology: Muse-Glimmer-30B-INT4 (compressed-tensors W4A16, Transformers backend, synthetic token-ID prompts) varies strongly at ctx=512 on every stack tried; HunyuanOCR (bf16, native impl, OCR prompts) was byte-stable on the same box/stack.
- At ctx=8192 both versions sample from a common set of 5–6 attractor sequences (4 byte-identical across arms): the model defines a small set of near-tied continuations; which one wins varies run to run. This is consistent with argmax flipping on logits whose low-order bits differ between executions — a numeric-noise-on-near-ties picture — but this experiment does not establish that mechanism; it is a hypothesis to test (e.g. by logging top-2 logit gaps at divergence positions).

Remaining axes (ordered by what the evidence now favors):

1. **Model/quantization path**: the RDNA3W4A16 dequant-GEMM path (selected identically on both arms — `Using RDNA3W4A16LinearKernel` in both logs) plus a prompt regime with many near-ties.
2. **ROCm runtime/kernel-level execution variance at bf16 on gfx1100** (cadamcat's own fixed-input kernel probe argued the decode kernel alone is bit-stable; the variance then enters above the kernel or in a different kernel than they probed).
3. **torch 2.12 wheel stack specifics** — untestable against their exact torch 2.11 here (their stack does not initialize on this container).

## On cadamcat's qualitative "settles after three generations" pattern

Their muse/512/eager TP=2 engine split 3-then-5. Our 18 TP=1 muse/512 engines show no analogous temporal transition (10 fully-singleton engines, scattered pairings otherwise). We report its absence rather than forcing the analogy — at TP=1 the within-engine variation is stronger, which plausibly masks any settling.

## Why the bisect phase (Phase 8) is skipped

Its precondition ("0.23 reproduces, 0.25 is stable, both under comparable dependencies") is not met: 0.25.1 varies. There is no good/bad interval to bisect on the version axis.

## Recommended next experiment (single highest-value)

Fix the prompt, model and version (0.25.1, TP=1, ctx=512) and log, per generated token, the top-2 logits and their gap, on a varying engine. If the flips are exclusively at gap ≈ 0 (ties within float noise), the mechanism is numeric noise on near-ties and the search shifts to which computation injects run-to-run variance (the W4A16 dequant-GEMM chain is the prime suspect on this stack); if flips occur at healthy gaps, something above the logits (e.g. sampling/host-side state) is implicated. This single instrument splits the remaining hypothesis space in half.
