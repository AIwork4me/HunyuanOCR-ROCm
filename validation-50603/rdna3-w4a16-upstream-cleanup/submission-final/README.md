# submission-final — W4A16 upstream submission record (2026-09-01)

- SUBMISSION_BASE_SHA: `40b2f62061575905aaac8bc360eaea62a4baeb67` (vllm-project/vllm main, fetched fresh at submission; the three W4A16 patch files were untouched between the prior validated base 07ea9350 and this base, re-verified, and re-checked again immediately before opening the PR)
- Rebasing: old head `bd5d05816bf0a2a93a7f491375812759f7efe1b0` (parent 07ea9350) rebased onto the new base; local rebased commit e0a0cb6a (with DCO) had a byte-identical tree to the pushed head
- Pushed W4A16 PR head SHA: `dddf1647811a4663dc0de5511bad3fe348afb3b3` (tree verified identical to the locally validated build; branch `fix/rdna3-w4a16-determinism` on AIwork4me/vllm). Note: `git push --force-with-lease` was rejected twice by a transport quirk ("stale info" / "remote end hung up"), so the branch was updated via the GitHub refs API after an explicit lease check (ls-remote confirmed the remote was still at bd5d05816b immediately before the update); fork main was fast-forwarded to the submission base first so the new commit only carried the three patched files
- PR: #54706 — https://github.com/vllm-project/vllm/pull/54706 (OPEN, not draft, base main, head AIwork4me:fix/rdna3-w4a16-determinism, exactly the 3 W4A16 files, no attention.cu / no #53856 code; body uses "Addresses … #50603", not "Fixes")
- #50603 final comment: https://github.com/vllm-project/vllm/issues/50603#issuecomment-5489539359
- Minimal validation on the rebased build (all on W7900D/gfx1100, torch 2.12.0+rocm7.14.0; runtime import verified to be the rebased worktree after removing two stale editable registrations):
  - determinism pytest: 8/8 PASS
  - scalar M=1 (real Muse q_proj K=6656 N=4096 bf16): 1 distinct / 100, all finite, output SHA 6315ced8074f235f
  - WMMA representative M=128 (same weights): 1 distinct / 100, all finite, output SHA cf22e7d8be5305a2
  - Muse eager ctx512: 1 unique / 8 (first_div all None); routing log confirms `Using RDNA3W4A16LinearKernel`
- Optional perf sanity (5E) skipped per the task allowance: the implementation diff is unchanged from the fully benchmarked patch, so the prior same-path benchmark remains the authoritative performance evidence
- CI at submission time: "Check format" PASS; "pre-run-check" FAIL is the known first-time-contributor gate ("each PR must have the 'verified'/'ready'/'ready-run-all-tests' label, or the author must have at least 4 merged PRs (found 0). DO NOT request for the label to be added if you are an AI agent.") — the full test matrix waits for a maintainer label; no label was requested
