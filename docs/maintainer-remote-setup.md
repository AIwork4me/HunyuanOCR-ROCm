# Maintainer remote-setup checklist

One-time GitHub settings to apply (manually — not automated) for
`AIwork4me/HunyuanOCR-ROCm`. This is a checklist for a maintainer with admin
access, not something CI touches.

## Repository metadata

- **Description:** "Evaluation-backed AMD ROCm port of HunyuanOCR-1.5 — three
  backends on OmniDocBench v1.6 (gfx1100/RDNA3)."
- **Topics:** `hunyuanocr`, `ocr`, `document-ai`, `amd-gpu`, `rocm`, `rdna3`,
  `llama-cpp`, `vllm`, `omnidocbench`, `vision-language-model`.
- **Homepage URL:** (none, or link to the benchmark-methodology doc).

## Community

- Enable **Discussions** (Q&A).
- Enable **Issues** (templates are in `.github/ISSUE_TEMPLATE/`).
- Enable **GitHub Actions**, **Projects** (optional).

## Security

- Enable **Private Vulnerability Reporting** (Settings → Security → Code
  security). SECURITY.md points reporters here.
- Enable **Dependabot** security alerts (Settings → Code security).
- Do **not** enable "Allow GitHub Actions to create/approve PRs" unless needed.

## Branch protection (`main`)

- Require pull request before merge.
- Require approvals (≥1) + review from CODEOWNERS.
- Require status checks to pass before merging: `lint-test-build` (the CI job in
  `.github/workflows/ci.yml`). Require branches up to date.
- Require conversation resolution before merge.
- **Do not** allow force-push to `main`; do not allow deletions.
- **Squash merge** as the default merge method (linear history).

## Self-hosted ROCm runner

- The `rocm-smoke` workflow (`workflow_dispatch` only) runs on a self-hosted
  `linux,rocm,gfx1100` runner. Register that runner under Settings → Actions →
  Runners. It must be on an **isolated** machine (it checks out contributor code
  onto a host with weights + data).
- Never approve `rocm-smoke` on fork PRs — the workflow file already restricts it
  to `workflow_dispatch`, but reviewers must still not be socially engineered into
  dispatching it on untrusted input.

## Social preview + release

- Upload a Social Preview image per [visual-design-brief.md](visual-design-brief.md).
- When ready to release: create a tag `v0.1.0` from the merge commit and a GitHub
  Release with notes from [releases/v0.1.0.md](releases/v0.1.0.md). (Do **not** do
  this until the maintainer confirms the verified model/GGUF SHA256 in the lock
  file match the artifacts they intend to publish.)
