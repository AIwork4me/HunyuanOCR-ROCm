# Security policy

## Reporting a vulnerability

Please report security issues **privately** via **GitHub Private Vulnerability
Reporting** (Repo → Security → Report a vulnerability). Do **not** open a public
issue for security-sensitive bugs.

> A dedicated security email is intentionally **not** listed because none is
> operated for this project. GitHub Private Vulnerability Reporting is the
> channel; a maintainer must enable it in the repo's Security settings (see
> [docs/maintainer-remote-setup.md](docs/maintainer-remote-setup.md)).

Please include: affected commit/version, the command and config used, and a
minimal reproduction. Acknowledgement is best-effort (this is a community
project).

## Supported versions

Only the latest release on `main` receives fixes. There are no LTS branches.

## Threat model you should know about

- **`llama-server` has no default strong authentication.** The Quick Start binds
  to `127.0.0.1` (loopback). If you expose the server, put an authenticating
  reverse proxy in front and firewall it — **never expose it to the public
  internet**. An open server lets anyone run inference and exfiltrate model
  output.
- **Model output is untrusted.** HunyuanOCR produces text/markdown from images;
  treat its output like any LLM output (may be wrong, may contain injected
  content from the source image). Do not execute or render it blindly.
- **Self-hosted ROCm runner.** The `rocm-runner-preflight` workflow runs on a
  self-hosted GPU runner and is `workflow_dispatch` only — it is **never**
  triggered by push or pull request, and must not be enabled on fork PRs. It
  checks out contributor code onto a machine holding weights/data; only a trusted
  maintainer dispatches it.

## What not to put in issues / PRs

- Secrets, tokens, API keys, credentials.
- Model weights or quantized artifacts (`.gguf`, `.safetensors`, `.bin`).
- Private or copyrighted documents / datasets.
- Personally identifying information.

The `.gitignore` blocks common heavy-data extensions; if you accidentally commit
any of the above, notify a maintainer so it can be force-pushed off history.
