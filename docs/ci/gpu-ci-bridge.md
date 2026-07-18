# GPU-CI Bridge (gfx1100 on Radeon Cloud / Docker)

How HunyuanOCR-ROCm runs a **real gfx1100** smoke from GitHub CI, even though
GitHub-hosted Actions has no AMD GPU and the only GPU is a single Radeon Cloud
Docker instance (Jupyter-as-PID-1, no systemd, behind a MITM proxy, no inbound
from GitHub).

## TL;DR — it works

`workflow_dispatch` on `main` → a `pending` commit status → the box poller
(5 GPUs away) checks out the SHA, runs the real 1-page smoke on gfx1100, and
posts `success`. **Measured end-to-end: ~61 s; smoke itself 21 s.**

Proof on `main` HEAD `80b8ea5` (2026-07-18):
```
gpu-smoke (gfx1100) | success | PASSED gfx1100 ROCm7.2.53211 torch2.9.1+gitff65f5b complete=1 21s
```

## Architecture — "GitHub control plane + box data plane"

```
 GitHub (control plane)                      gfx1100 Docker box (data plane)
 ┌──────────────────────────────┐           ┌──────────────────────────────────┐
 │ .github/workflows/gpu-smoke  │ dispatch  │ src/hunyuan_ocr/ci/poller.py     │
 │  workflow_dispatch {ref}     │ ────────▶ │  (nohup; polls api.github.com)   │
 │  GITHUB_TOKEN posts          │           │  reads pending status for SHA    │
 │  status=pending on <SHA>     │           │  git worktree <SHA> (trusted)    │
 │                              │           │  runs TRUSTED rocm_smoke.sh      │
 │                              │ ◀──────── │  posts status=success|failure    │
 └──────────────────────────────┘  result   └──────────────────────────────────┘
```

The Check Run / status is the **single state object**: the workflow creates
`pending`; the poller completes it. The box **pulls** work and **pushes** results
(no inbound needed).

## The key finding: use **commit statuses**, not check-runs

The first cut used GitHub **check-runs**. The live run proved:

> **A user OAuth token can read check-runs but NOT create/update them — only a
> GitHub App (or `GITHUB_TOKEN`) can.** `PATCH /check-runs/{id}` → `403 "You must
> authenticate via a GitHub App."`

So the poller (which authenticates with the maintainer's `gh` user token) could
see the queued check-run but couldn't move it. **Commit statuses** (`POST
/commits/{sha}/statuses`) are writable by a user token — so the bridge uses
statuses end-to-end. Trade-off: a status description is ≤140 chars (no markdown
output). For a pass/fail smoke that's fine; if rich output is later wanted,
upgrade to a GitHub App + check-runs (see Production).

## Method (what makes it robust)

- **`api.github.com`-only** on the box — verified reachable and **not** intercepted
  by the proxy (unlike `git push` / receive-pack, which the proxy breaks). The
  poller never does `git push`.
- **Trusted harness + explicit ref.** The poller runs the box's pinned
  `scripts/rocm_smoke.sh` against a `git worktree` of the dispatched SHA. Only the
  dispatched SHA's model driver executes; a malicious ref can't alter the harness.
  Runs only for refs a maintainer explicitly dispatches — never on PR-open.
- **flock** single-run lock (GPU is single-tenant per smoke).
- **Idempotency** — the latest status for a SHA being terminal ⇒ skip.
- **Stale-sweep** — a `pending` older than 30 min ⇒ post `failure` (no silent hangs).
- **Resilient loop** — a failed pass is logged + retried; the daemon never dies on
  one bad interval.

## How to run it (on the box)

```bash
# one-time: materialize the deterministic 1-page input (gitignored, machine-local)
python scripts/make_smoke_input.py \
  --full-gt "$DATA/OmniDocBench.json" --manifest eval/canary_148.manifest.json \
  --images-dir "$DATA/images" --out /root/ocr-eval/smoke/OmniDocBench_smoke_1page.json

# start the poller (it survives while the container is up)
export HUNYUANOCR_ROCM_DIR=/workspace/HunyuanOCR-ROCm
export HUNYUANOCR_GGUF_DIR=/root/models/HunyuanOCR-gguf
export HUNYUANOCR_LLAMA_SERVER=/root/llama.cpp/build/bin/llama-server
export HUNYUANOCR_SMOKE_GT=/root/ocr-eval/smoke/OmniDocBench_smoke_1page.json
export HUNYUANOCR_SMOKE_IMAGES=/root/ocr-eval/OmniDocBench_data/images
export HUNYUANOCR_SMOKE_OUT=/root/ocr-eval/smoke/out
export HUNYUANOCR_PYTHON=/opt/venv/bin/python3
export HIP_VISIBLE_DEVICES=0
setsid python -m hunyuan_ocr.ci.poller --owner AIwork4me --repo HunyuanOCR-ROCm \
  > ~/.rocm_ci_poller.log 2>&1 < /dev/null & disown

# request a smoke from GitHub (or the Actions UI)
gh workflow run gpu-smoke.yml --ref main -f ref=main
```

CPU tests: `pytest tests/test_ci_poller.py` (25 tests, no network/GPU — the
`gh`/subprocess boundary is injected).

## Honest limitations + artifacts

- **Persistence:** PID 1 is `jupyter-lab` (no systemd). The poller is `setsid nohup`;
  it does **not** auto-restart if the container restarts. A queued `pending` simply
  waits (it doesn't drop) and is swept to `failure` after 30 min.
- **No inbound** → poll-only (~45–180 s pickup latency). A stuck box can't be
  probed from GitHub.
- **Orphaned check-run** on `d9b3c92` from the check-run attempt — left `queued`
  forever (can't be cleared without a GitHub App). Harmless.
- `gpu-smoke-selftest` status on `80b8ea5` — a one-off capability check; harmless.

## Production upgrade (prioritized asks for anruicloud)

The MVP proved the path; these make it production-grade:

1. **P0 — persistence/auto-start.** A startup-hook / lifecycle-script (or a real
   systemd-init mode) so the poller survives container restart. Today: manual
   `nohup`.
2. **P1 — long-lived process guarantee.** Confirmation a `nohup`'d process
   survives Jupyter session disconnects and anruicloud scheduling windows.
3. **P2 (optional) — inbound HTTPS ingress.** A stable URL + auth so GitHub can
   **push** (`repository_dispatch`/webhook) instead of the box polling — removes
   pickup latency. Requires anruicloud to expose the box.
4. **GPU isolation** — `HIP_VISIBLE_DEVICES` to a free card (already supported).

**Final-solution mapping:** anruicloud P0+P1 ⇒ the status-poller is production.
Add a **GitHub App** ⇒ richer check-runs (markdown output) if wanted. A
persistent systemd-init box with optional ingress ⇒ the native **self-hosted
Actions runner** path (most native, but needs the App-free runner agent to
tolerate the MITM proxy — unverified).

## Non-goals (future)

tag→release auto-gate (make `gpu-smoke` a required check on the release branch);
PR-head dispatch (smoke arbitrary PR refs); multi-box; artifact upload of the
full smoke log (today: result is in the status description).
