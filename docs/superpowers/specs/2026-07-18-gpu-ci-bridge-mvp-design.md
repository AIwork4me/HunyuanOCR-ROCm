# GPU-CI Bridge — MVP Design (gfx1100 on Radeon Cloud / Docker)

**Date:** 2026-07-18
**Status:** design (awaiting review) → writing-plans
**Branch:** `gpu-ci-bridge-mvp`
**Target repo:** `AIwork4me/HunyuanOCR-ROCm` (public)

## 1. Context & goal

GitHub-hosted Actions runners have **no AMD GPU**, so the repo's real gfx1100
smoke (`rocm-runner-preflight` + `scripts/rocm_smoke.sh`) can never run there. The
only GPU we have is a single **Radeon Cloud (anruicloud)** instance: a Docker
container with `jupyter-lab` as PID 1 (no systemd init), behind a MITM proxy,
with **no inbound** reachability from GitHub, on which every smoke prerequisite
already lives (4× gfx1100, weights, built `llama-server`, scorer venv).

This spec defines an **MVP that proves the path end-to-end on the current Docker
box** ("0→1"), and a **lessons/method writeup + an anruicloud requirements list**
that scope the production-grade version. The MVP is the highest-value
verification: it exercises the full chain on real hardware and turns every
assumption into measured data.

### Definition of Done (the MVP must demonstrate this, live)

On `main` HEAD (`6ad5e711`): a manual `workflow_dispatch` from GitHub → the
nohup'd poller on the box picks it up within ~minutes → runs the **real** 1-page
smoke on gfx1100 (start `llama-server` → predict → validate → manifest verify) →
a **green** `gpu-smoke (gfx1100)` Check Run appears in the repo's Checks UI, with
an output summary carrying the ROCm/torch/llama.cpp env + manifest counts +
latency. The whole chain touches **only `api.github.com`** (never
git-receive-pack).

## 2. Non-goals (deferred to the anruicloud-backed production version)

- **Always-on / auto-start**: systemd unit or anruicloud startup hook so the
  poller survives container restart. (MVP = `nohup`; this is the #1 documented gap.)
- **Inbound webhook** (GitHub push → box) to remove polling latency. Needs anruicloud ingress.
- **`push: tags v*` → auto release-gate** + required-check on the release branch.
- **PR-branch dispatch** (smoke on arbitrary PR heads). MVP dispatches `main` + latest tag only.
- Multi-box/HA, metrics/alerting, artifact upload to GitHub.

## 3. Architecture

```
 GitHub (control plane)                         this gfx1100 Docker box (data plane)
 ┌───────────────────────────────────┐         ┌──────────────────────────────────────┐
 │ .github/workflows/gpu-smoke.yml   │ ① click │ src/hunyuan_ocr/ci/poller.py (nohup)  │
 │   workflow_dispatch {ref}         │  Run    │   loop every 180s:                    │
 │   (GitHub-hosted, CPU only)       │ ───────>│ ② GET api.github.com check-runs       │
 │   → create check-run              │         │     name=gpu-smoke (gfx1100), queued  │
 │     name=gpu-smoke (gfx1100)      │         │ ③ flock (single smoke at a time)      │
 │     status=queued, on <ref SHA>   │         │   checkout <SHA>, run TRUSTED         │
 │                                   │         │   scripts/rocm_smoke.sh on gfx1100    │
 │                                   │ ④ PATCH │ ④ PATCH check-run completed           │
 │ Checks UI: gpu-smoke (gfx1100)    │ <───────│     success|failure + env summary     │
 │   green ⇒ DoD met                 │         │ ⑤ timeout-sweep stale queued runs     │
 └───────────────────────────────────┘         └──────────────────────────────────────┘
```

**Why this shape:** GitHub is the control plane (dispatch + the Check Run that
gates); the box is the data plane (polls + executes + reports). Everything goes
through `api.github.com` (verified 200, **not** intercepted by the proxy),
sidestepping the known git-receive-pack breakage. The Check Run is the single
state object for request → heartbeat → result.

## 4. Components & interfaces

### 4.1 `.github/workflows/gpu-smoke.yml` (GitHub-hosted, CPU)
- Triggers: `workflow_dispatch` with input `ref` (default `main`).
  (A `push: tags: ['v*']` trigger is added in the production version, not the MVP.)
- One job, `runs-on: ubuntu-latest`, steps: checkout `ref`; resolve `ref` → SHA;
  `gh api` **create** a Check Run:
  - `name`: `gpu-smoke (gfx1100)`
  - `head_sha`: resolved SHA
  - `status`: `queued`
  - `external_id`: `<SHA>` (idempotency key for the poller)
  - `output`: `{title: "GPU smoke requested", summary: "waiting for the gfx1100 runner on Radeon Cloud; dispatched from <run url>"}`
- Then exit. No runner needed on the GitHub side.

### 4.2 `src/hunyuan_ocr/ci/__init__.py` + `src/hunyuan_ocr/ci/poller.py`
The bridge. Pure-Python, CPU-testable. Public surface:

```python
# poller.py
CHECK_NAME = "gpu-smoke (gfx1100)"
STALE_AFTER_SEC = 30 * 60

class GitHubClient:                 # thin wrapper over `gh api` (auth already on box)
    def list_queued(self, ref_sha: str) -> list[CheckRun]: ...
    def set_in_progress(self, check_run_id: int, sha: str) -> None: ...
    def complete(self, check_run_id: int, *, conclusion: str, output: dict) -> None: ...
    def ref_to_sha(self, ref: str) -> str: ...

def decide(check_run: CheckRun, now: float) -> str:
    """→ 'run' | 'skip_done' | 'timeout'. Pure; unit-tested."""

def run_smoke(sha: str, *, repo_dir: Path, smoke_script: Path,
              env: dict, timeout_s: int) -> SmokeResult:
    """checkout <sha> into a temp workdir; run the TRUSTED smoke_script with
    REPO=<workdir>; return SmokeResult(ok, env_summary, manifest, latency, log_tail)."""

def build_output(result: SmokeResult, sha: str) -> dict:
    """Assemble the Check Run output payload (env summary + manifest + latency)."""

def main(argv=None) -> int:        # the loop; --once for a single pass (testable)
```

**State machine (one `--once` pass):**
1. Acquire `flock` on `~/.rocm_ci_poller.lock` (non-blocking; if held, exit 0 — another pass is running).
2. For each watched ref (`main`, latest `v*` tag) → `ref_to_sha` → `list_queued(sha)`.
3. For each queued Check Run (oldest first):
   - If a **completed** `gpu-smoke` already exists for this SHA (`skip_done`) → skip (idempotency).
   - Else `set_in_progress` (records `started_at`), then `run_smoke(sha)` with a hard `timeout_s`.
   - On success → `complete(conclusion="success", output=build_output(...))`.
   - On failure/exception → `complete(conclusion="failure", output={..., log tail})`.
4. **Stale sweep:** any Check Run still `queued` or `in_progress` with age > `STALE_AFTER_SEC` → `complete(conclusion="failure", output="timeout — gfx1100 runner offline? is the Radeon Cloud box up?")`. No silent hangs.
5. `main` loops `--once` every `POLL_INTERVAL=180s`.

### 4.3 `scripts/rocm_smoke.sh` (TRUSTED harness — already exists, minor edit)
- Add `--ref <sha>` / `REPO` handling: the poller invokes the **box's** copy of
  this script (from the trusted `main` checkout) with `REPO=<temp checkout of the
  SHA under test>`. The script starts `llama-server`, predicts via
  `$REPO/scripts/run_phase2_vllm.py`, validates, and verifies the manifest.
- Trust split: the **harness** (server lifecycle, assertions, manifest verify) is
  the trusted main copy; the **model driver** (`run_phase2_vllm.py`) is the
  code-under-test from the dispatched SHA. A malicious SHA can affect only its
  own checked-out predict path, not the harness.
- `HIP_VISIBLE_DEVICES` pins one free GPU. Existing `trap` already kills the server on exit.

### 4.4 1-page smoke input (`scripts/make_smoke_input.py`, machine-local)
- Reads the local full GT (`/root/ocr-eval/OmniDocBench_data/OmniDocBench.json`)
  and the canary manifest (`eval/canary_148.manifest.json`), writes a
  **deterministic 1-page** GT (`OmniDocBench_smoke_1page.json`) using the
  canary's first page (stable, locked) + confirms its image exists.
- Output + image dir are **machine-local and gitignored** (e.g.
  `/root/ocr-eval/smoke/`). Never committed.
- The poller sets `HUNYUANOCR_SMOKE_GT` / `HUNYUANOCR_SMOKE_IMAGES` to these.

## 5. Check Run protocol

| field | created by workflow | set by poller (in_progress) | final (success) | final (failure) |
|---|---|---|---|---|
| `name` | `gpu-smoke (gfx1100)` | — | — | — |
| `status` | `queued` | `in_progress` | `completed` | `completed` |
| `conclusion` | — | — | `success` | `failure` |
| `external_id` | `<SHA>` | — | — | — |
| `output.title` | "GPU smoke requested" | "Running on gfx1100…" | "gpu-smoke PASSED" | "gpu-smoke FAILED" |
| `output.summary` | dispatch url | started_at | env + manifest + latency | error + log tail |

## 6. Security model

- The poller runs **only** for refs the maintainer explicitly dispatched (the
  workflow's `ref` input). It **never** auto-runs on PR-open or push.
- The harness is the **trusted pinned** box copy; only the model driver from the
  dispatched SHA executes. Fork PRs can only be smoked if the maintainer
  explicitly dispatches their ref.
- Weights/paths flow via env vars; the env summary recorded in the Check Run
  reports ROCm/torch/llama.cpp **versions and the GPU id**, never absolute weight
  paths or secrets.
- The `gh` token already on the box (`repo` scope) covers `checks:write` +
  `contents:read` (checkout). No new secret management for the MVP.

## 7. Persistence (honest)

- The poller runs as `nohup python -m hunyuan_ocr.ci.poller >~/.rocm_ci_poller.log 2>&1 & disown`,
  started from a JupyterLab terminal. It survives while the container is up.
- **No systemd** (PID 1 is `jupyter-lab`), so it does **not** auto-restart on
  container/anruicloud restart — the operator restarts it manually, and queued
  smokes simply wait (they do not drop).
- This is the MVP's #1 documented fragility and the top item on the anruicloud
  ask-list (§9).

## 8. Testing plan (CPU, in-repo; gated by the existing CPU CI matrix)

`tests/test_ci_poller.py` (no network, no GPU):
- `decide()` over fixture Check Runs → `run` / `skip_done` / `timeout` (incl. boundary at `STALE_AFTER_SEC`).
- idempotency: a SHA with an existing completed run → `skip_done`.
- `build_output()` assembles env summary + manifest counts + latency from a fixture `SmokeResult`.
- `run_smoke()` with a fake harness script + fake `gh`/checkout → asserts the trust split (trusted harness path, SHA workdir) and timeout enforcement.
- `flock`: two `--once` invocations under the same lock → the second is a no-op.
- A `repo`-local `--once` dry-run mode (`--dry-run`) that lists what it *would* do, for safe verification.

**Live end-to-end demo (manual, captured as evidence):** start the poller (nohup),
`gh workflow run gpu-smoke.yml -f ref=main`, watch the poller log, observe the
green Check Run on `6ad5e711`. Capture timing + env into the lessons doc.

## 9. Lessons/method doc + anruicloud requirements (the production handoff)

`docs/ci/gpu-ci-bridge.md` (written from the MVP run) contains:
- **Method:** GitHub control-plane + box data-plane; pin everything to
  `api.github.com`; Check Run as single state object; trusted-harness +
  explicit-ref security model; flock/idempotency/timeout-sweep.
- **Measured data:** end-to-end latency, poll interval vs. pickup time, model-load
  + single-page predict time, VRAM used.
- **Friction on Docker (prioritized ask for anruicloud):**
  1. **P0 — persistence:** a startup-hook / lifecycle-script, or a real
     systemd-init mode, so the poller auto-starts. (Without it, the runner must be
     manually restarted after every container/anruicloud restart.)
  2. **P1 — long-lived process guarantee:** confirmation a `nohup`'d process
     survives Jupyter session disconnects and anruicloud scheduling windows.
  3. **P2 (optional) — inbound HTTPS ingress:** a stable URL + auth so GitHub
     webhooks can push (removes polling latency).
  4. **GPU isolation:** pin one GPU via `HIP_VISIBLE_DEVICES`.
- **Final-solution mapping:** anruicloud (1)+(2) ⇒ poller goes production; adding
  (3) ⇒ webhook push (lower latency); a persistent systemd-init box with optional
  ingress ⇒ the native **self-hosted Actions runner** path. The MVP data decides.

## 10. Risks & mitigations

| risk | mitigation |
|---|---|
| `llama-server` headless start fails in container | `nohup` + existing `trap`; verified in the live demo |
| GPU contention with other box workloads | `HIP_VISIBLE_DEVICES` pin + 1-page (low VRAM); fail loud if no free GPU |
| poller dies (no systemd) | timeout-sweep marks the Check Run `failure` (visible, not silent); §9 P0 asks anruicloud to fix |
| `api.github.com` rate limits (polling) | 180s interval + conditional/etag headers; well within limits |
| check-run discovery across PR refs | MVP watches `main` + latest tag only; PR-head scan is a documented production extension |
| proxy interferes with `gh api` | `api.github.com` is verified NOT intercepted (200); `gh` uses it |

## 11. Open questions for review

- Watched refs = `main` + latest `v*` tag (MVP). Agree?
- `STALE_AFTER_SEC` = 30 min, `POLL_INTERVAL` = 180 s. Agree?
- Check Run name `gpu-smoke (gfx1100)` — stable enough to be a future required-check-on-release?
