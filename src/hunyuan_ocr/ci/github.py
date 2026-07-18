# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 AIwork4me
"""Thin `gh api` wrapper for the GPU-CI bridge, over **commit statuses**
(user-token read+write). The `runner` callable is injected so tests never touch
the network."""

from __future__ import annotations

import json
import subprocess

from hunyuan_ocr.ci.models import CHECK_NAME, SmokeStatus


def _gh(argv: list[str]) -> str:
    """Default runner: shell out to `gh api ...` and return stdout (JSON)."""
    cp = subprocess.run(["gh", *argv], capture_output=True, text=True, check=False)
    if cp.returncode != 0:
        raise RuntimeError(f"gh {' '.join(argv)} failed (rc={cp.returncode}): {cp.stderr.strip()}")
    return cp.stdout


class GitHubClient:
    """Minimal `gh api` client over commit statuses. `runner(argv)->str` injectable."""

    def __init__(self, owner: str, repo: str, *, runner=_gh):
        self.owner = owner
        self.repo = repo
        self._run = runner

    @property
    def _base(self) -> str:
        return f"repos/{self.owner}/{self.repo}"

    def ref_to_sha(self, ref: str) -> str:
        # pass through a raw 40-char SHA without an API call
        if len(ref) == 40 and all(c in "0123456789abcdef" for c in ref.lower()):
            return ref
        endpoint = ref if ref.startswith("refs/") else f"refs/heads/{ref}"
        out = json.loads(self._run(["api", f"{self._base}/git/{endpoint}"]))
        return out["object"]["sha"]

    def latest_tag(self, prefix: str = "v") -> tuple[str, str] | None:
        """Best-effort latest `<prefix>*` tag, dereferenced to its COMMIT sha
        (annotated tags point at a tag object, not the commit). Degrades to None
        on any API error (e.g. 404 / no tags) so the poller watches `main` only."""
        try:
            out = json.loads(self._run(["api", f"{self._base}/git/refs/tags"]))
        except (RuntimeError, ValueError):
            return None
        matches = [t for t in out if t["ref"].rsplit("/", 1)[-1].startswith(prefix)]
        if not matches:
            return None
        last = matches[-1]  # refs come back sorted; last == alphabetically latest
        name = last["ref"].rsplit("/", 1)[-1]
        obj = last["object"]
        sha = obj["sha"]
        if obj.get("type") == "tag":  # annotated tag → dereference to the commit
            try:
                tagobj = json.loads(self._run(["api", f"{self._base}/git/tags/{sha}"]))
                sha = tagobj["object"]["sha"]
            except (RuntimeError, ValueError, KeyError):
                return None
        return name, sha

    def list_smoke_statuses(self, sha: str) -> list[SmokeStatus]:
        """Commit statuses for our context on `sha`, most-recent-first."""
        out = json.loads(self._run(["api", f"{self._base}/commits/{sha}/statuses"]))
        res = []
        for s in out:
            if s.get("context") != CHECK_NAME:
                continue
            res.append(
                SmokeStatus(
                    sha=sha,
                    context=s["context"],
                    state=s["state"],
                    created_at=s.get("updated_at") or s.get("created_at"),
                    target_url=s.get("target_url"),
                )
            )
        return res

    def create_status(self, sha: str, *, state: str, description: str, target_url: str = "") -> None:
        args = [
            "api",
            "--method",
            "POST",
            f"{self._base}/statuses/{sha}",
            "-f",
            f"state={state}",
            "-f",
            f"context={CHECK_NAME}",
            "-f",
            f"description={description[:140]}",
        ]
        if target_url:
            args += ["-f", f"target_url={target_url}"]
        self._run(args)
