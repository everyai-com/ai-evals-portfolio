"""Post or update the gate comment on a GitHub PR via the REST API (urllib only).

The comment body carries a hidden HTML marker; an existing comment with the marker is PATCHed
instead of creating a duplicate, so re-runs on the same PR update one comment in place.

    python -m projects.p04_ci_regression_gate.post_comment --repo owner/name --pr 12 \
        --body-file gate_comment.md          # token from $GITHUB_TOKEN

`transport(method, url, headers, body) -> (status, json)` is injectable for tests.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import urllib.request
from typing import Any, Callable

from projects.p04_ci_regression_gate.core import COMMENT_MARKER

Transport = Callable[[str, str, dict[str, str], bytes | None], tuple[int, Any]]
API = "https://api.github.com"


def urllib_transport(method: str, url: str, headers: dict[str, str], body: bytes | None) -> tuple[int, Any]:
    req = urllib.request.Request(url, data=body, method=method, headers=headers)
    with urllib.request.urlopen(req, timeout=30) as resp:  # noqa: S310 (fixed https host)
        raw = resp.read()
        return resp.status, (json.loads(raw) if raw else None)


def post_or_update_comment(repo: str, pr_number: int, body: str, token: str,
                           transport: Transport = urllib_transport, marker: str = COMMENT_MARKER,
                           api: str = API) -> dict[str, Any]:
    """Create the comment if no comment on the PR contains `marker`, else update that one.
    Returns {"action": "created"|"updated", "id": comment_id}."""
    if marker not in body:
        body = marker + "\n" + body
    headers = {"Authorization": f"Bearer {token}", "Accept": "application/vnd.github+json",
               "Content-Type": "application/json", "User-Agent": "eval-gate"}
    existing_id = None
    page = 1
    while True:
        status, comments = transport("GET", f"{api}/repos/{repo}/issues/{pr_number}/comments?per_page=100&page={page}",
                                     headers, None)
        if status != 200:
            raise RuntimeError(f"GitHub API GET comments failed with status {status}")
        for c in comments or []:
            if marker in (c.get("body") or ""):
                existing_id = c["id"]
                break
        if existing_id is not None or not comments or len(comments) < 100:
            break
        page += 1
    payload = json.dumps({"body": body}).encode("utf-8")
    if existing_id is not None:
        status, data = transport("PATCH", f"{api}/repos/{repo}/issues/comments/{existing_id}", headers, payload)
        if status not in (200, 201):
            raise RuntimeError(f"GitHub API PATCH failed with status {status}")
        return {"action": "updated", "id": existing_id}
    status, data = transport("POST", f"{api}/repos/{repo}/issues/{pr_number}/comments", headers, payload)
    if status not in (200, 201):
        raise RuntimeError(f"GitHub API POST failed with status {status}")
    return {"action": "created", "id": data["id"]}


class FakeGitHub:
    """In-memory GitHub comments API for tests: records every request and returns stable ids."""

    def __init__(self) -> None:
        self.comments: list[dict[str, Any]] = []
        self.requests: list[tuple[str, str]] = []
        self._next = 1

    def __call__(self, method: str, url: str, headers: dict[str, str], body: bytes | None) -> tuple[int, Any]:
        self.requests.append((method, url))
        if method == "GET":
            return 200, list(self.comments)
        data = json.loads(body or b"{}")
        if method == "POST":
            c = {"id": self._next, "body": data["body"]}
            self._next += 1
            self.comments.append(c)
            return 201, c
        if method == "PATCH":
            cid = int(url.rsplit("/", 1)[1])
            for c in self.comments:
                if c["id"] == cid:
                    c["body"] = data["body"]
                    return 200, c
            return 404, None
        return 405, None


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="post_comment", description=__doc__.split("\n")[0])
    p.add_argument("--repo", required=True, help="owner/name")
    p.add_argument("--pr", required=True, type=int)
    p.add_argument("--body-file", required=True)
    p.add_argument("--token", default=None, help="defaults to $GITHUB_TOKEN")
    a = p.parse_args(argv)
    token = a.token or os.environ.get("GITHUB_TOKEN")
    if not token:
        print("no token: pass --token or set GITHUB_TOKEN", file=sys.stderr)
        return 2
    with open(a.body_file, encoding="utf-8") as f:
        body = f.read()
    res = post_or_update_comment(a.repo, a.pr, body, token)
    print(f"{res['action']} comment {res['id']} on {a.repo}#{a.pr}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
