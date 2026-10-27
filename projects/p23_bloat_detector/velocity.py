"""Optional git velocity: commits per day, the busiest 48 h window, added/deleted ratio.

`git` is only invoked through an injectable `runner(args, cwd) -> str`, so tests pass a fake that
returns a canned log and nothing here needs a real repository.

Format requested from git (one commit header line then numstat lines, blank line between commits):
    <sha>|<unix_ts>
    <added>\t<deleted>\t<path>
"""
from __future__ import annotations

import subprocess
from collections import Counter
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from typing import Any

Runner = Callable[[list[str], str], str]


@dataclass
class Commit:
    sha: str
    ts: int
    added: int
    deleted: int


@dataclass
class Velocity:
    days: int
    n_commits: int
    commits_per_day: dict[str, int]
    max_commits_48h: int
    busiest_window_start: str | None
    lines_added: int
    lines_deleted: int
    add_delete_ratio: float   # added / max(1, deleted); > 1 means the codebase is growing
    busiest_days: list[tuple[str, int]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def git_runner(args: list[str], cwd: str) -> str:
    return subprocess.run(["git", *args], cwd=cwd, capture_output=True, text=True, check=True).stdout


def is_git_repo(path: str, runner: Runner = git_runner) -> bool:
    try:
        return runner(["rev-parse", "--is-inside-work-tree"], path).strip() == "true"
    except Exception:  # an injected runner may raise anything: treat as not a repo  # noqa: BLE001
        return False


def parse_log(text: str) -> list[Commit]:
    commits: list[Commit] = []
    cur: Commit | None = None
    for raw in text.splitlines():
        line = raw.strip()
        if not line:
            continue
        if "|" in line and "\t" not in line:
            sha, ts = line.split("|", 1)
            cur = Commit(sha.strip(), int(ts.strip()), 0, 0)
            commits.append(cur)
        elif cur is not None and "\t" in line:
            a, d, *_ = line.split("\t")
            cur.added += int(a) if a.isdigit() else 0     # '-' for binary files
            cur.deleted += int(d) if d.isdigit() else 0
    return commits


def max_in_window(timestamps: list[int], window_s: int = 48 * 3600) -> tuple[int, int | None]:
    """Sliding window over sorted timestamps: largest count with ts[j] - ts[i] <= window_s."""
    ts = sorted(timestamps)
    best, start, i = 0, None, 0
    for j in range(len(ts)):
        while ts[j] - ts[i] > window_s:
            i += 1
        if j - i + 1 > best:
            best, start = j - i + 1, ts[i]
    return best, start


def compute_velocity(commits: list[Commit], days: int) -> Velocity:
    per_day = Counter(datetime.fromtimestamp(c.ts, tz=UTC).strftime("%Y-%m-%d") for c in commits)
    best, start = max_in_window([c.ts for c in commits])
    added = sum(c.added for c in commits)
    deleted = sum(c.deleted for c in commits)
    return Velocity(
        days=days, n_commits=len(commits), commits_per_day=dict(sorted(per_day.items())),
        max_commits_48h=best,
        busiest_window_start=datetime.fromtimestamp(start, tz=UTC).strftime("%Y-%m-%d %H:%M") if start else None,
        lines_added=added, lines_deleted=deleted, add_delete_ratio=round(added / max(1, deleted), 3),
        busiest_days=per_day.most_common(3))


def git_velocity(repo_dir: str, days: int = 30, runner: Runner = git_runner) -> Velocity | None:
    """None when `repo_dir` is not a git work tree (velocity is optional everywhere)."""
    if not is_git_repo(repo_dir, runner):
        return None
    text = runner(["log", f"--since={days} days ago", "--pretty=format:%H|%ct", "--numstat", "--no-merges"], repo_dir)
    return compute_velocity(parse_log(text), days)
