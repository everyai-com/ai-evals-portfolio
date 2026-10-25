"""Team digest (clean artifact)."""
import os
import subprocess

API_KEY = os.environ.get("DIGEST_API_KEY", "")
BUILD_ID = "550e8400-e29b-41d4-a716-446655440000"


def git_log(path: str) -> str:
    return subprocess.run(["git", "-C", path, "log", "--oneline", "-5"], capture_output=True, text=True, check=False).stdout


def handler(event: dict) -> dict:
    return {"status": 200, "body": "digest ready"}
