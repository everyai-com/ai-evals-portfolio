"""Builds the deploy-gate demo artifact at runtime from a placeholder template.

The committed template (``data/sample_artifact_template/``) holds placeholders such as
``{{FAKE:slack_1}}`` instead of secret-shaped strings, so repository secret scanners (GitHub
push protection) never see a whole token. Each value below is split into two adjacent string
literals for the same reason; Python joins them at import. Every value is fabricated and grants
access to nothing.
"""
from __future__ import annotations

import re
import shutil
from pathlib import Path

HERE = Path(__file__).resolve().parent
TEMPLATE = HERE / "data" / "sample_artifact_template"
_PLACEHOLDER = re.compile(r"\{\{FAKE:([a-z0-9_]+)\}\}")

FAKE: dict[str, str] = {
    "aws_1": "AKI" "A4Q7ZP2M9XCV1BTR8",
    "aws_2": "AKI" "A9Z8Y7X6W5V4U3T2S",
    "aws_3": "AKI" "AFIXTUREFIXTURE01",
    "github_1": "ghp" "_q1W2e3R4t5Y6u7I8o9P0a1S2d3F4g5H6j7K8l9Z",
    "jwt_1": "eyJ" "hbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiJhcHAtMTIzIiwicm9sZSI6ImFkbWluIn0.sflKxwRJSMeKKF2QT4fwpMeJf36POk6yJV_adQssw5c",
    "slack_1": "xox" "b-123456789012-1234567890123-AbCdEfGhIjKlMnOpQrStUvWx",
    "stripe_live_1": "sk_" "live_4eC39HqLyjWDarjtT1zdp7dc",
    "stripe_test_1": "sk_" "test_xxxxxxxxxxxxxxxxxxxxxxxx",
}


def render(text: str) -> str:
    """Replace every ``{{FAKE:name}}`` placeholder; unknown names are an error, not a blank."""

    def sub(m: re.Match[str]) -> str:
        if m.group(1) not in FAKE:
            raise KeyError(f"unknown fake-secret placeholder {m.group(0)}")
        return FAKE[m.group(1)]

    return _PLACEHOLDER.sub(sub, text)


def materialize_sample(dest: str | Path) -> Path:
    """Copy the template to ``dest`` with placeholders filled in. ``dest`` must not already hold files."""
    dest = Path(dest)
    if dest.exists() and any(dest.iterdir()):
        raise FileExistsError(f"{dest} is not empty; refusing to overwrite it")
    shutil.copytree(TEMPLATE, dest, dirs_exist_ok=True)
    for path in dest.rglob("*"):
        if not path.is_file():
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        if "{{FAKE:" in text:
            path.write_text(render(text), encoding="utf-8")
    return dest
