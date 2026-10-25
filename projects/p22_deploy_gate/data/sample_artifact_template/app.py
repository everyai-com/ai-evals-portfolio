"""Expense helper (demo artifact with planted issues)."""
import os
import subprocess

from flask import Flask

app = Flask(__name__)

AWS_ACCESS_KEY_ID = "{{FAKE:aws_1}}"                      # planted: looks like a real key
STRIPE_KEY = "{{FAKE:stripe_test_1}}"                # placeholder: must NOT be flagged
INTERNAL_SIGNING_TOKEN = "dev-only-signing-token-not-a-secret-1234"  # deploygate:allow
ASSET_INTEGRITY = "sha256-47DEQpj8HBSa+/TImW+5JCeuQeRkm5NMpJWZG3hSuFU="  # content hash, allowed in .deploygateignore
BUILD_ID = "550e8400-e29b-41d4-a716-446655440000"              # UUID: hex, under the entropy bar
WEBHOOK_NONCE = "Qm7vX2pLk9ZrT4wNc8Yb3HsJ6dF1gA5eUo"           # planted: random string with no keyword
DB_PASSWORD = os.environ.get("DB_PASSWORD", "")                 # fine: read from the environment


def list_dir(path: str):
    return subprocess.run(f"ls -la {path}", shell=True, capture_output=True)  # planted: shell=True + f-string


@app.route("/")
def index():
    return "expenses"


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080)  # planted: binds its own server
