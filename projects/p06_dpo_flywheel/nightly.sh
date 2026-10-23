#!/usr/bin/env bash
# Nightly DPO flywheel. Cron example (02:30 UTC):  30 2 * * * /path/to/repo/projects/p06_dpo_flywheel/nightly.sh
# Env: EVENTS (feedback JSONL), STATE (state dir), EVALKIT_TRAIN=1 to run real peft/trl training.
set -euo pipefail
REPO="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$REPO"
PY="${PY:-$REPO/.venv/bin/python}"
EVENTS="${EVENTS:-projects/p06_dpo_flywheel/data/feedback.jsonl}"
STATE="${STATE:-projects/p06_dpo_flywheel/state}"
DATE="${DATE:-$(date -u +%F)}"
mkdir -p "$STATE"
"$PY" -m projects.p06_dpo_flywheel.cli nightly --events "$EVENTS" --state "$STATE" --date "$DATE" \
  --json "$STATE/nightly_$DATE.json" | tee "$STATE/nightly_$DATE.md"
