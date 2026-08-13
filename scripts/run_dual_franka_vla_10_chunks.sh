#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RPENT_ROOT="${RPENT_ROOT:-$(cd "$SCRIPT_DIR/.." && pwd)}"
PYTHON="${PYTHON:-$RPENT_ROOT/.venv/bin/python}"

cd "$RPENT_ROOT"
export PYTHONPATH="$RPENT_ROOT:${PYTHONPATH:-}"

"$PYTHON" scripts/dual_franka_manual.py \
  --json resources/dual_franka/manual_calls/run_vla_skill_10_chunks.json \
  --timeout-s 1800
