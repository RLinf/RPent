#!/usr/bin/env bash
# Run the dual-Franka dirty/clean sorting task with project-local Codex state.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/rpent_live_env.sh"

cd "${RPENT_REPO_ROOT}"

RUN_TAG="${RPENT_RUN_TAG:-dirty-clean-t3}"
OUT="${RPENT_OUTPUT_DIR:-${RPENT_REPO_ROOT}/logs/$(date +%Y%m%d-%H%M%S)-${RUN_TAG}}"
mkdir -p "${OUT}"

exec .venv/bin/python -m rpent.cli.main \
  --robot dual_franka \
  --task-id "${RPENT_TASK_ID:-3}" \
  --planner "${RPENT_PLANNER:-codex}" \
  --model "${RPENT_CODEX_MODEL:-gpt-5.5}" \
  --reasoning-effort "${RPENT_REASONING_EFFORT:-medium}" \
  --memory-profile local \
  --memory-dir "${RPENT_LIVE_MEMORY_DIR}" \
  --env-endpoint "${RPENT_ENV_ENDPOINT}" \
  --vla-endpoint "${RPENT_VLA_ENDPOINT}" \
  --sam3-endpoint "${RPENT_SAM3_ENDPOINT}" \
  --robot-config "${RPENT_ROBOT_CONFIG}" \
  --calibration-path "${RPENT_CALIBRATION_PATH}" \
  --output-dir "${OUT}" \
  "$@"
