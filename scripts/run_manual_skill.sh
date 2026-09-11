#!/usr/bin/env bash
# Convenience wrapper around scripts/dual_franka_manual_call.py.
#
# Examples:
#   scripts/run_manual_skill.sh --list-primitives
#   scripts/run_manual_skill.sh --schema segment
#   scripts/run_manual_skill.sh --primitive view_env_state
#   scripts/run_manual_skill.sh --primitive vla_right_grasp --params '{"prompt":"grasp the cup","max_chunks":10}' --timeout-s 1800

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/rpent_live_env.sh"

cd "${RPENT_REPO_ROOT}"

exec .venv/bin/python scripts/dual_franka_manual_call.py \
  --env-endpoint "${RPENT_ENV_ENDPOINT}" \
  --vla-endpoint "${RPENT_VLA_ENDPOINT}" \
  --sam3-endpoint "${RPENT_SAM3_ENDPOINT}" \
  "$@"
