#!/usr/bin/env bash
# Convenience wrapper around robots/dual_franka/dual_franka_manual_call.py.
#
# Examples:
#   robots/dual_franka/run_manual_skill.sh --list-primitives
#   robots/dual_franka/run_manual_skill.sh --schema segment
#   robots/dual_franka/run_manual_skill.sh --primitive view_env_state
#   robots/dual_franka/run_manual_skill.sh --primitive vla_right_grasp --params '{"prompt":"grasp the cup","max_chunks":10}' --timeout-s 1800

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/rpent_live_env.sh"

cd "${RPENT_REPO_ROOT}"

exec .venv/bin/python robots/dual_franka/dual_franka_manual_call.py \
  --env-endpoint "${RPENT_ENV_ENDPOINT}" \
  --vla-endpoint "${RPENT_VLA_ENDPOINT}" \
  --sam3-endpoint "${RPENT_SAM3_ENDPOINT}" \
  "$@"
