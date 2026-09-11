#!/usr/bin/env bash
# Shared environment for dual-Franka live runs.
#
# Source this file before launching RPent robot runners or manual skill tests:
#
#   source scripts/rpent_live_env.sh
#
# It intentionally keeps Codex state under this RPent checkout so live robot
# sessions do not reuse the user's normal ~/.codex history/config/memory.

if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
  echo "This file must be sourced, not executed: source scripts/rpent_live_env.sh" >&2
  exit 2
fi

_RPENT_LIVE_ENV_SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export RPENT_REPO_ROOT="${RPENT_REPO_ROOT:-$(cd "${_RPENT_LIVE_ENV_SCRIPT_DIR}/.." && pwd)}"
export RPENT_TEST_ROOT="${RPENT_TEST_ROOT:-$(cd "${RPENT_REPO_ROOT}/.." && pwd)}"

# External checkout used by RPent/Ray workers.
export RLINF_REPO_PATH="${RLINF_REPO_PATH:-${RPENT_TEST_ROOT}/RLinf}"

# Isolate Codex records/config/cache/memory used by robot-control sessions from
# the user's normal assistant-for-coding Codex state.
export CODEX_HOME="${CODEX_HOME:-${RPENT_REPO_ROOT}/.codex-rpent-live}"
export RPENT_LIVE_MEMORY_DIR="${RPENT_LIVE_MEMORY_DIR:-${CODEX_HOME}/memory}"
mkdir -p "${CODEX_HOME}" "${RPENT_LIVE_MEMORY_DIR}"

if [[ -n "${PYTHONPATH:-}" ]]; then
  export PYTHONPATH="${RLINF_REPO_PATH}:${RPENT_REPO_ROOT}:${PYTHONPATH}"
else
  export PYTHONPATH="${RLINF_REPO_PATH}:${RPENT_REPO_ROOT}"
fi

# Shared live service endpoints. Override these before sourcing, or set them on
# the command line, if a run uses different ports.
export RPENT_ENV_ENDPOINT="${RPENT_ENV_ENDPOINT:-http://127.0.0.1:6001}"
export RPENT_VLA_ENDPOINT="${RPENT_VLA_ENDPOINT:-http://127.0.0.1:6000}"
export RPENT_SAM3_ENDPOINT="${RPENT_SAM3_ENDPOINT:-http://127.0.0.1:8114}"

# Hardware/config defaults for the current dual-Franka setup.
#
# PhysicalAgent alignment note: the calibration path intentionally points at
# the old deployed project because those hand-eye/base-frame numbers are the
# reference used by previous clean-desk logs.  Override RPENT_CALIBRATION_PATH
# for any other machine/table instead of treating this as a portable default.
export RPENT_ROBOT_CONFIG="${RPENT_ROBOT_CONFIG:-${RPENT_REPO_ROOT}/robots/dual_franka/config/example.yaml}"
export RPENT_CALIBRATION_PATH="${RPENT_CALIBRATION_PATH:-/home/raojiaji/nieyi/physicalagent/physical_agent/envs/dual_franka/calibration/hand_eye_calibration.json}"

# Planner defaults for reproducible live runs. Per-run command line flags still
# win inside RPent; these variables are used by the wrapper scripts below.
export RPENT_PLANNER="${RPENT_PLANNER:-codex}"
export RPENT_CODEX_MODEL="${RPENT_CODEX_MODEL:-gpt-5.5}"
export RPENT_REASONING_EFFORT="${RPENT_REASONING_EFFORT:-medium}"
export RPENT_TASK_ID="${RPENT_TASK_ID:-3}"
export RPENT_RUN_TAG="${RPENT_RUN_TAG:-dirty-clean-t3}"

unset _RPENT_LIVE_ENV_SCRIPT_DIR
