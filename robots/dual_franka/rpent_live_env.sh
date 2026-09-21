#!/usr/bin/env bash
# Shared environment for dual-Franka live runs.
#
# Source this file before launching RPent robot runners or manual skill tests:
#
#   source robots/dual_franka/rpent_live_env.sh
#
# It intentionally keeps Codex state under this RPent checkout so live robot
# sessions do not reuse the user's normal ~/.codex history/config/memory.

if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
  echo "This file must be sourced, not executed: source robots/dual_franka/rpent_live_env.sh" >&2
  exit 2
fi

_RPENT_LIVE_ENV_SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export RPENT_REPO_ROOT="${RPENT_REPO_ROOT:-$(cd "${_RPENT_LIVE_ENV_SCRIPT_DIR}/../.." && pwd)}"
export RPENT_TEST_ROOT="${RPENT_TEST_ROOT:-$(cd "${RPENT_REPO_ROOT}/.." && pwd)}"

# External checkout used by RPent/Ray workers.
export RLINF_REPO_PATH="${RLINF_REPO_PATH:-${RPENT_TEST_ROOT}/RLinf}"

# Isolate Codex records/config/cache/memory used by robot-control sessions from
# the user's normal assistant-for-coding Codex state.
export CODEX_HOME="${RPENT_CODEX_HOME:-${RPENT_REPO_ROOT}/.codex-rpent-live}"
export CODEX_SQLITE_HOME="${CODEX_HOME}"
export RPENT_LIVE_MEMORY_DIR="${RPENT_LIVE_MEMORY_DIR:-${CODEX_HOME}/memory}"
mkdir -p "${CODEX_HOME}" "${RPENT_LIVE_MEMORY_DIR}"

# Do not inherit provider credentials/endpoints from a coding session.
# With no dedicated API key, authenticate separately in RPENT_CODEX_HOME.
unset CODEX_API_KEY CODEX_BASE_URL OPENAI_API_KEY OPENAI_BASE_URL RPENT_CODEX_PROVIDER_KEY
if [[ -n "${RPENT_CODEX_API_KEY:-}" ]]; then
  export CODEX_API_KEY="${RPENT_CODEX_API_KEY}"
fi
if [[ -n "${RPENT_CODEX_BASE_URL:-}" ]]; then
  export CODEX_BASE_URL="${RPENT_CODEX_BASE_URL}"
fi

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
export RPENT_ROBOT_CONFIG="${RPENT_ROBOT_CONFIG:-${RPENT_REPO_ROOT}/robots/dual_franka/config/example.yaml}"

# Planner defaults for reproducible live runs. Per-run command line flags still
# win inside RPent; these variables are used by the wrapper scripts below.
export RPENT_PLANNER="${RPENT_PLANNER:-codex}"
export RPENT_CODEX_MODEL="${RPENT_CODEX_MODEL:-gpt-5.5}"
export RPENT_REASONING_EFFORT="${RPENT_REASONING_EFFORT:-medium}"
export CODEX_MODEL="${RPENT_CODEX_MODEL}"
export CODEX_SERVICE_TIER="${RPENT_CODEX_SERVICE_TIER:-fast}"
export RPENT_TASK_ID="${RPENT_TASK_ID:-3}"
export RPENT_RUN_TAG="${RPENT_RUN_TAG:-dirty-clean-t3}"

unset _RPENT_LIVE_ENV_SCRIPT_DIR
