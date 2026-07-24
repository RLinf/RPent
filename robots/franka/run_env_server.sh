#!/usr/bin/env bash
# Launch the standalone Franka env server in the RLinf .venv with the
# serl_franka_controllers catkin workspace sourced.
#
# The agent (agent venv) connects to this over TCP. Two ways to use it:
#
#   1. Fixed port, then attach the agent via --env-endpoint:
#        bash robots/franka/run_env_server.sh --output-dir /tmp/franka_run \
#             --transport socket --host 127.0.0.1 --port 5599
#        # in the agent venv:
#        python rpent/cli/main.py --env franka \
#             --env-endpoint socket://127.0.0.1:5599 ...
#
#   2. Let the agent CLI spawn it (it invokes this script automatically):
#        python rpent/cli/main.py --env franka ...
#
# Override the machine-specific bits via env vars:
#   FRANKA_CATKIN_SETUP  catkin devel setup.bash (default: RLinf .venv workspace)
#   RLINF_VENV_PYTHON    python in the RLinf .venv
#   FRANKA_ROBOT_IP      robot IP (default 172.16.0.2)
set -euo pipefail

FRANKA_CATKIN_SETUP="${FRANKA_CATKIN_SETUP:-/home/franka/franka/RLinf/.venv/franka_catkin_ws/devel/setup.bash}"
RLINF_VENV_PYTHON="${RLINF_VENV_PYTHON:-/home/franka/franka/RLinf/.venv/bin/python}"
FRANKA_ROBOT_IP="${FRANKA_ROBOT_IP:-172.16.0.2}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENV_SERVER="${SCRIPT_DIR}/env_server.py"

if [[ ! -f "${FRANKA_CATKIN_SETUP}" ]]; then
  echo "!! catkin setup not found: ${FRANKA_CATKIN_SETUP}" >&2
  echo "   set FRANKA_CATKIN_SETUP to your serl_franka_controllers workspace." >&2
  exit 1
fi
if [[ ! -x "${RLINF_VENV_PYTHON}" ]]; then
  echo "!! RLinf venv python not found: ${RLINF_VENV_PYTHON}" >&2
  echo "   set RLINF_VENV_PYTHON to the RLinf .venv python." >&2
  exit 1
fi

# shellcheck disable=SC1090
source "${FRANKA_CATKIN_SETUP}"

exec "${RLINF_VENV_PYTHON}" "${ENV_SERVER}" --robot-ip "${FRANKA_ROBOT_IP}" "$@"
