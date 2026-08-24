#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Start the dual-Franka PhysicalAgent env server after joining the remote arm node
to the current local Ray head.

This script is intentionally conservative:
  - it does not run ray stop
  - it does not kill Franka controllers
  - it does not start a new local Ray head
  - it only asks the remote host to join the existing local Ray head

Usage:
  scripts/start_dual_franka_env_server.sh [options] [-- extra env-server args]

Options:
  --remote-host HOST       SSH host for the right-arm machine (default: master)
  --remote-node-rank N     RLINF_NODE_RANK for the remote node (default: 1)
  --head-addr ADDR         Ray head address, e.g. 192.168.121.164:6379
                           (default: /tmp/ray/ray_current_cluster)
  --transport-host HOST    Env RPC bind host (default: 127.0.0.1)
  --transport-port PORT    Env RPC bind port (default: 5556)
  --skip-remote-ray        Do not SSH/start remote Ray worker
  --remote-only            Start/check remote Ray worker and exit
  -h, --help               Show this help

Environment overrides:
  RPENT_ROOT               Default: repo root containing this script
  RLINF_ROOT               Default: sibling ../rlinf
  REMOTE_RLINF_ROOT        Default: same path as RLINF_ROOT
  VENV                     Default: $RLINF_ROOT/requirements/.venv
  REMOTE_VENV              Default: $REMOTE_RLINF_ROOT/requirements/.venv
  TASK_DESCRIPTION         Default: dual arm physical agent task

Examples:
  scripts/start_dual_franka_env_server.sh

  scripts/start_dual_franka_env_server.sh --remote-only

  scripts/start_dual_franka_env_server.sh -- \
    --robot-config /path/to/robot_config.yaml

Server-side config:
  Robot, camera, workspace, and controller settings are read from
  robots/dual_franka/robot_config.yaml by default.
EOF
}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RPENT_ROOT="${RPENT_ROOT:-$(cd "$SCRIPT_DIR/.." && pwd)}"
RLINF_ROOT="${RLINF_ROOT:-$(cd "$RPENT_ROOT/../rlinf" && pwd)}"
REMOTE_RLINF_ROOT="${REMOTE_RLINF_ROOT:-$RLINF_ROOT}"
VENV="${VENV:-$RLINF_ROOT/requirements/.venv}"
REMOTE_VENV="${REMOTE_VENV:-$REMOTE_RLINF_ROOT/requirements/.venv}"

REMOTE_HOST="${REMOTE_HOST:-master}"
REMOTE_NODE_RANK="${REMOTE_NODE_RANK:-1}"
HEAD_ADDR="${HEAD_ADDR:-}"
TRANSPORT_HOST="${TRANSPORT_HOST:-127.0.0.1}"
TRANSPORT_PORT="${TRANSPORT_PORT:-5556}"
TASK_DESCRIPTION="${TASK_DESCRIPTION:-dual arm physical agent task}"
SKIP_REMOTE_RAY=0
REMOTE_ONLY=0
SERVER_ARGS=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    --remote-host)
      REMOTE_HOST="${2:?--remote-host requires a value}"
      shift 2
      ;;
    --remote-node-rank)
      REMOTE_NODE_RANK="${2:?--remote-node-rank requires a value}"
      shift 2
      ;;
    --head-addr)
      HEAD_ADDR="${2:?--head-addr requires a value}"
      shift 2
      ;;
    --transport-host)
      TRANSPORT_HOST="${2:?--transport-host requires a value}"
      shift 2
      ;;
    --transport-port)
      TRANSPORT_PORT="${2:?--transport-port requires a value}"
      shift 2
      ;;
    --skip-remote-ray)
      SKIP_REMOTE_RAY=1
      shift
      ;;
    --remote-only)
      REMOTE_ONLY=1
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    --)
      shift
      SERVER_ARGS+=("$@")
      break
      ;;
    *)
      SERVER_ARGS+=("$1")
      shift
      ;;
  esac
done

if [[ ! -d "$RPENT_ROOT" ]]; then
  echo "RPENT_ROOT does not exist: $RPENT_ROOT" >&2
  exit 2
fi
if [[ ! -d "$RLINF_ROOT" ]]; then
  echo "RLINF_ROOT does not exist: $RLINF_ROOT" >&2
  exit 2
fi
if [[ ! -x "$VENV/bin/python" ]]; then
  echo "VENV python not found: $VENV/bin/python" >&2
  exit 2
fi

if [[ -z "$HEAD_ADDR" ]]; then
  if [[ -s /tmp/ray/ray_current_cluster ]]; then
    HEAD_ADDR="$(tr -d '[:space:]' < /tmp/ray/ray_current_cluster)"
  else
    echo "No Ray head address found. Start the local Ray head first, or pass --head-addr HOST:6379." >&2
    exit 2
  fi
fi

quote() {
  printf '%q' "$1"
}

start_remote_ray() {
  echo "[dual-franka] local Ray head: $HEAD_ADDR"
  echo "[dual-franka] joining remote Ray node: $REMOTE_HOST (rank $REMOTE_NODE_RANK)"

  local remote_script
  remote_script='
set -euo pipefail
cd "$REMOTE_RLINF_ROOT"
source "$REMOTE_VENV/bin/activate"
export PYTHONPATH="$REMOTE_RLINF_ROOT:${PYTHONPATH:-}"
export RLINF_NODE_RANK="$REMOTE_NODE_RANK"

if [[ -s /tmp/ray/ray_current_cluster ]]; then
  current="$(tr -d "[:space:]" < /tmp/ray/ray_current_cluster || true)"
  if [[ "$current" == "$HEAD_ADDR" ]]; then
    echo "[dual-franka] remote Ray already points at $current"
  elif pgrep -u "$USER" -f "raylet|gcs_server|dashboard|runtime_env_agent" >/dev/null 2>&1; then
    echo "[dual-franka] remote Ray appears to be running but points at $current, expected $HEAD_ADDR." >&2
    echo "[dual-franka] Not touching existing Ray processes. Stop/rejoin manually when safe." >&2
    exit 3
  else
    echo "[dual-franka] ray start --address=$HEAD_ADDR"
    ray start --address="$HEAD_ADDR" --disable-usage-stats
  fi
else
  echo "[dual-franka] ray start --address=$HEAD_ADDR"
  ray start --address="$HEAD_ADDR" --disable-usage-stats
fi
'

  ssh "$REMOTE_HOST" \
    "HEAD_ADDR=$(quote "$HEAD_ADDR") REMOTE_RLINF_ROOT=$(quote "$REMOTE_RLINF_ROOT") REMOTE_VENV=$(quote "$REMOTE_VENV") REMOTE_NODE_RANK=$(quote "$REMOTE_NODE_RANK") bash -lc $(quote "$remote_script")"
}

if [[ "$SKIP_REMOTE_RAY" -eq 0 ]]; then
  start_remote_ray
else
  echo "[dual-franka] skipping remote Ray join"
fi

if [[ "$REMOTE_ONLY" -eq 1 ]]; then
  echo "[dual-franka] remote-only requested; not starting env server"
  exit 0
fi

cd "$RPENT_ROOT"
source "$VENV/bin/activate"
export PYTHONPATH="$RPENT_ROOT:$RLINF_ROOT:${PYTHONPATH:-}"

echo "[dual-franka] starting env server on $TRANSPORT_HOST:$TRANSPORT_PORT"
exec python -u robots/dual_franka/env_server.py \
  --transport socket \
  --host "$TRANSPORT_HOST" \
  --port "$TRANSPORT_PORT" \
  --task-description "$TASK_DESCRIPTION" \
  "${SERVER_ARGS[@]}"
