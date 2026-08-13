#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Start the two-node Ray cluster required by the dual-Franka PhysicalAgent server.

This script only manages Ray startup:
  - starts/checks the local Ray head
  - SSHes to the remote arm host and starts/checks the Ray worker
  - writes /tmp/ray/ray_current_cluster with the head address
  - does not start the PhysicalAgent env server
  - does not run ray stop or kill existing processes

Usage:
  scripts/start_dual_franka_ray.sh [options]

Options:
  --local-node-ip IP      Local wired-network IP for Ray head
                          (default: 192.168.121.167)
  --remote-node-ip IP     Remote wired-network IP for Ray worker
                          (default: 192.168.121.168)
  --port PORT             Ray GCS/head port (default: 6379)
  --remote-host HOST      SSH host for remote machine (default: master)
  --remote-node-rank N    RLINF_NODE_RANK for remote machine (default: 1)
  --head-addr ADDR        Explicit head address, overrides local-ip/port
                          (default: <local-node-ip>:<port>)
  --local-only            Start/check only local Ray head
  --remote-only           Start/check only remote Ray worker
  --skip-remote           Alias for --local-only
  -h, --help              Show this help

Environment overrides:
  RPENT_ROOT              Default: repo root containing this script
  RLINF_ROOT              Default: sibling ../rlinf
  REMOTE_RLINF_ROOT       Default: same path as RLINF_ROOT
  VENV                    Default: $RLINF_ROOT/requirements/.venv
  REMOTE_VENV             Default: $REMOTE_RLINF_ROOT/requirements/.venv
  RLINF_COMM_NET_DEVICES  Optional network interface hint forwarded to both nodes

Examples:
  scripts/start_dual_franka_ray.sh

  scripts/start_dual_franka_ray.sh \
    --local-node-ip 192.168.121.167 \
    --remote-node-ip 192.168.121.168

  scripts/start_dual_franka_ray.sh --remote-only --head-addr 192.168.121.167:6379
EOF
}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RPENT_ROOT="${RPENT_ROOT:-$(cd "$SCRIPT_DIR/.." && pwd)}"
RLINF_ROOT="${RLINF_ROOT:-$(cd "$RPENT_ROOT/../rlinf" && pwd)}"
REMOTE_RLINF_ROOT="${REMOTE_RLINF_ROOT:-$RLINF_ROOT}"
VENV="${VENV:-$RLINF_ROOT/requirements/.venv}"
REMOTE_VENV="${REMOTE_VENV:-$REMOTE_RLINF_ROOT/requirements/.venv}"

LOCAL_NODE_IP="${LOCAL_NODE_IP:-192.168.121.167}"
REMOTE_NODE_IP="${REMOTE_NODE_IP:-192.168.121.168}"
RAY_PORT="${RAY_PORT:-6379}"
REMOTE_HOST="${REMOTE_HOST:-master}"
REMOTE_NODE_RANK="${REMOTE_NODE_RANK:-1}"
HEAD_ADDR="${HEAD_ADDR:-}"
LOCAL_ONLY=0
REMOTE_ONLY=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --local-node-ip)
      LOCAL_NODE_IP="${2:?--local-node-ip requires a value}"
      shift 2
      ;;
    --remote-node-ip)
      REMOTE_NODE_IP="${2:?--remote-node-ip requires a value}"
      shift 2
      ;;
    --port)
      RAY_PORT="${2:?--port requires a value}"
      shift 2
      ;;
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
    --local-only|--skip-remote)
      LOCAL_ONLY=1
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
    *)
      echo "Unknown argument: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

if [[ -z "$HEAD_ADDR" ]]; then
  HEAD_ADDR="${LOCAL_NODE_IP}:${RAY_PORT}"
fi

if [[ ! -d "$RLINF_ROOT" ]]; then
  echo "RLINF_ROOT does not exist: $RLINF_ROOT" >&2
  exit 2
fi
if [[ ! -x "$VENV/bin/python" ]]; then
  echo "VENV python not found: $VENV/bin/python" >&2
  exit 2
fi

quote() {
  printf '%q' "$1"
}

ray_processes_running() {
  ps -u "$USER" -o stat=,cmd= \
    | awk '$1 !~ /^Z/ && $0 ~ /([r]aylet|[g]cs_server|[d]ashboard|[r]untime_env_agent)/ { found=1 } END { exit !found }'
}

ray_current_cluster() {
  if [[ -s /tmp/ray/ray_current_cluster ]]; then
    tr -d '[:space:]' < /tmp/ray/ray_current_cluster || true
  fi
}

write_current_cluster() {
  mkdir -p /tmp/ray
  printf '%s\n' "$HEAD_ADDR" > /tmp/ray/ray_current_cluster
}

start_local_head() {
  cd "$RLINF_ROOT"
  source "$VENV/bin/activate"
  export PYTHONPATH="$RLINF_ROOT:${PYTHONPATH:-}"
  export RLINF_NODE_RANK=0
  if [[ -n "${RLINF_COMM_NET_DEVICES:-}" ]]; then
    export RLINF_COMM_NET_DEVICES
  fi

  local current
  current="$(ray_current_cluster)"
  if [[ "$current" == "$HEAD_ADDR" ]] && ray_processes_running; then
    echo "[dual-franka-ray] local Ray already points at $current"
    return
  fi

  if ray_processes_running; then
    echo "[dual-franka-ray] local Ray appears to be running but current head is '${current:-unknown}', expected '$HEAD_ADDR'." >&2
    echo "[dual-franka-ray] Not touching existing Ray processes. Stop/rejoin manually when safe." >&2
    exit 3
  fi

  echo "[dual-franka-ray] starting local Ray head at $HEAD_ADDR"
  ray start --head \
    --node-ip-address="$LOCAL_NODE_IP" \
    --port="$RAY_PORT" \
    --disable-usage-stats
  write_current_cluster
}

start_remote_worker() {
  local remote_script
  remote_script='
set -euo pipefail

cd "$REMOTE_RLINF_ROOT"
source "$REMOTE_VENV/bin/activate"
export PYTHONPATH="$REMOTE_RLINF_ROOT:${PYTHONPATH:-}"
export RLINF_NODE_RANK="$REMOTE_NODE_RANK"
if [[ -n "${RLINF_COMM_NET_DEVICES:-}" ]]; then
  export RLINF_COMM_NET_DEVICES
fi

ray_processes_running() {
  ps -u "$USER" -o stat=,cmd= \
    | awk '\''$1 !~ /^Z/ && $0 ~ /([r]aylet|[g]cs_server|[d]ashboard|[r]untime_env_agent)/ { found=1 } END { exit !found }'\''
}

ray_current_cluster() {
  if [[ -s /tmp/ray/ray_current_cluster ]]; then
    tr -d "[:space:]" < /tmp/ray/ray_current_cluster || true
  fi
}

write_current_cluster() {
  mkdir -p /tmp/ray
  printf "%s\n" "$HEAD_ADDR" > /tmp/ray/ray_current_cluster
}

current="$(ray_current_cluster)"
if [[ "$current" == "$HEAD_ADDR" ]] && ray_processes_running; then
  echo "[dual-franka-ray] remote Ray already points at $current"
  exit 0
fi

if ray_processes_running; then
  echo "[dual-franka-ray] remote Ray appears to be running but current head is ${current:-unknown}, expected $HEAD_ADDR." >&2
  echo "[dual-franka-ray] Not touching existing Ray processes. Stop/rejoin manually when safe." >&2
  exit 3
fi

echo "[dual-franka-ray] starting remote Ray worker at $REMOTE_NODE_IP -> $HEAD_ADDR"
ray start \
  --address="$HEAD_ADDR" \
  --node-ip-address="$REMOTE_NODE_IP" \
  --disable-usage-stats
write_current_cluster
'

  echo "[dual-franka-ray] joining remote Ray node: $REMOTE_HOST"
  ssh "$REMOTE_HOST" \
    "HEAD_ADDR=$(quote "$HEAD_ADDR") REMOTE_NODE_IP=$(quote "$REMOTE_NODE_IP") REMOTE_RLINF_ROOT=$(quote "$REMOTE_RLINF_ROOT") REMOTE_VENV=$(quote "$REMOTE_VENV") REMOTE_NODE_RANK=$(quote "$REMOTE_NODE_RANK") RLINF_COMM_NET_DEVICES=$(quote "${RLINF_COMM_NET_DEVICES:-}") bash -lc $(quote "$remote_script")"
}

if [[ "$REMOTE_ONLY" -eq 0 ]]; then
  start_local_head
else
  echo "[dual-franka-ray] remote-only requested; not checking local Ray head"
fi

if [[ "$LOCAL_ONLY" -eq 0 ]]; then
  start_remote_worker
else
  echo "[dual-franka-ray] local-only requested; not checking remote Ray worker"
fi

echo "[dual-franka-ray] Ray head address: $HEAD_ADDR"
