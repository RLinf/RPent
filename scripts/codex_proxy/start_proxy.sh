#!/usr/bin/env bash
# Launch a litellm proxy in the background, exposing infini-ai models
# (kimi-k2.6, deepseek-v4-flash, deepseek-v4-pro) over the OpenAI
# Responses API for Codex.
#
# Usage:
#   ./start_proxy.sh                # background, default port 4100
#   PORT=4200 ./start_proxy.sh      # custom port
#   FOREGROUND=1 ./start_proxy.sh   # run in foreground (for debugging)
#
# After it starts, point Codex at the proxy via rpent/cli/main.py:
#   --base_url http://127.0.0.1:${PORT} --api_key sk-codex-proxy
# (or any non-empty key; the proxy's master_key is the only thing it
#  validates — its own forwarded calls use INFINI_API_KEY).

set -euo pipefail

HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-4100}"
INFINI_API_KEY="${INFINI_API_KEY:-sk-xxx}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
CONFIG_FILE="${SCRIPT_DIR}/litellm_config.yaml"
LOG_DIR="${LOG_DIR:-/tmp/codex_proxy}"
mkdir -p "${LOG_DIR}"
LOG_FILE="${LOG_DIR}/codex_proxy.log"
PID_FILE="${LOG_DIR}/codex_proxy.pid"

if [[ ! -f "${CONFIG_FILE}" ]]; then
  echo "missing config: ${CONFIG_FILE}" >&2
  exit 2
fi

if [[ -f "${PID_FILE}" ]]; then
  existing="$(cat "${PID_FILE}" 2>/dev/null || true)"
  if [[ -n "${existing}" ]] && kill -0 "${existing}" 2>/dev/null; then
    echo "litellm proxy already running (pid=${existing}); log: ${LOG_FILE}"
    exit 0
  fi
  rm -f "${PID_FILE}"
fi

export INFINI_API_KEY
# litellm imports the callback by module name; the module lives in this repo,
# so make the repo root importable.
export PYTHONPATH="${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"

cmd=(
  litellm
  --config "${CONFIG_FILE}"
  --host "${HOST}"
  --port "${PORT}"
  --num_workers 1
  --telemetry False
)

cd "${REPO_ROOT}"

if [[ "${FOREGROUND:-0}" == "1" ]]; then
  exec "${cmd[@]}"
fi

nohup "${cmd[@]}" > "${LOG_FILE}" 2>&1 &
proxy_pid=$!
echo "${proxy_pid}" > "${PID_FILE}"

# litellm exposes /health/liveliness for non-credentialed liveness checks.
url="http://${HOST}:${PORT}/health/liveliness"
deadline=$(( $(date +%s) + 90 ))
while (( $(date +%s) < deadline )); do
  if curl -fsS "${url}" >/dev/null 2>&1; then
    echo "litellm proxy ready: pid=${proxy_pid} url=http://${HOST}:${PORT} log=${LOG_FILE}"
    exit 0
  fi
  if ! kill -0 "${proxy_pid}" 2>/dev/null; then
    echo "litellm proxy died during startup; tail of log:" >&2
    tail -n 80 "${LOG_FILE}" >&2 || true
    rm -f "${PID_FILE}"
    exit 1
  fi
  sleep 0.5
done

echo "litellm proxy did not become ready in 90s; tail of log:" >&2
tail -n 80 "${LOG_FILE}" >&2 || true
exit 1
