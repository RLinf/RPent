#!/usr/bin/env bash
# Start an external SAM3-compatible server for the LIBERO segment tool.
#
# This optional startup helper invokes a user-provided SAM3 launcher. It is not
# called automatically by cli/main.py; the runtime segment tool only calls the
# service configured by SAM3_SERVER_URL.

set -euo pipefail

SAM3_GPU="${SAM3_GPU:-0}"
SAM3_HOST="${SAM3_HOST:-127.0.0.1}"
SAM3_PORT="${SAM3_PORT:-8114}"
SAM3_READY_ATTEMPTS="${SAM3_READY_ATTEMPTS:-60}"
SAM3_READY_INTERVAL_S="${SAM3_READY_INTERVAL_S:-5}"
URL="http://${SAM3_HOST}:${SAM3_PORT}"

is_up() {
  # Any HTTP response means the service is reachable; "/" does not need to
  # return 200. --noproxy avoids localhost proxy false positives.
  curl -s --noproxy '*' -o /dev/null -m 3 "$URL/" 2>/dev/null
}

if is_up; then
  echo "[run_sam3_server] SAM3 service already reachable at $URL"
  echo "[run_sam3_server] To use this service with RPent in the run shell:"
  echo "export SAM3_SERVER_URL=$URL"
  exit 0
fi

if [ -z "${SAM3_LAUNCHER:-}" ]; then
  echo "[run_sam3_server] ERROR: SAM3_LAUNCHER is not set." >&2
  echo "[run_sam3_server] Example:" >&2
  echo "  export SAM3_LAUNCHER=/absolute/path/to/run_sam3_server.sh" >&2
  exit 1
fi

if [ ! -f "$SAM3_LAUNCHER" ]; then
  echo "[run_sam3_server] ERROR: SAM3 launcher not found: $SAM3_LAUNCHER" >&2
  echo "[run_sam3_server] Check SAM3_LAUNCHER=$SAM3_LAUNCHER" >&2
  exit 1
fi

echo "[run_sam3_server] optional SAM3 service config"
echo "[run_sam3_server]   launcher: $SAM3_LAUNCHER"
echo "[run_sam3_server]   gpu:  $SAM3_GPU"
echo "[run_sam3_server]   url:  $URL"
echo
echo "[run_sam3_server] To use this service with RPent in the run shell:"
echo "export SAM3_SERVER_URL=$URL"
echo
echo "[run_sam3_server] launching optional SAM3 service..."

if ! SAM3_GPU="$SAM3_GPU" \
     SAM3_CUDA_DEVICE="$SAM3_GPU" \
     SAM3_HOST="$SAM3_HOST" \
     SAM3_PORT="$SAM3_PORT" \
       bash "$SAM3_LAUNCHER"; then
  echo "[run_sam3_server] ERROR: SAM3 launcher failed: $SAM3_LAUNCHER" >&2
  exit 1
fi

echo "[run_sam3_server] waiting for SAM3 service at $URL (up to $((SAM3_READY_ATTEMPTS * SAM3_READY_INTERVAL_S))s) ..."
for _ in $(seq 1 "$SAM3_READY_ATTEMPTS"); do
  if is_up; then
    echo "[run_sam3_server] SAM3 service reachable at $URL"
    echo "[run_sam3_server] Remember to run this in the RPent shell:"
    echo "export SAM3_SERVER_URL=$URL"
    exit 0
  fi
  sleep "$SAM3_READY_INTERVAL_S"
done

echo "[run_sam3_server] ERROR: SAM3 service is not reachable at $URL after $((SAM3_READY_ATTEMPTS * SAM3_READY_INTERVAL_S))s" >&2
echo "[run_sam3_server] The helper can only print the export command; it cannot set SAM3_SERVER_URL in the parent shell." >&2
exit 1
