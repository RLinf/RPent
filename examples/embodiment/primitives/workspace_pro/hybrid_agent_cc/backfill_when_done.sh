#!/usr/bin/env bash
# Wait for the in-flight spatial grid to exit, confirm the API is reachable
# again (the 12:29-13:44 ConnectionRefused outage wiped swap t5-t9), then
# relaunch the idempotent grid to backfill the ~70 MISSING cells. Cells that
# already have an audit are skipped, so only the gaps re-run.
set -u
cd /mnt/public2/zhangyixian/RLinf_agentic
GRID=examples/embodiment/primitives/workspace_pro/hybrid_agent_cc/run_spatial_grid.sh

echo "[$(date +%T)] waiting for current grid (run_spatial_grid.sh) to exit..."
while pgrep -f "run_spatial_grid.sh" >/dev/null 2>&1; do sleep 30; done
echo "[$(date +%T)] grid exited; letting last workers/drivers settle (30s)"
sleep 30

# Don't relaunch into a dead network — probe until the API endpoint answers.
echo "[$(date +%T)] probing API reachability before backfill..."
for i in $(seq 1 30); do
  code=$(curl -sS -o /dev/null -w '%{http_code}' --max-time 10 \
         -X POST https://api.anthropic.com/v1/messages 2>/dev/null || echo 000)
  if [ "$code" != "000" ]; then
    echo "[$(date +%T)] API reachable (HTTP $code) — launching backfill"
    break
  fi
  echo "[$(date +%T)] API still unreachable (code=$code), retry $i/30 in 60s"
  sleep 60
done

nohup bash "$GRID" > /tmp/spatial_grid_backfill.log 2>&1 &
echo "[$(date +%T)] backfill grid relaunched, pid $!  (log: /tmp/spatial_grid_backfill.log)"
