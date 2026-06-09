#!/usr/bin/env bash
# Ride through the recurring ConnectionRefused outages: repeatedly re-run the
# idempotent spatial grid (skips cells that already have an audit) until all
# 300 exist or MAXPASS is reached. Each pass waits for any running grid to
# exit, counts gaps, and only relaunches once the API is reachable again.
set -u
cd /mnt/public/jxqiu/physicalagent
GRID=physicalagent/primitives/workspace_pro/hybrid_agent_cc/run_spatial_grid.sh
OUT=physicalagent/primitives/workspace_pro/multi_seed_exp/spatial
MAXPASS=12

count_missing() {
  local m=0
  for reg in task lan swap; do
    for t in 0 1 2 3 4 5 6 7 8 9; do for s in 0 1 2 3 4 5 6 7 8 9; do
      [ -f "$OUT/spatial_${reg}_t${t}_s${s}.json" ] || m=$((m+1))
    done; done
  done
  echo "$m"
}

for pass in $(seq 1 $MAXPASS); do
  echo "[$(date +%T)] pass $pass: waiting for any running grid to exit..."
  while pgrep -f "run_spatial_grid.sh" >/dev/null 2>&1; do sleep 30; done
  sleep 20

  miss=$(count_missing)
  echo "[$(date +%T)] pass $pass: $miss cells missing of 300"
  if [ "$miss" -eq 0 ]; then
    echo "[$(date +%T)] COMPLETE — all 300 cells present. Stopping."
    break
  fi

  # Gate on API reachability so we don't burn a pass into a dead network.
  echo "[$(date +%T)] probing API before relaunch..."
  for i in $(seq 1 60); do
    code=$(curl -sS -o /dev/null -w '%{http_code}' --max-time 10 \
           -X POST https://api.anthropic.com/v1/messages 2>/dev/null || echo 000)
    [ "$code" != "000" ] && { echo "[$(date +%T)] API up (HTTP $code)"; break; }
    echo "[$(date +%T)] API down (code=$code), retry $i/60 in 60s"; sleep 60
  done

  echo "[$(date +%T)] relaunching grid (pass $pass) -> /tmp/spatial_backfill_pass${pass}.log"
  nohup bash "$GRID" > /tmp/spatial_backfill_pass${pass}.log 2>&1 &
  sleep 15   # let it register so the top-of-loop pgrep catches it
done

echo "[$(date +%T)] retry loop done. final missing: $(count_missing)/300"
