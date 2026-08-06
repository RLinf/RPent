#!/usr/bin/env bash
# Autonomous chain: run the SR-descending UNSOLVED tasks one at a time on GPU1, each
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# until it SUCCEEDS, then auto-advance to the next. Handles the recurring NETWORK drop
# (agent stalls -> driver "RUN" but agent.log frozen) by detecting the stall and
# relaunching, ALWAYS keeping experience.md (the persistent playbook) so progress
# accumulates across restarts. Adopts an already-running task instead of duplicating it.
set -u
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RUN="$REPO_ROOT/scripts/robocasa_run_explore.sh"
SIM="$REPO/rldx/eval/sim/robocasa365/robocasa365_uv/.venv/bin/python"
GPU=${GPU:-1}; TIMEOUT=${TIMEOUT:-21600}; STALL=${STALL:-600}
# SR-descending unsolved (DeliverStraw intentionally excluded per user)
TASKS="${TASKS_LIST:-SteamInMicrowave SetUpCuttingStation PrepareCoffee SearingMeat PackIdenticalLunches}"

solved(){ f="$REPO_ROOT/explore_results/$1/$1_s0.json"; [ -f "$f" ] && "$SIM" -c "import json,sys;sys.exit(0 if json.load(open('$f')).get('success') else 1)" 2>/dev/null; }
driver_up(){ pgrep -f "driver.py --env $1" >/dev/null 2>&1; }
agent_fresh(){ wd="/tmp/explore_$1_s0"; l=$(stat -c %Y "$wd/agent.log" 2>/dev/null||echo 0); [ $(( $(date +%s) - l )) -lt $STALL ]; }
alive(){ driver_up "$1" && agent_fresh "$1"; }   # truly progressing (not network-stalled)

capture_progress(){  # append best task_progress sub-goals to experience.md before a relaunch
  "$SIM" - "$1" <<'PY' 2>/dev/null || true
import json,glob,sys,os
t=sys.argv[1]; wd=f"/tmp/explore_{t}_s0"
fs=sorted(glob.glob(f"{wd}/state_*.json"))
best={}
for f in fs:
    for k,v in (json.load(open(f)).get("task_progress") or {}).items():
        if isinstance(v,bool): best[k]=best.get(k,False) or v
        elif isinstance(v,(int,float)): best[k]=max(best.get(k,0),v)
if best:
    exp=f"$REPO_ROOT/explore_results/{t}/experience.md"
    open(exp,"a").write(f"\n## auto-chain progress snapshot (pre-restart): {best}\n")
PY
}

kill_task(){
  for p in $(pgrep -f "robocasa_run_explore.sh $1|driver.py --env $1" 2>/dev/null); do kill -9 "$p" 2>/dev/null; done
  for p in $(ls /proc 2>/dev/null|grep -E '^[0-9]+$'); do [ "$(readlink /proc/$p/cwd 2>/dev/null)" = "/tmp/explore_$1_s0" ] && { tr '\0' ' ' </proc/$p/cmdline 2>/dev/null|grep -q claude && kill -9 "$p" 2>/dev/null; }; done
  sleep 4
}
launch(){  # fresh run, KEEP experience.md
  capture_progress "$1"; kill_task "$1"
  rm -f "$REPO_ROOT/explore_results/$1/${1}_s0.json"
  rm -rf "$REPO_ROOT/explore_results/$1/attempts" "/tmp/explore_$1_s0"
  setsid bash "$RUN" "$1" "$GPU" 0 "$TIMEOUT" </dev/null >"/tmp/chain_$1.log" 2>&1 &
  echo "[chain] $(date '+%T') launched $1"; sleep 90
}

echo "[chain] START $(date '+%T') tasks: $TASKS"
for t in $TASKS; do
  if solved "$t"; then echo "[chain] $t already solved, skip"; continue; fi
  echo "[chain] === $t === $(date '+%T')"
  while ! solved "$t"; do
    if alive "$t"; then sleep 60; continue; fi
    echo "[chain] $t not progressing (dead/stalled) -> (re)launch"
    launch "$t"
  done
  echo "[chain] *** $t SOLVED $(date '+%T') ***"
done
echo "[chain] ALL DONE $(date '+%T')"
