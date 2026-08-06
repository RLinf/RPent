#!/usr/bin/env bash
# Parallel explore sweep: 16 composite tasks x 5 seeds = 80 cells, POOL of 3
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# concurrent claude -p explore workers. Resume-aware (skips cells with an audit).
# Each worker = robocasa_run_explore.sh (driver + STANDALONE claude -p, reset-based).
# Launch detached:  setsid bash scripts/robocasa_run_explore_sweep.sh >sweep.out 2>&1 &
set -u
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RUN="$REPO_ROOT/scripts/robocasa_run_explore.sh"

# 16 composite_seen — ordered by fullshot VLA SR (high -> low) so the most-likely
# tasks run first. Already-solved cells are skipped via existing audits.
TASKS=(ScrubCuttingBoard StackBowlsCabinet WashLettuce RinseSinkBasin PreSoakPan
       StirVegetables LoadDishwasher SteamInMicrowave SetUpCuttingStation
       GetToastedBread DeliverStraw KettleBoiling PrepareCoffee StoreLeftoversInBowl
       SearingMeat PackIdenticalLunches)               # 16 composite_seen, SR-desc
read -ra SEEDS <<< "${SEEDS_LIST:-0 9 19 29 39}"   # override: SEEDS_LIST="0"
read -ra GPUS <<< "${GPUS_LIST:-0 1 2 3}"   # override: GPUS_LIST="0 1 2"
PARALLEL=${PARALLEL:-3}
CELL_TIMEOUT=${CELL_TIMEOUT:-1200}                     # per-cell agent budget (s)
LOGDIR="$REPO_ROOT/explore_results/_sweep_logs"
mkdir -p "$LOGDIR"

# build the 80-cell list
CELLS=()
for t in "${TASKS[@]}"; do for s in "${SEEDS[@]}"; do CELLS+=("$t|$s"); done; done
echo "[sweep] $(date +%T) start: ${#CELLS[@]} cells, PARALLEL=$PARALLEL, CELL_TIMEOUT=${CELL_TIMEOUT}s"

idx=0
for cell in "${CELLS[@]}"; do
  task=${cell%|*}; seed=${cell#*|}
  audit="$REPO_ROOT/explore_results/$task/${task}_s${seed}.json"
  if [ -f "$audit" ]; then
    echo "[sweep] SKIP  $task s$seed (audit exists)"; continue
  fi
  # throttle to PARALLEL concurrent workers
  while [ "$(jobs -rp | wc -l)" -ge "$PARALLEL" ]; do sleep 5; done
  gpu=${GPUS[$((idx % ${#GPUS[@]}))]}; idx=$((idx+1))
  echo "[sweep] $(date +%T) START $task s$seed -> GPU$gpu"
  (
    bash "$RUN" "$task" "$gpu" "$seed" "$CELL_TIMEOUT" \
      > "$LOGDIR/${task}_s${seed}.log" 2>&1
    if [ -f "$audit" ]; then
      ok=$(grep -o '"success": *true' "$audit" 2>/dev/null | head -1)
      echo "[sweep] $(date +%T) DONE  $task s$seed -> audit ${ok:+SUCCESS}${ok:-(fail/none)}"
    else
      echo "[sweep] $(date +%T) DONE  $task s$seed -> NO AUDIT"
    fi
  ) &
  sleep 3
done
wait
echo "[sweep] $(date +%T) ALL DONE ($(ls $REPO_ROOT/explore_results/*/*.json 2>/dev/null | wc -l) audits)"
