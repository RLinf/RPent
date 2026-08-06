#!/usr/bin/env bash
# Hybrid explore sweep over the 18 ATOMIC tasks (same framework as the unseen sweep):
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# per cell = robocasa_run_explore.sh (driver + STANDALONE claude -p, reset-based multi-attempt).
# Resume-aware (skips cells with an audit). Launch detached:
#   setsid bash scripts/robocasa_run_explore_atomic.sh >sweep_atomic.out 2>&1 &
set -u
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RUN="$REPO_ROOT/scripts/robocasa_run_explore.sh"

TASKS=(CloseBlenderLid CloseFridge CloseToasterOvenDoor CoffeeSetupMug NavigateKitchen
       OpenCabinet OpenDrawer OpenStandMixerHead PickPlaceCounterToCabinet
       PickPlaceCounterToStove PickPlaceDrawerToCounter PickPlaceSinkToCounter
       PickPlaceToasterToCounter SlideDishwasherRack TurnOffStove TurnOnElectricKettle
       TurnOnMicrowave TurnOnSinkFaucet)                                  # 18 atomic
# Optional subset override:  TASKS_LIST="OpenCabinet TurnOnMicrowave PickPlaceCounterToCabinet"
[ -n "${TASKS_LIST:-}" ] && read -ra TASKS <<< "$TASKS_LIST"
read -ra SEEDS <<< "${SEEDS_LIST:-0}"
read -ra GPUS  <<< "${GPUS_LIST:-0 1 2 3 4 5 6 7}"
PARALLEL=${PARALLEL:-4}
CELL_TIMEOUT=${CELL_TIMEOUT:-3600}                  # 60 min/cell (articulation tasks need more)
LOGDIR="$REPO_ROOT/explore_results/_sweep_logs"
mkdir -p "$LOGDIR"

CELLS=()
# seed-OUTER ordering: all tasks' seed N before seed N+1 -> broad coverage fast,
# avoids front-loading every seed of the single hardest task.
for s in "${SEEDS[@]}"; do for t in "${TASKS[@]}"; do CELLS+=("$t|$s"); done; done
echo "[sweep-atomic] $(date +%T) start: ${#CELLS[@]} cells, PARALLEL=$PARALLEL, CELL_TIMEOUT=${CELL_TIMEOUT}s"

idx=0
for cell in "${CELLS[@]}"; do
  task=${cell%|*}; seed=${cell#*|}
  audit="$REPO_ROOT/explore_results/$task/${task}_s${seed}.json"
  if [ -f "$audit" ]; then echo "[sweep-atomic] SKIP $task s$seed (audit exists)"; continue; fi
  while [ "$(jobs -rp | wc -l)" -ge "$PARALLEL" ]; do sleep 5; done
  gpu=${GPUS[$((idx % ${#GPUS[@]}))]}; idx=$((idx+1))
  echo "[sweep-atomic] $(date +%T) START $task s$seed -> GPU$gpu"
  (
    bash "$RUN" "$task" "$gpu" "$seed" "$CELL_TIMEOUT" > "$LOGDIR/${task}_s${seed}.log" 2>&1
    if [ -f "$audit" ]; then
      ok=$(grep -o '"success": *true' "$audit" 2>/dev/null | head -1)
      echo "[sweep-atomic] $(date +%T) DONE $task s$seed -> ${ok:+SUCCESS}${ok:-(fail/none)}"
    else echo "[sweep-atomic] $(date +%T) DONE $task s$seed -> NO AUDIT"; fi
  ) &
  sleep 3
done
wait
echo "[sweep-atomic] $(date +%T) ALL DONE"
