#!/usr/bin/env bash
# NO-RESET multi-seed: 18 atomic tasks x seeds 1-9 = 162 cells, single-episode (no reset),
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# 3-way parallel on GPU0-2. New rules: agent_task_prompt_robocasa.md (synced #1 RULE),
# max_chunks=40, settle off, LIGHT dump (won't fill /). Resume-aware; cleans each WD after.
set -u
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RUN="$REPO_ROOT/scripts/robocasa_run_explore.sh"
export PROMPT_FILE="$REPO_ROOT/resources/robocasa/agent_task_prompt_robocasa.md"
export OUT_BASE="$REPO_ROOT/explore_results_noreset_atomic"
export RLDX_MAX_CHUNKS=40 RLDX_SETTLE_PATIENCE=999
TASKS=(CloseBlenderLid CloseFridge CloseToasterOvenDoor CoffeeSetupMug NavigateKitchen
       OpenCabinet OpenDrawer OpenStandMixerHead PickPlaceCounterToCabinet
       PickPlaceCounterToStove PickPlaceDrawerToCounter PickPlaceSinkToCounter
       PickPlaceToasterToCounter SlideDishwasherRack TurnOffStove TurnOnElectricKettle
       TurnOnMicrowave TurnOnSinkFaucet)
SEEDS=(1 2 3 4 5 6 7 8 9)
GPUS=(0 1 2); PARALLEL=3; CELL_TIMEOUT=1800
LOGDIR="$OUT_BASE/_logs"; mkdir -p "$LOGDIR"
# seed-OUTER ordering for broad coverage fast
CELLS=(); for s in "${SEEDS[@]}"; do for t in "${TASKS[@]}"; do CELLS+=("$t|$s"); done; done
echo "[ms] $(date +%T) start: ${#CELLS[@]} cells, PARALLEL=$PARALLEL, no-reset, max40 settle-off light"
idx=0
for cell in "${CELLS[@]}"; do
  task=${cell%|*}; seed=${cell#*|}
  audit="$OUT_BASE/$task/${task}_s${seed}.json"
  [ -f "$audit" ] && { echo "[ms] SKIP $task s$seed"; continue; }
  while [ "$(jobs -rp | wc -l)" -ge "$PARALLEL" ]; do sleep 5; done
  gpu=${GPUS[$((idx % ${#GPUS[@]}))]}; idx=$((idx+1))
  echo "[ms] $(date +%T) START $task s$seed -> GPU$gpu"
  ( bash "$RUN" "$task" "$gpu" "$seed" "$CELL_TIMEOUT" > "$LOGDIR/${task}_s${seed}.log" 2>&1
    if [ -f "$audit" ]; then ok=$(grep -o '"success": *true' "$audit"|head -1)
      echo "[ms] $(date +%T) DONE $task s$seed -> ${ok:+SUCCESS}${ok:-fail}"
    else echo "[ms] $(date +%T) DONE $task s$seed -> NO AUDIT"; fi
    rm -rf "/tmp/explore_${task}_s${seed}" 2>/dev/null ) &   # free WD after archive
  sleep 3
done
wait
echo "[ms] $(date +%T) ALL DONE"
