#!/usr/bin/env bash
# NO-RESET multi-seed on the 7 composite_seen tasks that HAVE a recipe (seeds 1-9 = 63 cells).
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# Uses agent_task_prompt_robocasa.md (recipe-aware: reads exploration_seen_recipe/ first).
# 3-way parallel, max_chunks=40, settle off, LIGHT dump. Resume-aware; cleans each WD.
set -u
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RUN="$REPO_ROOT/scripts/robocasa_run_explore.sh"
export PROMPT_FILE="$REPO_ROOT/resources/robocasa/agent_task_prompt_robocasa.md"
export OUT_BASE="$REPO_ROOT/explore_results_noreset_seen"
export RLDX_MAX_CHUNKS=40 RLDX_SETTLE_PATIENCE=999
TASKS=(GetToastedBread KettleBoiling LoadDishwasher PreSoakPan StackBowlsCabinet StirVegetables StoreLeftoversInBowl)
SEEDS=(1 2 3 4 5 6 7 8 9)
GPUS=(0 1 2); PARALLEL=3; CELL_TIMEOUT=3600   # composite = long-horizon, give 60 min
LOGDIR="$OUT_BASE/_logs"; mkdir -p "$LOGDIR"
CELLS=(); for s in "${SEEDS[@]}"; do for t in "${TASKS[@]}"; do CELLS+=("$t|$s"); done; done
echo "[seen] $(date +%T) start: ${#CELLS[@]} cells, PARALLEL=$PARALLEL, no-reset+recipe, to=${CELL_TIMEOUT}s"
idx=0
for cell in "${CELLS[@]}"; do
  task=${cell%|*}; seed=${cell#*|}
  audit="$OUT_BASE/$task/${task}_s${seed}.json"
  [ -f "$audit" ] && { echo "[seen] SKIP $task s$seed"; continue; }
  while [ "$(jobs -rp | wc -l)" -ge "$PARALLEL" ]; do sleep 5; done
  gpu=${GPUS[$((idx % ${#GPUS[@]}))]}; idx=$((idx+1))
  echo "[seen] $(date +%T) START $task s$seed -> GPU$gpu"
  ( bash "$RUN" "$task" "$gpu" "$seed" "$CELL_TIMEOUT" > "$LOGDIR/${task}_s${seed}.log" 2>&1
    if [ -f "$audit" ]; then ok=$(grep -o '"success": *true' "$audit"|head -1)
      echo "[seen] $(date +%T) DONE $task s$seed -> ${ok:+SUCCESS}${ok:-fail}"
    else echo "[seen] $(date +%T) DONE $task s$seed -> NO AUDIT"; fi
    rm -rf "/tmp/explore_${task}_s${seed}" 2>/dev/null ) &
  sleep 3
done
wait
echo "[seen] $(date +%T) ALL DONE"
