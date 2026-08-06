#!/usr/bin/env bash
# Parallel explore sweep over the 16 composite_UNSEEN tasks, POOL of 2 concurrent
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# claude -p explore workers (reset-based, resume-aware: skips cells with an audit).
# Each worker = robocasa_run_explore.sh (driver + STANDALONE claude -p).
# Launch detached:  setsid bash scripts/robocasa_run_explore_unseen.sh >sweep_unseen.out 2>&1 &
set -u
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RUN="$REPO_ROOT/scripts/robocasa_run_explore.sh"

TASKS=(ArrangeBreadBasket ArrangeTea BreadSelection CategorizeCondiments
       CuttingToolSelection GarnishPancake GatherTableware HeatKebabSandwich
       MakeIceLemonade PanTransfer PortionHotDogs RecycleBottlesByType
       SeparateFreezerRack WaffleReheat WashFruitColander WeighIngredients)  # 16 composite_unseen
read -ra SEEDS <<< "${SEEDS_LIST:-0}"            # override: SEEDS_LIST="0 9 19"
read -ra GPUS  <<< "${GPUS_LIST:-1 2 4 5}"       # free GPUs
PARALLEL=${PARALLEL:-2}
CELL_TIMEOUT=${CELL_TIMEOUT:-7200}               # per-cell agent wall-clock budget (s) = 2h
LOGDIR="$REPO_ROOT/explore_results/_sweep_logs"
mkdir -p "$LOGDIR"

CELLS=()
for t in "${TASKS[@]}"; do for s in "${SEEDS[@]}"; do CELLS+=("$t|$s"); done; done
echo "[sweep-unseen] $(date +%T) start: ${#CELLS[@]} cells, PARALLEL=$PARALLEL, CELL_TIMEOUT=${CELL_TIMEOUT}s"

idx=0
for cell in "${CELLS[@]}"; do
  task=${cell%|*}; seed=${cell#*|}
  audit="$REPO_ROOT/explore_results/$task/${task}_s${seed}.json"
  if [ -f "$audit" ]; then
    echo "[sweep-unseen] SKIP  $task s$seed (audit exists)"; continue
  fi
  while [ "$(jobs -rp | wc -l)" -ge "$PARALLEL" ]; do sleep 5; done
  gpu=${GPUS[$((idx % ${#GPUS[@]}))]}; idx=$((idx+1))
  echo "[sweep-unseen] $(date +%T) START $task s$seed -> GPU$gpu"
  (
    bash "$RUN" "$task" "$gpu" "$seed" "$CELL_TIMEOUT" \
      > "$LOGDIR/${task}_s${seed}.log" 2>&1
    if [ -f "$audit" ]; then
      ok=$(grep -o '"success": *true' "$audit" 2>/dev/null | head -1)
      echo "[sweep-unseen] $(date +%T) DONE  $task s$seed -> ${ok:+SUCCESS}${ok:-(fail/none)}"
    else
      echo "[sweep-unseen] $(date +%T) DONE  $task s$seed -> NO AUDIT"
    fi
  ) &
  sleep 3
done
wait
echo "[sweep-unseen] $(date +%T) ALL DONE ($(ls $REPO_ROOT/explore_results/*/*.json 2>/dev/null | wc -l) total audits)"
