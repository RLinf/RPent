#!/bin/bash
# Backfill ONLY the missing cells of a multi-seed sweep — no skip-walk.
#
# run_object_grid.sh / run_spatial_grid.sh re-launch is idempotent but
# WASTEFUL after a big partial sweep: run_parallel_seeds.sh sleeps
# STAGGER_S before every launch INCLUDING cells it then instantly skips,
# so walking past ~270 done cells burns ~4h of pure sleep. This script
# instead enumerates the (variant,task,seed) cells whose audit JSON is
# MISSING, builds a flat list, and dispatches only those across $GPUS
# with a slot pool — stagger applies only to real launches.
#
# Each cell still goes through run_one_cell.sh (same driver + claude -p +
# LIBERO_TYPE + budget wiring), so behaviour is identical to the grid.
#
# Usage (defaults shown):
#   SUITES="libero_object_task libero_object_lan libero_object_swap" \
#   TASKS="0 1 2 3 4 5 6 7 8 9" SEEDS="0 1 2 3 4 5 6 7 8 9" \
#   GPUS="0 1 2 3" MODEL=claude-opus-4-7 STAGGER_S=60 MAX_BUDGET_USD=10 \
#   OUTPUT_DIR=.../multi_seed_exp/object \
#     bash run_backfill.sh
#
# Launch detached:
#   nohup bash .../hybrid_agent_cc/run_backfill.sh > /tmp/object_backfill_outer.log 2>&1 &

set -e
cd /mnt/public/jxqiu/physicalagent
HCC=physicalagent/primitives/workspace_pro/hybrid_agent_cc
RES=physicalagent/primitives/workspace_pro
RUN_ONE="$HCC/run_one_cell.sh"

SUITES=${SUITES:-"libero_object_task libero_object_lan libero_object_swap"}
TASKS=${TASKS:-"0 1 2 3 4 5 6 7 8 9"}
SEEDS=${SEEDS:-"0 1 2 3 4 5 6 7 8 9"}
GPUS=${GPUS:-"0 1 2 3"}
MODEL=${MODEL:-claude-opus-4-7}
STAGGER_S=${STAGGER_S:-60}
MAX_BUDGET_USD=${MAX_BUDGET_USD:-10}
MAX_TURNS=${MAX_TURNS:-60}
export LIBERO_TYPE=${LIBERO_TYPE:-pro}
OUTPUT_DIR=${OUTPUT_DIR:-$RES/multi_seed_exp/object}
MASTER_LOG=${MASTER_LOG:-/tmp/object_backfill.log}
mkdir -p "$OUTPUT_DIR"
> "$MASTER_LOG"

log() { echo "[$(date +%T)] $*" | tee -a "$MASTER_LOG"; }

# --- 1. enumerate MISSING cells -> flat list of "suite task seed" ---
MISSING=()
for suite in $SUITES; do
    short=${suite/libero_/}
    for t in $TASKS; do for s in $SEEDS; do
        [ -f "$OUTPUT_DIR/${short}_t${t}_s${s}.json" ] || MISSING+=("$suite $t $s")
    done; done
done
N=${#MISSING[@]}
log "==== backfill: $N missing cells ===="
log "suites=[$SUITES]  gpus=[$GPUS]  model=$MODEL  libero_type=$LIBERO_TYPE  output=$OUTPUT_DIR"
if [ "$N" -eq 0 ]; then log "nothing to backfill — all audits present."; exit 0; fi
for m in "${MISSING[@]}"; do log "  MISSING: $m"; done

# --- 2. dispatch across GPU slots (slot pool, stagger only on launch) ---
GPU_ARR=($GPUS); N_GPUS=${#GPU_ARR[@]}
declare -A SLOT_PID SLOT_TAG

free_slot() {
    while true; do
        for sidx in "${!SLOT_PID[@]}"; do
            if ! kill -0 "${SLOT_PID[$sidx]}" 2>/dev/null; then
                wait "${SLOT_PID[$sidx]}" 2>/dev/null || true
                echo "[$(date +%T)]   [done] slot $sidx tag=${SLOT_TAG[$sidx]}" >> "$MASTER_LOG"
                unset SLOT_PID[$sidx]; unset SLOT_TAG[$sidx]
                echo "$sidx"; return 0
            fi
        done
        sleep 5
    done
}

T0=$(date +%s); launched=0
for m in "${MISSING[@]}"; do
    read -r suite task seed <<< "$m"
    slot=""
    if [ ${#SLOT_PID[@]} -lt $N_GPUS ]; then
        for sidx in $(seq 0 $((N_GPUS-1))); do
            [ -z "${SLOT_PID[$sidx]:-}" ] && { slot=$sidx; break; }
        done
    else
        slot=$(free_slot)
    fi
    gpu=${GPU_ARR[$slot]}
    tag=${suite/libero_/}_t${task}_s${seed}
    log "[launch] slot=$slot gpu=$gpu tag=$tag"
    [ $launched -gt 0 ] && sleep "$STAGGER_S"
    SUITE="$suite" TASK="$task" SEED="$seed" CUDA_DEVICE="$gpu" MODEL="$MODEL" \
    MAX_TURNS="$MAX_TURNS" MAX_BUDGET_USD="$MAX_BUDGET_USD" \
    OUTPUT_DIR="$OUTPUT_DIR" \
        bash "$RUN_ONE" "$suite" "$task" "$seed" >> "$MASTER_LOG" 2>&1 &
    SLOT_PID[$slot]=$!; SLOT_TAG[$slot]=$tag
    launched=$((launched+1))
done

for sidx in "${!SLOT_PID[@]}"; do
    wait "${SLOT_PID[$sidx]}" 2>/dev/null || true
    log "  [done] slot $sidx tag=${SLOT_TAG[$sidx]}"
done

log "==== backfill DONE in $(( $(date +%s) - T0 ))s ===="
# summary of what's still missing after this pass
still=0
for suite in $SUITES; do short=${suite/libero_/}
    for t in $TASKS; do for s in $SEEDS; do
        [ -f "$OUTPUT_DIR/${short}_t${t}_s${s}.json" ] || { still=$((still+1)); log "  STILL MISSING: ${short}_t${t}_s${s}"; }
    done; done
done
log "FINAL: $((N-still))/$N backfilled; $still still missing"
