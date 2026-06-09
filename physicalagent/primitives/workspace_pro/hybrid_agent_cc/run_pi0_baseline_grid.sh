#!/bin/bash
# Pi0 fullshot baseline GRID — the apples-to-apples comparison for the
# hybrid claude -p sweeps. Pi0.5 drives end-to-end (no LLM, no scripting).
#
# Default: libero_object PRO, 3 perturbation regimes x 10 tasks x 10 seeds
#          = 300 cells, 8-GPU parallel (1 cell/GPU).
#
# This is a pure Python+sim job — NO claude -p, so no subscription quota
# and no API dependency. Each cell is one `pi0_baseline.py` invocation
# pinned to one GPU. Idempotent: skips a cell whose output JSON exists.
#
# Failure mode note: pi0_baseline runs a SINGLE short end-to-end episode
# (<=60 chunks), so the EGL accumulation that crashes the long hybrid
# driver is far less likely here. Still capped per-cell by CELL_TIMEOUT_S.
#
# Launch detached:
#   nohup bash physicalagent/primitives/workspace_pro/hybrid_agent_cc/run_pi0_baseline_grid.sh \
#       > /tmp/pi0_baseline_grid_outer.log 2>&1 &
#
# Subset / other suite examples:
#   SUITES="libero_object_task"               bash run_pi0_baseline_grid.sh   # 1 regime
#   SUITES="libero_spatial_task libero_spatial_lan libero_spatial_swap" \
#       OUTPUT_DIR=.../multi_seed_exp/spatial_pi0_baseline  bash run_pi0_baseline_grid.sh

set -e
cd /mnt/public/jxqiu/physicalagent
RES=physicalagent/primitives/workspace_pro

SUITES=${SUITES:-"libero_object_task libero_object_lan libero_object_swap"}
TASKS=${TASKS:-"0 1 2 3 4 5 6 7 8 9"}
SEEDS=${SEEDS:-"0 1 2 3 4 5 6 7 8 9"}
GPUS=${GPUS:-"0 1 2 3 4 5 6 7"}
MAX_CHUNKS=${MAX_CHUNKS:-60}
MAX_EPISODE_STEPS=${MAX_EPISODE_STEPS:-600}
STAGGER_S=${STAGGER_S:-10}
CELL_TIMEOUT_S=${CELL_TIMEOUT_S:-900}
export LIBERO_TYPE=${LIBERO_TYPE:-pro}

PYTHON_BIN=${PYTHON_BIN:-/opt/venv/openpi/bin/python}
OUTPUT_DIR=${OUTPUT_DIR:-$RES/multi_seed_exp/object_pi0_baseline}
LOGDIR=${LOGDIR:-/tmp/pi0_baseline_grid_logs}
MASTER_LOG=${MASTER_LOG:-/tmp/pi0_baseline_grid_master.log}
mkdir -p "$OUTPUT_DIR" "$OUTPUT_DIR/imgs" "$LOGDIR"

> "$MASTER_LOG"
log() { echo "[$(date +%T)] $*" | tee -a "$MASTER_LOG"; }

GPU_ARR=($GPUS)
N_GPUS=${#GPU_ARR[@]}

log "==== Pi0 fullshot baseline grid ===="
log "suites=[$SUITES]"
log "tasks=[$TASKS]  seeds=[$SEEDS]  gpus=[$GPUS]  libero_type=$LIBERO_TYPE"
log "output_dir=$OUTPUT_DIR  concurrency=$N_GPUS  stagger=${STAGGER_S}s  cell_timeout=${CELL_TIMEOUT_S}s"

# Concurrent slot pool (one slot per GPU). Round-robin reuse of freed slots.
declare -A SLOT_PID
declare -A SLOT_TAG

free_slot() {
    while true; do
        for s in "${!SLOT_PID[@]}"; do
            if ! kill -0 "${SLOT_PID[$s]}" 2>/dev/null; then
                tag=${SLOT_TAG[$s]}
                wait "${SLOT_PID[$s]}" 2>/dev/null || true
                echo "[$(date +%T)]   [done] slot $s (gpu ${GPU_ARR[$s]}) tag=$tag" >> "$MASTER_LOG"
                unset SLOT_PID[$s]
                unset SLOT_TAG[$s]
                echo "$s"
                return 0
            fi
        done
        sleep 3
    done
}

# Run one cell (skip if output exists). Pinned to GPU $1.
run_cell() {
    local gpu=$1 suite=$2 task=$3 seed=$4
    local tag=${suite/libero_/}_t${task}_s${seed}
    local out=$OUTPUT_DIR/baseline_${suite}_t${task}_s${seed}.json
    local img=$OUTPUT_DIR/imgs/${suite}_t${task}_s${seed}
    local clog=$LOGDIR/${tag}.log
    if [ -f "$out" ]; then
        echo "[$(date +%T)] [$tag] SKIP: $out exists" >> "$MASTER_LOG"
        return 0
    fi
    LIBERO_TYPE=$LIBERO_TYPE CUDA_VISIBLE_DEVICES=$gpu \
        timeout --kill-after=15 "$CELL_TIMEOUT_S" \
        "$PYTHON_BIN" physicalagent/primitives/pi0_baseline.py \
            --suite "$suite" --task "$task" --seed "$seed" \
            --max_chunks "$MAX_CHUNKS" --max_episode_steps "$MAX_EPISODE_STEPS" \
            --out "$out" --save_image_dir "$img" \
            > "$clog" 2>&1
}

T_TOTAL=$(date +%s)
launched=0
for suite in $SUITES; do
    for task in $TASKS; do
        for seed in $SEEDS; do
            tag=${suite/libero_/}_t${task}_s${seed}
            # find a free slot
            slot=""
            if [ ${#SLOT_PID[@]} -lt $N_GPUS ]; then
                for s in $(seq 0 $((N_GPUS-1))); do
                    if [ -z "${SLOT_PID[$s]:-}" ]; then slot=$s; break; fi
                done
            else
                slot=$(free_slot)
            fi
            gpu=${GPU_ARR[$slot]}
            log "[launch] slot=$slot gpu=$gpu  tag=$tag"
            if [ $launched -gt 0 ]; then sleep "$STAGGER_S"; fi
            run_cell "$gpu" "$suite" "$task" "$seed" &
            SLOT_PID[$slot]=$!
            SLOT_TAG[$slot]=$tag
            launched=$((launched+1))
        done
    done
done

log ""
log "[grid] all $launched cells launched; waiting for stragglers..."
for s in "${!SLOT_PID[@]}"; do
    wait "${SLOT_PID[$s]}" 2>/dev/null || true
    log "  [done] slot $s tag=${SLOT_TAG[$s]}"
done

ELAPSED=$(( $(date +%s) - T_TOTAL ))
log ""
log "==== ALL DONE in ${ELAPSED}s ===="

# Summary
log "=== summary ==="
n_total=0; n_ok=0; n_missing=0
for suite in $SUITES; do
    for task in $TASKS; do
        for seed in $SEEDS; do
            out=$OUTPUT_DIR/baseline_${suite}_t${task}_s${seed}.json
            n_total=$((n_total+1))
            if [ -f "$out" ]; then
                term=$("$PYTHON_BIN" -c "import json;print(json.load(open('$out')).get('libero_terminated'))" 2>/dev/null)
                [ "$term" = "True" ] && n_ok=$((n_ok+1))
            else
                n_missing=$((n_missing+1))
            fi
        done
    done
done
log ""
log "FINAL: $n_ok/$n_total libero_terminated=True  ($n_missing missing)"
