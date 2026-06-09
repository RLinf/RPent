#!/bin/bash
# Sequential sweep of one (suite, task) over multiple seeds via `claude -p`.
# Each cell is a fresh `claude -p` invocation; no context bleeds between cells.
#
# Usage:
#   bash run_all_seeds.sh                     # spatial_lan t0 seeds 0..9
#   SUITE=libero_object_task TASK=2 SEEDS="0 1 2" bash run_all_seeds.sh

set -e

SUITE=${SUITE:-libero_spatial_lan}
TASK=${TASK:-0}
SEEDS=${SEEDS:-"0 1 2 3 4 5 6 7 8 9"}
CUDA_DEVICE=${CUDA_DEVICE:-0}
MASTER_LOG=${MASTER_LOG:-/tmp/claude_p_master_${SUITE}_t${TASK}.log}

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
RUN_ONE="$SCRIPT_DIR/run_one_cell.sh"

> "$MASTER_LOG"

log() {
    echo "[$(date +%T)] $*" | tee -a "$MASTER_LOG"
}

log "==== Claude Code sequential sweep ===="
log "suite=$SUITE  task=$TASK  seeds=$SEEDS  gpu=$CUDA_DEVICE"
log "master log: $MASTER_LOG"

T_TOTAL=$(date +%s)
for seed in $SEEDS; do
    log "[start] s$seed"
    T_CELL=$(date +%s)

    SUITE=$SUITE TASK=$TASK SEED=$seed CUDA_DEVICE=$CUDA_DEVICE \
        bash "$RUN_ONE" "$SUITE" "$TASK" "$seed" 2>&1 | tee -a "$MASTER_LOG" || true

    log "[end] s$seed elapsed=$(( $(date +%s) - T_CELL ))s"
done

log "==== ALL DONE in $(( $(date +%s) - T_TOTAL ))s ===="

# Print summary
log ""
log "=== summary ==="
OUTPUT_DIR=${OUTPUT_DIR:-/mnt/public/jxqiu/physicalagent/physicalagent/primitives/workspace_pro/results_claude_p_runs}
for seed in $SEEDS; do
    tag=${SUITE/libero_/}_t${TASK}_s${seed}
    audit=$OUTPUT_DIR/${tag}.json
    if [ -f "$audit" ]; then
        term=$(/opt/venv/openpi/bin/python -c "import json;d=json.load(open('$audit'));print(d.get('libero_terminated'))" 2>/dev/null)
        log "  $tag  libero_term=$term"
    else
        log "  $tag  MISSING_AUDIT"
    fi
done
