#!/bin/bash
# Parallel sweep — one (suite, task, seed) cell per GPU concurrently
# via `claude -p`. Each cell:
#   - lives in its own /tmp/hybrid_repl_<tag>/ (no collision)
#   - has its own claude -p log + recipe + audit
#   - is pinned to one GPU via CUDA_DEVICE
#
# Usage:
#   GPUS="0 1 2 3 4 5 6 7" SEEDS="0 1 2 3 4 5 6 7" \
#     bash run_parallel_seeds.sh
#
#   # different cell, 4 seeds on 4 GPUs:
#   SUITE=libero_object_task TASK=2 \
#   GPUS="0 1 2 3" SEEDS="0 1 2 3" \
#     bash run_parallel_seeds.sh
#
# Concurrency = min(len(SEEDS), len(GPUS)). If SEEDS exceeds GPUS, the
# extras WAIT until a GPU frees and reuse it (round-robin). If you
# want true wave-by-wave, just split the seed list and run two batches.
#
# Each launch is staggered by STAGGER_S (default 30) to spread Pi0
# load disk IO and avoid hammering the API simultaneously.

set -e

SUITE=${SUITE:-libero_spatial_lan}
TASK=${TASK:-0}
SEEDS=${SEEDS:-"0 1 2 3"}
GPUS=${GPUS:-"0 1 2 3"}
MODEL=${MODEL:-sonnet}
MAX_TURNS=${MAX_TURNS:-60}
MAX_BUDGET_USD=${MAX_BUDGET_USD:-10}
MAX_EPISODE_STEPS=${MAX_EPISODE_STEPS:-600}
STAGGER_S=${STAGGER_S:-30}
MASTER_LOG=${MASTER_LOG:-/tmp/claude_p_parallel_${SUITE}_t${TASK}.log}

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
RUN_ONE="$SCRIPT_DIR/run_one_cell.sh"

> "$MASTER_LOG"

log() {
    echo "[$(date +%T)] $*" | tee -a "$MASTER_LOG"
}

GPU_ARR=($GPUS)
SEED_ARR=($SEEDS)
N_GPUS=${#GPU_ARR[@]}
N_SEEDS=${#SEED_ARR[@]}

log "==== Claude Code parallel sweep ===="
log "suite=$SUITE  task=$TASK  seeds=${SEEDS}  gpus=${GPUS}  model=$MODEL"
log "master log: $MASTER_LOG  concurrency=$N_GPUS  stagger=${STAGGER_S}s"
log "max_budget_usd=$MAX_BUDGET_USD  max_turns=$MAX_TURNS"

# Track PIDs and which GPU each owns, so we can reuse a GPU when its
# cell finishes (round-robin queue if N_SEEDS > N_GPUS).
declare -A SLOT_PID
declare -A SLOT_TAG

# Helper: wait until any one slot is free, return its index in ${!SLOT_PID[@]}
free_slot() {
    while true; do
        for s in "${!SLOT_PID[@]}"; do
            if ! kill -0 "${SLOT_PID[$s]}" 2>/dev/null; then
                # finished
                tag=${SLOT_TAG[$s]}
                wait "${SLOT_PID[$s]}" 2>/dev/null || true
                # Write straight to the log file, NOT via log(): log() tees to
                # stdout, and this function's stdout is captured by $(free_slot).
                # The captured log text used to corrupt the returned slot index,
                # which silently dropped every queued seed (seeds > GPUs).
                echo "[$(date +%T)]   [done] slot $s (gpu ${GPU_ARR[$s]}) tag=$tag" >> "$MASTER_LOG"
                unset SLOT_PID[$s]
                unset SLOT_TAG[$s]
                echo "$s"
                return 0
            fi
        done
        sleep 5
    done
}

T_TOTAL=$(date +%s)
launched=0
SKIP_OUTPUT_DIR=${OUTPUT_DIR:-/mnt/public2/zhangyixian/RLinf_agentic/examples/embodiment/primitives/workspace_pro/results_claude_p_runs_parallel}
for seed in $SEEDS; do
    # Pre-launch skip: if this cell already has an audit, don't allocate a
    # slot OR pay the STAGGER_S sleep. run_one_cell.sh skips too, but only
    # AFTER the parent has slept the stagger — so on a resume across many
    # already-done cells the orchestrator used to crawl ~9 min/batch in pure
    # sleep. This makes resume instant for finished cells while real launches
    # keep their 60s EGL stagger.
    tag_pre=${SUITE/libero_/}_t${TASK}_s${seed}
    if [ -f "${SKIP_OUTPUT_DIR}/${tag_pre}.json" ]; then
        log "[skip] $tag_pre (audit exists; no slot, no stagger)"
        continue
    fi

    # Find a free slot — initially all slots are free; later we wait
    slot=""
    if [ ${#SLOT_PID[@]} -lt $N_GPUS ]; then
        # Find first unused slot
        for s in $(seq 0 $((N_GPUS-1))); do
            if [ -z "${SLOT_PID[$s]:-}" ]; then
                slot=$s; break
            fi
        done
    else
        # All slots in use — wait for one to free
        slot=$(free_slot)
    fi

    gpu=${GPU_ARR[$slot]}
    tag=${SUITE/libero_/}_t${TASK}_s${seed}
    log "[launch] slot=$slot gpu=$gpu  tag=$tag"

    # stagger before launching the next one
    if [ $launched -gt 0 ]; then
        sleep "$STAGGER_S"
    fi

    SUITE="$SUITE" TASK="$TASK" SEED="$seed" \
    CUDA_DEVICE="$gpu" MODEL="$MODEL" \
    MAX_TURNS="$MAX_TURNS" MAX_BUDGET_USD="$MAX_BUDGET_USD" \
    MAX_EPISODE_STEPS="$MAX_EPISODE_STEPS" \
    OUTPUT_DIR="${OUTPUT_DIR:-/mnt/public2/zhangyixian/RLinf_agentic/examples/embodiment/primitives/workspace_pro/results_claude_p_runs_parallel}" \
        bash "$RUN_ONE" "$SUITE" "$TASK" "$seed" \
        >> "$MASTER_LOG" 2>&1 &
    SLOT_PID[$slot]=$!
    SLOT_TAG[$slot]=$tag
    launched=$((launched+1))
done

log ""
log "[parallel] all $launched seeds launched; waiting for stragglers..."
for s in "${!SLOT_PID[@]}"; do
    wait "${SLOT_PID[$s]}" 2>/dev/null || true
    log "  [done] slot $s tag=${SLOT_TAG[$s]}"
done

ELAPSED=$(( $(date +%s) - T_TOTAL ))
log ""
log "==== ALL DONE in ${ELAPSED}s ===="

# Summary
log "=== summary ==="
OUTPUT_DIR=${OUTPUT_DIR:-/mnt/public2/zhangyixian/RLinf_agentic/examples/embodiment/primitives/workspace_pro/results_claude_p_runs_parallel}
n_total=0; n_ok=0
for seed in $SEEDS; do
    tag=${SUITE/libero_/}_t${TASK}_s${seed}
    audit=$OUTPUT_DIR/${tag}.json
    n_total=$((n_total+1))
    if [ -f "$audit" ]; then
        term=$(/opt/venv/openpi/bin/python -c "import json;d=json.load(open('$audit'));print(d.get('libero_terminated'))" 2>/dev/null)
        log "  $tag  libero_term=$term"
        [ "$term" = "True" ] && n_ok=$((n_ok+1))
    else
        log "  $tag  MISSING_AUDIT"
    fi
done
log ""
log "FINAL: $n_ok/$n_total succeeded"
