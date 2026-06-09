#!/bin/bash
# Full BASE-LIBERO sweep via claude -p:
#   4 suites x 10 tasks x 10 seeds = 400 cells.
#
# Structure (this is the "batching" answer):
#   - WITHIN a (suite,task): the 10 seeds run in PARALLEL across $GPUS
#     (concurrency = len(GPUS); default 8 = one cell per GPU on GPUs 0-7).
#   - ACROSS (suite,task) pairs: SEQUENTIAL. run_parallel_seeds.sh blocks
#     until its 10 seeds finish, so at most len(GPUS) claude -p sessions
#     are ever live at once. The onboarding validated 8 concurrent on one
#     subscription with zero rate-limit errors; do NOT run two pairs at once.
#   - BY SUITE: one output dir per suite (results_base_<short>/).
#   - One master log per (suite,task): /tmp/claude_p_base_<short>_t<task>.log
#
# LIBERO_TYPE defaults to "standard" (base LIBERO), NOT pro.
#
# Launch detached so it survives disconnects (~6-8h for the full 400 on sonnet):
#   nohup bash physicalagent/primitives/workspace_pro/hybrid_agent_cc/run_base_sweep.sh \
#       > /tmp/base_sweep_outer.log 2>&1 &
#
# Subset examples:
#   SUITES="libero_spatial"            bash run_base_sweep.sh   # one suite (100 cells)
#   SUITES="libero_object" TASKS="0 1" bash run_base_sweep.sh   # 2 tasks (20 cells)
#   MODEL=claude-opus-4-7              bash run_base_sweep.sh   # opus instead of sonnet

set -e
cd /mnt/public/jxqiu/physicalagent
HCC=physicalagent/primitives/workspace_pro/hybrid_agent_cc
RES=physicalagent/primitives/workspace_pro

SUITES=${SUITES:-"libero_spatial libero_object libero_goal libero_10"}
TASKS=${TASKS:-"0 1 2 3 4 5 6 7 8 9"}
SEEDS=${SEEDS:-"0 1 2 3 4 5 6 7 8 9"}
GPUS=${GPUS:-"0 1 2 3 4 5 6 7"}
MODEL=${MODEL:-sonnet}
STAGGER_S=${STAGGER_S:-30}
MAX_BUDGET_USD=${MAX_BUDGET_USD:-10}
export LIBERO_TYPE=${LIBERO_TYPE:-standard}

T0=$(date +%s)
echo "[$(date +%T)] BASE sweep start"
echo "  suites=[$SUITES]"
echo "  tasks=[$TASKS]  seeds=[$SEEDS]  gpus=[$GPUS]  model=$MODEL  libero_type=$LIBERO_TYPE"

for suite in $SUITES; do
    short=${suite/libero_/}
    outdir=$RES/results_base_${short}
    for task in $TASKS; do
        echo "[$(date +%T)] ===== $suite t$task (${SEEDS// /,} seeds) -> $outdir ====="
        SUITE=$suite TASK=$task SEEDS="$SEEDS" GPUS="$GPUS" \
        MODEL=$MODEL STAGGER_S=$STAGGER_S MAX_BUDGET_USD=$MAX_BUDGET_USD \
        OUTPUT_DIR=$outdir \
        MASTER_LOG=/tmp/claude_p_base_${short}_t${task}.log \
            bash "$HCC/run_parallel_seeds.sh" \
            || echo "[$(date +%T)] WARN: $suite t$task returned nonzero (continuing)"
    done
done

echo "[$(date +%T)] BASE sweep DONE in $(( $(date +%s) - T0 ))s"
