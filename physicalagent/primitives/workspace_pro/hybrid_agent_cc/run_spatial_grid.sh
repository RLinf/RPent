#!/bin/bash
# Full libero_spatial PRO sweep via claude -p:
#   3 perturbation regimes (task, lan, swap) x 10 tasks x 10 seeds = 300 cells.
#
# Structure (mirrors run_base_sweep.sh but for PRO):
#   - WITHIN a (suite,task): the 10 seeds run in PARALLEL across $GPUS
#     (concurrency = len(GPUS); default 8 = one cell per GPU on GPUs 0-7).
#     With 10 seeds on 8 GPUs the last 2 round-robin onto freed slots.
#   - ACROSS (suite,task) pairs: SEQUENTIAL. run_parallel_seeds.sh blocks
#     until its 10 seeds finish, so at most 8 claude -p sessions are live.
#   - One FLAT output dir: multi_seed_exp/spatial/ (tags already unique).
#   - One master log per (suite,task) in /tmp.
#
# LIBERO_TYPE=pro is forced — these suites are PRO perturbations.
#
# Launch detached:
#   nohup bash physicalagent/primitives/workspace_pro/hybrid_agent_cc/run_spatial_grid.sh \
#       > /tmp/spatial_grid_outer.log 2>&1 &

set -e
cd /mnt/public/jxqiu/physicalagent
HCC=physicalagent/primitives/workspace_pro/hybrid_agent_cc
RES=physicalagent/primitives/workspace_pro

SUITES=${SUITES:-"libero_spatial_task libero_spatial_lan libero_spatial_swap"}
TASKS=${TASKS:-"0 1 2 3 4 5 6 7 8 9"}
SEEDS=${SEEDS:-"0 1 2 3 4 5 6 7 8 9"}
# 4-GPU default (1 cell/GPU) — empirically stable. 8-way (1/GPU) hit
# EGL_NOT_INITIALIZED crashes on 5/10 t1 cells on 2026-05-23; 4 GPUs
# avoids the simultaneous EGL contention that triggers the crash.
GPUS=${GPUS:-"0 1 2 3"}
MODEL=${MODEL:-claude-opus-4-7}
# 60s stagger spreads Pi0 disk load + EGL init further apart than the
# 30s default; pays ~2 min per batch but avoids the crash.
STAGGER_S=${STAGGER_S:-60}
MAX_BUDGET_USD=${MAX_BUDGET_USD:-10}
export LIBERO_TYPE=${LIBERO_TYPE:-pro}

OUTPUT_DIR=${OUTPUT_DIR:-$RES/multi_seed_exp/spatial}
mkdir -p "$OUTPUT_DIR"

T0=$(date +%s)
echo "[$(date +%T)] spatial PRO sweep start"
echo "  suites=[$SUITES]"
echo "  tasks=[$TASKS]  seeds=[$SEEDS]  gpus=[$GPUS]  model=$MODEL  libero_type=$LIBERO_TYPE"
echo "  output_dir=$OUTPUT_DIR"

for suite in $SUITES; do
    short=${suite/libero_/}
    for task in $TASKS; do
        echo "[$(date +%T)] ===== $suite t$task (${SEEDS// /,} seeds) -> $OUTPUT_DIR ====="
        SUITE=$suite TASK=$task SEEDS="$SEEDS" GPUS="$GPUS" \
        MODEL=$MODEL STAGGER_S=$STAGGER_S MAX_BUDGET_USD=$MAX_BUDGET_USD \
        OUTPUT_DIR=$OUTPUT_DIR \
        MASTER_LOG=/tmp/spatial_grid_${short}_t${task}.log \
            bash "$HCC/run_parallel_seeds.sh" \
            || echo "[$(date +%T)] WARN: $suite t$task returned nonzero (continuing)"
    done
done

echo "[$(date +%T)] spatial PRO sweep DONE in $(( $(date +%s) - T0 ))s"

# Final summary
echo ""
echo "==== FINAL SUMMARY ===="
n_total=0; n_ok=0
for suite in $SUITES; do
    short=${suite/libero_/}
    for task in $TASKS; do
        for seed in $SEEDS; do
            tag=${short}_t${task}_s${seed}
            audit=$OUTPUT_DIR/${tag}.json
            n_total=$((n_total+1))
            if [ -f "$audit" ]; then
                term=$(/opt/venv/openpi/bin/python -c "import json;d=json.load(open('$audit'));print(d.get('libero_terminated'))" 2>/dev/null)
                [ "$term" = "True" ] && n_ok=$((n_ok+1))
            fi
        done
    done
done
echo "FINAL: $n_ok/$n_total succeeded"
