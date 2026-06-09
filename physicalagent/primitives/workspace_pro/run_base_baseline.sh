#!/bin/bash
# Pi0 fullshot BASE-LIBERO baseline for the same grid the hybrid sweep covers:
#   4 suites x 10 tasks x 10 seeds = 400 baseline runs.
#
# Pure GPU (no claude / no subscription quota). Each (suite,task) = 10 seeds
# across 4 GPUs (run_pi0_baseline_seeds.sh hardcodes GPUS=(0 1 2 3)), ~5 min.
# Pairs run sequentially -> ~40 x 5 min ~= 3-4h total.
#
# NOTE: this uses GPUs 0-3. Do NOT run it at the same time as run_base_sweep.sh
# (which uses GPUs 0-7) or they will contend on 0-3. Run baseline FIRST, then
# the hybrid sweep -- or run baseline on a box the hybrid sweep isn't using.
#
# LIBERO_TYPE defaults to "standard" (base LIBERO), NOT pro.
#
# Launch detached:
#   nohup bash physicalagent/primitives/workspace_pro/run_base_baseline.sh \
#       > /tmp/base_baseline_outer.log 2>&1 &

set -e
cd /mnt/public/jxqiu/physicalagent
RES=physicalagent/primitives/workspace_pro

SUITES=${SUITES:-"libero_spatial libero_object libero_goal libero_10"}
TASKS=${TASKS:-"0 1 2 3 4 5 6 7 8 9"}
SEEDS=${SEEDS:-"0 1 2 3 4 5 6 7 8 9"}
export LIBERO_TYPE=${LIBERO_TYPE:-standard}

T0=$(date +%s)
echo "[$(date +%T)] BASE Pi0 baseline start: suites=[$SUITES] tasks=[$TASKS] libero_type=$LIBERO_TYPE"

for suite in $SUITES; do
    short=${suite/libero_/}
    for task in $TASKS; do
        echo "[$(date +%T)] ===== baseline $suite t$task ====="
        SUITE=$suite TASK=$task SEEDS="$SEEDS" \
        OUTDIR=$RES/results_base_baseline_${short} \
            bash "$RES/run_pi0_baseline_seeds.sh" \
            > /tmp/pi0_base_${short}_t${task}.log 2>&1 \
            || echo "[$(date +%T)] WARN: baseline $suite t$task returned nonzero (continuing)"
    done
done

echo "[$(date +%T)] BASE Pi0 baseline DONE in $(( $(date +%s) - T0 ))s"
