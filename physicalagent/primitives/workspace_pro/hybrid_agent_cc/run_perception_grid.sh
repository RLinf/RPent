#!/bin/bash
# PERCEPTION-ISOLATED multi-seed grid launcher.
#
# Runs LIBERO-PRO cells where the worker gets NO object world coords — it must
# localize from image_cam_NN.png + depth_NN.npy + camera_meta.json (back-
# projection). Loops over REGIMES x TASKS, each (suite,task) group parallelized
# across GPUS by run_parallel_seeds.sh. Results go to NEW percep_* dirs.
#
# It bakes in the three things that make a run "perception-isolated" (which the
# oracle-mode ONBOARDING does NOT set):
#   PROMPT_TEMPLATE     -> agent_task_prompt_perception.md (localize-from-depth)
#   DRIVER_EXTRA_FLAGS  -> --hide_object_coords --always_render
#   CELL_TIMEOUT_S      -> 1200 (perceptual localization + manipulation is slower)
#
# Usage (object swap+task, all tasks, seeds 0-9, 4 GPUs):
#   ENV_BASE=libero_object REGIMES="swap task" \
#   TASKS="0 1 2 3 4 5 6 7 8 9" SEEDS="0 1 2 3 4 5 6 7 8 9" \
#   GPUS="0 1 2 3" \
#     bash run_perception_grid.sh > /tmp/percep_grid_outer.log 2>&1 &
#
# Pilot (one task, few seeds):
#   ENV_BASE=libero_object REGIMES="swap" TASKS="2" SEEDS="0 1 2" GPUS="0 1 2" \
#     bash run_perception_grid.sh > /tmp/percep_pilot.log 2>&1 &
set -u

REPO=/mnt/public/jxqiu/physicalagent
PERT=$REPO/physicalagent/primitives/workspace_pro
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
RUNNER="$SCRIPT_DIR/run_parallel_seeds.sh"
PROMPT="$SCRIPT_DIR/agent_task_prompt_perception.md"

ENV_BASE=${ENV_BASE:-libero_object}          # libero_object | libero_spatial | libero_goal | libero_10
REGIMES=${REGIMES:-"swap task"}              # perturbation regimes to sweep
TASKS=${TASKS:-"0 1 2 3 4 5 6 7 8 9"}
SEEDS=${SEEDS:-"0 1 2 3 4 5 6 7 8 9"}
GPUS=${GPUS:-"0 1 2 3"}
MODEL=${MODEL:-claude-opus-4-7}
STAGGER_S=${STAGGER_S:-60}
MAX_BUDGET_USD=${MAX_BUDGET_USD:-10}
# libero_10 is long-horizon; everything else is short.
case "$ENV_BASE" in
  *libero_10*) MAX_EPISODE_STEPS=${MAX_EPISODE_STEPS:-5000} ;;
  *)           MAX_EPISODE_STEPS=${MAX_EPISODE_STEPS:-2000} ;;
esac
OUT_ROOT=${OUT_ROOT:-$PERT/multi_seed_exp}
OUTER_LOG=${OUTER_LOG:-/tmp/percep_grid_${ENV_BASE#libero_}_outer.log}

short=${ENV_BASE#libero_}
echo "[$(date +%T)] ==== perception grid: env=$ENV_BASE regimes=[$REGIMES] tasks=[$TASKS] seeds=[$SEEDS] ====" | tee "$OUTER_LOG"

for regime in $REGIMES; do
  for task in $TASKS; do
    suite=${ENV_BASE}_${regime}
    outdir=$OUT_ROOT/percep_${short}_${regime}_t${task}
    mlog=/tmp/percep_${short}_${regime}_t${task}.log
    echo "[$(date +%T)] >>> group regime=$regime task=$task seeds=[$SEEDS] -> $outdir" | tee -a "$OUTER_LOG"
    SUITE="$suite" TASK="$task" SEEDS="$SEEDS" \
    GPUS="$GPUS" MODEL="$MODEL" STAGGER_S="$STAGGER_S" \
    LIBERO_TYPE=pro MAX_BUDGET_USD="$MAX_BUDGET_USD" \
    MAX_EPISODE_STEPS="$MAX_EPISODE_STEPS" CELL_TIMEOUT_S=1200 \
    PROMPT_TEMPLATE="$PROMPT" \
    DRIVER_EXTRA_FLAGS="--hide_object_coords --always_render" \
    OUTPUT_DIR="$outdir" MASTER_LOG="$mlog" \
        bash "$RUNNER" >> "$OUTER_LOG" 2>&1
    echo "[$(date +%T)] <<< group regime=$regime task=$task done" | tee -a "$OUTER_LOG"
  done
done
echo "[$(date +%T)] ==== ALL PERCEPTION GROUPS DONE ====" | tee -a "$OUTER_LOG"
