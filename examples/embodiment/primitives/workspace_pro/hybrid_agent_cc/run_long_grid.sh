#!/bin/bash
# Full libero_10 ("long", long-horizon) PRO sweep via claude -p:
#   3 perturbation regimes (task, swap, lan) x 10 tasks x 10 seeds = 300 cells.
#
# Mirrors run_spatial_grid.sh, with libero_10-specific knobs:
#   - MAX_EPISODE_STEPS=5000 passed EXPLICITLY. run_one_cell.sh auto-bumps
#     libero_10 to 5000, but run_parallel_seeds.sh would otherwise pass its
#     own default 600 and clobber that (memory max-episode-steps-libero) ->
#     mid-recipe ValueError, no audit. So we set it here and thread it through.
#   - CELL_TIMEOUT_S=1200 (exported). libero_10 cells run longer than the
#     ~183s spatial/object cells (more primitives, more Pi0 chunks, more image
#     reads), so the 600s default kills legitimately-progressing cells. Re-run
#     the grid afterward to backfill any that still time out (skip-if-audit
#     makes that safe).
#
# Concurrency: 4-way (4 GPUs, 1 cell/GPU) via GPUS="0 1 2 3" + STAGGER_S=60.
#   This is the documented-stable default. The dangerous config is 8 DISTINCT
#   GPUs at 1/GPU (8 simultaneous EGL contexts -> EGL_NOT_INITIALIZED, the
#   2026-05-23 incident). 2/GPU (GPUS="0 1 2 3 0 1 2 3") is also validated and
#   roughly doubles throughput, but 4-way is the conservative choice for an
#   unattended overnight run.
#
# Launch detached:
#   nohup bash examples/embodiment/primitives/workspace_pro/hybrid_agent_cc/run_long_grid.sh \
#       > /tmp/long_grid_outer.log 2>&1 &

set -e
cd /mnt/public2/zhangyixian/RLinf_agentic
HCC=examples/embodiment/primitives/workspace_pro/hybrid_agent_cc
RES=examples/embodiment/primitives/workspace_pro

SUITES=${SUITES:-"libero_10_task libero_10_swap libero_10_lan"}
TASKS=${TASKS:-"0 1 2 3 4 5 6 7 8 9"}
SEEDS=${SEEDS:-"0 1 2 3 4 5 6 7 8 9"}
GPUS=${GPUS:-"0 1 2 3"}
MODEL=${MODEL:-claude-opus-4-7}
STAGGER_S=${STAGGER_S:-60}
MAX_BUDGET_USD=${MAX_BUDGET_USD:-10}
# libero_10-specific: long-horizon episode budget + longer worker wall cap.
MAX_EPISODE_STEPS=${MAX_EPISODE_STEPS:-5000}
export CELL_TIMEOUT_S=${CELL_TIMEOUT_S:-1200}
export LIBERO_TYPE=${LIBERO_TYPE:-pro}

OUTPUT_DIR=${OUTPUT_DIR:-$RES/multi_seed_exp/long}
mkdir -p "$OUTPUT_DIR"

T0=$(date +%s)
echo "[$(date +%T)] libero_10 (long) PRO sweep start"
echo "  suites=[$SUITES]"
echo "  tasks=[$TASKS]  seeds=[$SEEDS]  gpus=[$GPUS]  model=$MODEL  libero_type=$LIBERO_TYPE"
echo "  max_episode_steps=$MAX_EPISODE_STEPS  cell_timeout_s=$CELL_TIMEOUT_S"
echo "  output_dir=$OUTPUT_DIR"

for suite in $SUITES; do
    short=${suite/libero_/}
    for task in $TASKS; do
        echo "[$(date +%T)] ===== $suite t$task (${SEEDS// /,} seeds) -> $OUTPUT_DIR ====="
        SUITE=$suite TASK=$task SEEDS="$SEEDS" GPUS="$GPUS" \
        MODEL=$MODEL STAGGER_S=$STAGGER_S MAX_BUDGET_USD=$MAX_BUDGET_USD \
        MAX_EPISODE_STEPS=$MAX_EPISODE_STEPS \
        OUTPUT_DIR=$OUTPUT_DIR \
        MASTER_LOG=/tmp/long_grid_${short}_t${task}.log \
            bash "$HCC/run_parallel_seeds.sh" \
            || echo "[$(date +%T)] WARN: $suite t$task returned nonzero (continuing)"
    done
done

echo "[$(date +%T)] libero_10 (long) PRO sweep DONE in $(( $(date +%s) - T0 ))s"

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
