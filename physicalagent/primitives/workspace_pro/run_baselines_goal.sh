#!/bin/bash
# Run Pi0 fullshot baselines for libero_goal PRO 4-cell experiment, t0-t9.
# Uses CUDA_VISIBLE_DEVICES=1 so it doesn't compete with the hybrid driver
# (which uses GPU 0). Sequential to avoid OOM within one GPU.

set -e
cd /mnt/public/jxqiu/physicalagent

OUTDIR=physicalagent/primitives/workspace_pro/results_goal_pert
mkdir -p $OUTDIR/baseline_imgs

run_one() {
    local SUITE=$1
    local T=$2
    local TAG=$3
    local OUT=$OUTDIR/baseline_pi0_goal_${TAG}_t${T}_s0.json
    local IMG=$OUTDIR/baseline_imgs/${TAG}_t${T}
    if [ -f "$OUT" ]; then
        echo "[skip] $OUT exists"
        return
    fi
    echo "[run] $SUITE t$T -> $OUT"
    LIBERO_TYPE=pro CUDA_VISIBLE_DEVICES=1 /opt/venv/openpi/bin/python \
        physicalagent/primitives/pi0_baseline.py \
        --suite $SUITE --task $T --seed 0 --max_chunks 60 \
        --out $OUT \
        --save_image_dir $IMG \
        >> /tmp/pi0_baseline_goal.log 2>&1
}

for T in 0 1 2 3 4 5 6 7 8 9; do
    run_one libero_goal       $T base
    run_one libero_goal_task  $T task
    run_one libero_goal_swap  $T swap
    run_one libero_goal_lan   $T lan
done

echo "[done] all baselines"
