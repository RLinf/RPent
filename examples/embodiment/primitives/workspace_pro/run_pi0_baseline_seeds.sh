#!/bin/bash
# Pi0 fullshot baseline sweep for libero_spatial_lan task 0, seeds 0..9.
# Distributes 10 baselines across 4 GPUs (2-3 per GPU concurrently).

set -e
cd /mnt/public2/zhangyixian/RLinf_agentic

SUITE=${SUITE:-libero_spatial_lan}
TASK=${TASK:-0}
SEEDS=${SEEDS:-"0 1 2 3 4 5 6 7 8 9"}
OUTDIR=${OUTDIR:-examples/embodiment/primitives/workspace_pro/results_pi0_baseline_seeds}
LOGDIR=/tmp/pi0_baseline_logs
mkdir -p $OUTDIR $LOGDIR

GPUS=(0 1 2 3)
i=0
pids=()
for seed in $SEEDS; do
    gpu=${GPUS[$((i % 4))]}
    out=$OUTDIR/baseline_${SUITE}_t${TASK}_s${seed}.json
    img=$OUTDIR/imgs/${SUITE}_t${TASK}_s${seed}
    log=$LOGDIR/pi0_baseline_${SUITE}_t${TASK}_s${seed}.log
    if [ -f "$out" ]; then
        echo "[skip] $out exists"
        continue
    fi
    echo "[launch] seed=$seed GPU=$gpu -> $out"
    LIBERO_TYPE=${LIBERO_TYPE:-standard} CUDA_VISIBLE_DEVICES=$gpu /opt/venv/openpi/bin/python \
        examples/embodiment/primitives/pi0_baseline.py \
        --suite $SUITE --task $TASK --seed $seed --max_chunks 60 \
        --out $out --save_image_dir $img \
        > $log 2>&1 &
    pids+=($!)
    i=$((i+1))
    # stagger to spread Pi0 model-load IO
    sleep 10
done

echo "[parallel] waiting for ${#pids[@]} baselines..."
for pid in "${pids[@]}"; do
    wait $pid || true
done

echo ""
echo "[done] all baselines"
echo ""
echo "=== summary ==="
for seed in $SEEDS; do
    out=$OUTDIR/baseline_${SUITE}_t${TASK}_s${seed}.json
    if [ -f "$out" ]; then
        term=$(/opt/venv/openpi/bin/python -c "import json;d=json.load(open('$out'));print(d.get('libero_terminated'))" 2>/dev/null)
        chunks=$(/opt/venv/openpi/bin/python -c "import json;d=json.load(open('$out'));print(d.get('result',{}).get('chunks_used'))" 2>/dev/null)
        echo "  seed=$seed: libero_term=$term  chunks=$chunks"
    else
        echo "  seed=$seed: MISSING"
    fi
done
