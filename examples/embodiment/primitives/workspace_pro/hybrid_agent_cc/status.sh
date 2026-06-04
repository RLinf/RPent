#!/bin/bash
# Print live status of a Claude Code sequential sweep (for monitoring).
#
# Usage (from anywhere):
#   bash status.sh                              # default cell
#   SUITE=libero_object_task TASK=2 bash status.sh

SUITE=${SUITE:-libero_spatial_lan}
TASK=${TASK:-0}
SEEDS=${SEEDS:-"0 1 2 3 4 5 6 7 8 9"}
OUTPUT_DIR=${OUTPUT_DIR:-/mnt/public2/zhangyixian/RLinf_agentic/examples/embodiment/primitives/workspace_pro/results_claude_p_runs}

echo "=== Claude Code sweep status — $SUITE t$TASK ==="
echo "$(date +%T)"
echo ""

# Check what's currently running
ALIVE=$(ps -ef | grep -E "claude -p|run_all_seeds.sh|interactive_driver.py" | grep -v grep | awk '{print $2 " " $9 " " $10 " " $11 " " $12 " " $13 " " $14 " " $15 " " $16 " " $17 " " $18}')
if [ -n "$ALIVE" ]; then
    echo "ACTIVE PROCESSES:"
    echo "$ALIVE" | head -5 | awk '{print "  "$0}'
else
    echo "no active processes"
fi
echo ""

# Per-seed state
echo "PER-CELL STATE:"
n_done=0; n_pending=0; n_running=0; n_ok=0
for seed in $SEEDS; do
    tag=${SUITE/libero_/}_t${TASK}_s${seed}
    wd=/tmp/hybrid_repl_$tag
    audit=$OUTPUT_DIR/${tag}.json
    if [ -f "$audit" ]; then
        term=$(/opt/venv/openpi/bin/python -c "import json;d=json.load(open('$audit'));print(d.get('libero_terminated'))" 2>/dev/null)
        printf "  s%-2s  DONE     libero_term=%s\n" "$seed" "$term"
        n_done=$((n_done+1))
        [ "$term" = "True" ] && n_ok=$((n_ok+1))
    elif [ -d "$wd" ] && [ -f "$wd/state_00.json" ]; then
        n=$(ls $wd/done_*.flag 2>/dev/null | wc -l)
        printf "  s%-2s  RUNNING  cmds=%s\n" "$seed" "$n"
        n_running=$((n_running+1))
    else
        printf "  s%-2s  pending\n" "$seed"
        n_pending=$((n_pending+1))
    fi
done

echo ""
echo "SUMMARY: $n_done done ($n_ok success), $n_running running, $n_pending pending"
