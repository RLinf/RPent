#!/usr/bin/env bash
# One-command launcher for the unseen explore sweep WITH grasp-detection logging.
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# See scripts/RUNBOOK_unseen_explore.md for full details.
#
#   bash scripts/robocasa_start_unseen_explore.sh           # resume (skip cells with an audit)
#   FRESH=1 bash scripts/robocasa_start_unseen_explore.sh   # archive all 16 unseen results, run all
#
# Env overrides: PARALLEL CELL_TIMEOUT GPUS_LIST SEEDS_LIST  (passed to robocasa_run_explore_unseen.sh)
set -u
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO"
RESULTS="$REPO_ROOT/explore_results"
UNSEEN=(ArrangeBreadBasket ArrangeTea BreadSelection CategorizeCondiments
        CuttingToolSelection GarnishPancake GatherTableware HeatKebabSandwich
        MakeIceLemonade PanTransfer PortionHotDogs RecycleBottlesByType
        SeparateFreezerRack WaffleReheat WashFruitColander WeighIngredients)

# ── 1. CONNECTIVITY GATE — child claude -p must reach the real API ──────────────
echo "[start] connectivity gate (standalone claude -p)..."
OUT=$(env -u CLAUDE_CODE_SSE_PORT -u CLAUDE_CODE_CHILD_SESSION -u CLAUDE_CODE_ENTRYPOINT \
          -u CLAUDECODE -u CLAUDE_CODE_SESSION_ID -u AI_AGENT CLAUDE_CODE_DISABLE_AUTO_MEMORY=1 \
        timeout 90 claude -p "Reply with exactly: CONNECTIVITY_OK" \
          --output-format stream-json --verbose 2>&1)
if echo "$OUT" | grep -q CONNECTIVITY_OK; then
  echo "[start] connectivity OK."
else
  echo "[start] ❌ NO API CONNECTIVITY (ConnectionRefused / retry). Aborting — fix the"
  echo "         connection first, else every cell dies in ~3 min having run 0 commands."
  echo "$OUT" | grep -o 'ConnectionRefused\|api_retry\|CONNECTIVITY_OK' | sort | uniq -c
  exit 1
fi

# ── 2. optional FRESH: archive existing unseen result dirs so all 16 re-run ─────
if [ "${FRESH:-0}" = "1" ]; then
  ARCH="$RESULTS/_archive_$(ls -d "$RESULTS"/_archive_* 2>/dev/null | wc -l)"
  mkdir -p "$ARCH"; n=0
  for d in "${UNSEEN[@]}"; do
    [ -d "$RESULTS/$d" ] && mv "$RESULTS/$d" "$ARCH/" && n=$((n+1))
  done
  echo "[start] FRESH: archived $n unseen result dirs -> $ARCH"
else
  have=0; for d in "${UNSEEN[@]}"; do [ -f "$RESULTS/$d/${d}_s0.json" ] && have=$((have+1)); done
  echo "[start] resume mode: $have/16 cells already have an audit and will be SKIPPED"
  echo "        (use FRESH=1 to archive + re-run all 16)"
fi

# ── 3. launch the sweep detached ────────────────────────────────────────────────
mkdir -p "$RESULTS/_sweep_logs"
SWEEP_OUT="$RESULTS/_sweep_logs/sweep_unseen_$(ls "$RESULTS"/_sweep_logs/sweep_unseen_*.out 2>/dev/null | wc -l).out"
setsid bash "$REPO_ROOT/scripts/robocasa_robocasa_run_explore_unseen.sh" > "$SWEEP_OUT" 2>&1 &
echo "[start] sweep launched (pid $!). overview -> $SWEEP_OUT"
sleep 3; sed -n '1,4p' "$SWEEP_OUT" 2>/dev/null

cat <<EOF

[start] LOGS:
  live workdir : /tmp/explore_<TASK>_s0/{log_NN.json,state_NN.json,agent.log,_prompt.md}
  archived     : $RESULTS/<TASK>/run_logs/<TASK>_s0/  (after each cell finishes)
  audits       : $RESULTS/<TASK>/<TASK>_s0.json
  sweep overview: $SWEEP_OUT   (per-cell: _sweep_logs/<TASK>_s0.log)

[start] MONITOR grasp detection:
  python3 scripts/monitor_grasp.py                    # summary table (all cells)
  python3 scripts/monitor_grasp.py <TASK> --calls     # per-VLA-call grasp signals
EOF
