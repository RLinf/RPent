#!/bin/bash
# Run one (suite, task, seed) cell via `claude -p` (Claude Code subscription).
#
# Workflow:
#   1. Clean a per-cell workdir, start interactive_driver.py in background.
#   2. Wait until state_00.json appears (Pi0 load ~80s).
#   3. Substitute the agent_task_prompt.md template with cell vars.
#   4. Invoke `claude -p ...` with that prompt and Bash/Read/Write allowed.
#   5. Send {"action":"exit"} to the driver, kill it, return.
#
# Idempotent per cell: re-run with same args replays from scratch.
# Outputs: $OUTPUT_DIR/{recipe_<tag>.jsonl, <tag>.json, claude_<tag>.txt}

set -e

SUITE=${1:-libero_spatial_lan}
TASK=${2:-0}
SEED=${3:-0}
CUDA_DEVICE=${CUDA_DEVICE:-0}
# max_episode_steps is suite-dependent: libero_10 is long-horizon
# (~15 primitives incl. carry/articulate/set_object_pose, 1000+ cumulative
# env.steps) and blows through a 600 cap, hitting robosuite's
# "executing action in terminated episode" ValueError mid-recipe (which
# kills the worker with no audit — memory max-episode-steps-libero). Use
# 5000 there, matching the basic interactive-exploration default. Short
# suites (spatial/object/goal) are fine at 600. An explicit env var still
# wins over both.
case "$SUITE" in
  *libero_10*) MAX_EPISODE_STEPS=${MAX_EPISODE_STEPS:-5000} ;;
  *)           MAX_EPISODE_STEPS=${MAX_EPISODE_STEPS:-600} ;;
esac
MAX_TURNS=${MAX_TURNS:-60}
MODEL=${MODEL:-sonnet}
# LIBERO variant: standard (base) | pro (liberopro) | plus (liberoplus).
# Default standard so base-LIBERO runs don't rely on the silent pro->base fallback.
LIBERO_TYPE=${LIBERO_TYPE:-standard}

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
# Find repo root from this script (4 dirs up): hybrid_agent_cc -> workspace_pro -> primitives -> embodiment -> examples -> REPO
REPO="${REPO:-$(cd "$SCRIPT_DIR/../../../../.." && pwd)}"
PYTHON_BIN="${PYTHON_BIN:-/opt/venv/openpi/bin/python}"
MEMORY_DIR="${MEMORY_DIR:-$REPO/physicalagent/primitives/workspace_pro/memory_snapshot}"
TAG=${SUITE/libero_/}_t${TASK}_s${SEED}
WORKDIR=${WORKDIR_ROOT:-/tmp}/hybrid_repl_${TAG}
OUTPUT_DIR=${OUTPUT_DIR:-$REPO/physicalagent/primitives/workspace_pro/results_claude_p_runs}
PROMPT_TEMPLATE=${PROMPT_TEMPLATE:-$SCRIPT_DIR/agent_task_prompt.md}
DRIVER_LOG=/tmp/cc_driver_${TAG}.log
# Hard wall-clock cap on claude -p so a stuck worker (e.g. polling for a
# done_NN.flag that will never appear because the driver crashed with
# EOFError/EGL_NOT_INITIALIZED — see memory feedback_pi0_chunks_egl_crash)
# can't block the orchestrator forever. 2026-05-23 incident: 5 cells hung
# for 6.5h on this same failure mode.
CELL_TIMEOUT_S=${CELL_TIMEOUT_S:-600}

mkdir -p "$OUTPUT_DIR"

# === 0. skip if this cell already has a successful audit ===
# Lets the orchestrator be re-run idempotently after a partial sweep.
if [ -f "$OUTPUT_DIR/${TAG}.json" ]; then
    echo "[$(date +%T)] [$TAG] SKIP: audit already exists at $OUTPUT_DIR/${TAG}.json"
    exit 0
fi

# === 1. start driver ===
echo "[$(date +%T)] [$TAG] starting driver on GPU $CUDA_DEVICE"
rm -rf "$WORKDIR"
mkdir -p "$WORKDIR"

LIBERO_TYPE=$LIBERO_TYPE CUDA_VISIBLE_DEVICES=$CUDA_DEVICE \
    "$PYTHON_BIN" "$REPO/physicalagent/primitives/interactive_driver.py" \
    --suite "$SUITE" --task "$TASK" --seed "$SEED" \
    --workdir "$WORKDIR" --max_episode_steps "$MAX_EPISODE_STEPS" \
    ${DRIVER_EXTRA_FLAGS:-} \
    > "$DRIVER_LOG" 2>&1 &
DRIVER_PID=$!

# === 2. wait for driver ready ===
echo "[$(date +%T)] [$TAG] driver pid=$DRIVER_PID; waiting for state_00.json ..."
T0=$(date +%s)
while [ ! -f "$WORKDIR/state_00.json" ]; do
    sleep 5
    if ! kill -0 "$DRIVER_PID" 2>/dev/null; then
        echo "[$(date +%T)] [$TAG] driver died before becoming ready"
        tail -50 "$DRIVER_LOG"
        exit 2
    fi
    if [ $(( $(date +%s) - T0 )) -gt 300 ]; then
        echo "[$(date +%T)] [$TAG] driver not ready after 300s"
        kill -9 "$DRIVER_PID" 2>/dev/null || true
        exit 3
    fi
done
echo "[$(date +%T)] [$TAG] driver ready in $(( $(date +%s) - T0 ))s"

# === 3. build prompt by substituting placeholders ===
PROMPT_FILE=$(mktemp /tmp/cc_prompt_${TAG}.XXXXXX.md)
sed \
    -e "s|{SUITE}|$SUITE|g" \
    -e "s|{TASK}|$TASK|g" \
    -e "s|{SEED}|$SEED|g" \
    -e "s|{WORKDIR}|$WORKDIR|g" \
    -e "s|{TAG}|$TAG|g" \
    -e "s|{OUTPUT_DIR}|$OUTPUT_DIR|g" \
    "$PROMPT_TEMPLATE" > "$PROMPT_FILE"

# === 4. run claude -p ===
echo "[$(date +%T)] [$TAG] invoking claude -p (model=$MODEL, max-turns=$MAX_TURNS)"
CLAUDE_OUT="$OUTPUT_DIR/claude_${TAG}.txt"
T_CLAUDE=$(date +%s)

cd "$REPO"
# Disable set -e for this one command so we can capture the real exit
# code from claude -p (previously `|| true` swallowed it and CC_RC was
# always 0 — bug noticed 2026-05-22).
set +e

# === driver-death watchdog ===
# A long libero_10 episode can crash the driver subprocess mid-run with
# EGL_NOT_INITIALIZED / EOFError (memory feedback_pi0_chunks_egl_crash). When it
# does, the worker keeps polling for a done_NN.flag that can never appear and
# would otherwise burn the whole CELL_TIMEOUT_S in a no-op loop, holding the GPU
# slot idle (observed 2026-05-27: 3 swap_t0 cells each wasted 1200s this way).
# This watchdog notices the dead driver and kills the worker (plus its hung
# poll-loop bash children) immediately, targeted precisely by this cell's unique
# workdir tag, which appears in the worker's argv (--add-dir $WORKDIR + the poll
# commands) but NOT in run_one_cell's own argv. It exits on its own if the driver
# stays alive (normal completion), so it never fires on a healthy cell.
(
    while kill -0 "$DRIVER_PID" 2>/dev/null; do sleep 5; done
    sleep 10   # grace: let the driver flush its final crash traceback first
    if pgrep -f "hybrid_repl_${TAG}" >/dev/null 2>&1; then
        echo "[$(date +%T)] [$TAG] WATCHDOG: driver pid=$DRIVER_PID died mid-run; killing worker (no audit possible — backfill re-run will retry this cell)" | tee -a "$CLAUDE_OUT"
        pkill -TERM -f "hybrid_repl_${TAG}" 2>/dev/null
        sleep 3
        pkill -KILL -f "hybrid_repl_${TAG}" 2>/dev/null
    fi
) &
WATCHDOG_PID=$!

timeout --kill-after=15 "$CELL_TIMEOUT_S" \
    claude -p "$(cat "$PROMPT_FILE")" \
        --model "$MODEL" \
        --output-format text \
        --add-dir "$WORKDIR" \
        --add-dir "$MEMORY_DIR" \
        --allowedTools "Bash Read Write Glob Grep" \
        --max-budget-usd "${MAX_BUDGET_USD:-10}" \
        > "$CLAUDE_OUT" 2>&1
CC_RC=$?

# Stop the watchdog if it is still polling a live driver (normal-completion
# path). If the watchdog already fired (driver crashed), it has exited and these
# are no-ops.
kill "$WATCHDOG_PID" 2>/dev/null
wait "$WATCHDOG_PID" 2>/dev/null
set -e
if [ "$CC_RC" = 124 ]; then
    echo "[$(date +%T)] [$TAG] claude -p TIMEOUT after ${CELL_TIMEOUT_S}s (likely driver crash + worker poll loop)"
fi
echo "[$(date +%T)] [$TAG] claude -p finished in $(( $(date +%s) - T_CLAUDE ))s rc=$CC_RC"
rm -f "$PROMPT_FILE"

# === 5. stop driver ===
echo '{"action": "exit"}' > "$WORKDIR/command.json"
sleep 3
kill -9 "$DRIVER_PID" 2>/dev/null || true

# === 6. report ===
AUDIT="$OUTPUT_DIR/${TAG}.json"
if [ -f "$AUDIT" ]; then
    TERM=$("$PYTHON_BIN" -c "import json;d=json.load(open('$AUDIT'));print(d.get('libero_terminated'))" 2>/dev/null)
    echo "[$(date +%T)] [$TAG] DONE   libero_terminated=$TERM   audit=$AUDIT"
else
    # Fallback: check the last state_NN.json
    LAST_STATE=$(ls -1 $WORKDIR/state_*.json 2>/dev/null | sort | tail -1)
    if [ -n "$LAST_STATE" ]; then
        SIM_TERM=$("$PYTHON_BIN" -c "import json;d=json.load(open('$LAST_STATE'));print(d.get('libero_terminated'))" 2>/dev/null)
        echo "[$(date +%T)] [$TAG] NO AUDIT   sim_libero_terminated=$SIM_TERM"
    else
        echo "[$(date +%T)] [$TAG] NO STATE FILES"
    fi
fi
