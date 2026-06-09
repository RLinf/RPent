#!/bin/bash
# One-click perception-isolated cell launcher.
#
# Wraps run_one_cell.sh with the three things that make a cell perception-mode
# (and that the oracle default doesn't set):
#   PROMPT_TEMPLATE     -> agent_task_prompt_perception.md (localize-from-depth)
#   DRIVER_EXTRA_FLAGS  -> --hide_object_coords --always_render
#   CELL_TIMEOUT_S      -> 1200 (perception localization + manipulation is slower)
#
# Auto-routes LIBERO_TYPE=pro for any *_swap / *_task / *_lan suite (override
# with LIBERO_TYPE=standard env var if you want the base benchmark).
#
# Usage (most common):
#   bash run_perception_cell.sh libero_object_swap 2 0
#   bash run_perception_cell.sh libero_goal_task 5 0
#   bash run_perception_cell.sh libero_10_task 5 0
#   bash run_perception_cell.sh libero_spatial 3 0    # standard (no PRO suffix)
#
# Override defaults via env vars (just like run_one_cell.sh):
#   CUDA_DEVICE=2  MODEL=claude-opus-4-7  MAX_BUDGET_USD=10  OUTPUT_DIR=...  \
#     bash run_perception_cell.sh libero_object_swap 2 0
#
# See STRICT_HYBRID_GUIDE_PERCEPTION.md (in physicalagent/primitives/)
# for the full perception protocol the worker follows.
set -u

if [ $# -lt 3 ]; then
    cat <<HELP
usage: bash $(basename "$0") SUITE TASK SEED [OUTPUT_DIR]

required:
  SUITE       libero_{spatial,object,goal,10}[_swap|_task|_lan]
  TASK        task id (0-9)
  SEED        seed id (0-9)

optional:
  OUTPUT_DIR  defaults to /tmp/percep_cell_<tag>

env overrides (any of the run_one_cell.sh knobs work):
  CUDA_DEVICE, MODEL, LIBERO_TYPE, MAX_BUDGET_USD, MAX_EPISODE_STEPS,
  CELL_TIMEOUT_S, MAX_TURNS
HELP
    exit 1
fi

SUITE=$1
TASK=$2
SEED=$3

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
RUN_ONE="$SCRIPT_DIR/run_one_cell.sh"
PROMPT_PATH="$SCRIPT_DIR/agent_task_prompt_perception.md"

if [ ! -x "$RUN_ONE" ]; then chmod +x "$RUN_ONE"; fi
if [ ! -f "$PROMPT_PATH" ]; then
    echo "ERROR: missing perception prompt at $PROMPT_PATH" >&2
    exit 2
fi

# Default OUTPUT_DIR if caller didn't supply one
TAG=${SUITE/libero_/}_t${TASK}_s${SEED}
OUTPUT_DIR=${4:-${OUTPUT_DIR:-/tmp/percep_cell_${TAG}}}
mkdir -p "$OUTPUT_DIR"

# Auto-route LIBERO_TYPE based on suite suffix (caller can still override).
case "$SUITE" in
    *_swap|*_task|*_lan) LIBERO_TYPE_DEFAULT=pro ;;
    *)                   LIBERO_TYPE_DEFAULT=standard ;;
esac

LIBERO_TYPE=${LIBERO_TYPE:-$LIBERO_TYPE_DEFAULT} \
PROMPT_TEMPLATE="$PROMPT_PATH" \
DRIVER_EXTRA_FLAGS="--hide_object_coords --always_render" \
CELL_TIMEOUT_S=${CELL_TIMEOUT_S:-1200} \
MODEL=${MODEL:-claude-opus-4-7} \
MAX_BUDGET_USD=${MAX_BUDGET_USD:-10} \
CUDA_DEVICE=${CUDA_DEVICE:-0} \
OUTPUT_DIR="$OUTPUT_DIR" \
    bash "$RUN_ONE" "$SUITE" "$TASK" "$SEED"

# Surface the audit/recipe location for convenience
if [ -f "$OUTPUT_DIR/${TAG}.json" ]; then
    echo ""
    echo "[done] audit:  $OUTPUT_DIR/${TAG}.json"
    echo "[done] recipe: $OUTPUT_DIR/recipe_${TAG}.jsonl"
fi
