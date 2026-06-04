#!/bin/bash
# PERCEPTION-ISOLATED + NO-SKILL-LIBRARY multi-seed grid launcher.
#
# Same as hybrid_agent_cc/run_perception_grid.sh (worker gets NO object world
# coords — it localizes from image_cam + depth + camera_meta), with ONE extra
# ablation: the worker also gets NO worked-solution prior. It must NOT read any
# recipe_*.jsonl or prior audit *.json; it solves each cell from scratch using
# only the image + the operating memory + the generic guides. This isolates the
# value of the skill/recipe library: compare these results against the matching
# percep_* cells (which DID have the recipe prior).
#
# What it bakes in (vs the oracle ONBOARDING which sets none of these):
#   PROMPT_TEMPLATE     -> no_skill_exp/agent_task_prompt_no_skill.md
#   DRIVER_EXTRA_FLAGS  -> --hide_object_coords --always_render   (perception)
#   CELL_TIMEOUT_S      -> 1200   (localize + manipulate + reason-from-scratch is slower)
# Everything else (driver, parallel runner, per-cell runner, memory snapshot) is
# shared with the perception harness in ../hybrid_agent_cc/.
#
# Usage (the pilot — object swap, task 2, seeds 0-2, 3 GPUs):
#   ENV_BASE=libero_object REGIMES="swap" TASKS="2" SEEDS="0 1 2" GPUS="0 1 2" \
#     bash run_no_skill_grid.sh > /tmp/noskill_pilot.log 2>&1 &
#
# Usage (full object swap+task A/B vs percep_object, 4 GPUs):
#   ENV_BASE=libero_object REGIMES="swap task" \
#   TASKS="0 1 2 3 4 5 6 7 8 9" SEEDS="0 1 2 3 4 5 6 7 8 9" GPUS="0 1 2 3" \
#     bash run_no_skill_grid.sh > /tmp/noskill_object_outer.log 2>&1 &
set -u

# ── CLOSE THE AUTO-MEMORY LEAK (critical for a valid no-skill ablation) ──
# Claude Code auto-injects this git-repo's project memory
# (~/.claude_local/projects/<cwd-slug>/memory/MEMORY.md) into every `claude -p`
# worker's context, and that store has accumulated per-CELL solved experience
# from prior runs (e.g. "object_swap t0 needs full BDDL prompt", the percep
# back-projection fix "confirmed on object_swap t2 s0"). That is exactly the
# skill experience this experiment removes — so disable it. The var is exported
# here and inherited all the way down to the `claude -p` subprocess (grid ->
# run_parallel_seeds -> run_one_cell -> claude); no shared-script edits needed.
# Verified to actually close the channel via hard_audit_no_skill.sh.
export CLAUDE_CODE_DISABLE_AUTO_MEMORY=1

REPO=/mnt/public2/zhangyixian/RLinf_agentic
PERT=$REPO/examples/embodiment/primitives/workspace_pro
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
# Reuse the shared parallel runner + per-cell runner from the perception harness.
RUNNER="$PERT/hybrid_agent_cc/run_parallel_seeds.sh"
PROMPT="$SCRIPT_DIR/agent_task_prompt_no_skill.md"

ENV_BASE=${ENV_BASE:-libero_object}          # libero_object | libero_spatial | libero_goal | libero_10
REGIMES=${REGIMES:-"swap"}                   # perturbation regimes to sweep
TASKS=${TASKS:-"2"}
SEEDS=${SEEDS:-"0 1 2"}
GPUS=${GPUS:-"0 1 2"}
MODEL=${MODEL:-claude-opus-4-7}
STAGGER_S=${STAGGER_S:-60}
MAX_BUDGET_USD=${MAX_BUDGET_USD:-10}
# libero_10 is long-horizon; everything else is short.
case "$ENV_BASE" in
  *libero_10*) MAX_EPISODE_STEPS=${MAX_EPISODE_STEPS:-5000} ;;
  *)           MAX_EPISODE_STEPS=${MAX_EPISODE_STEPS:-2000} ;;
esac
# No-skill results live UNDER no_skill_exp/, separate from the perception
# multi_seed_exp/ runs so the A/B comparison stays clean and un-clobbered.
OUT_ROOT=${OUT_ROOT:-$SCRIPT_DIR/results}
OUTER_LOG=${OUTER_LOG:-/tmp/noskill_grid_${ENV_BASE#libero_}_outer.log}

short=${ENV_BASE#libero_}
echo "[$(date +%T)] ==== no-skill grid: env=$ENV_BASE regimes=[$REGIMES] tasks=[$TASKS] seeds=[$SEEDS] ====" | tee "$OUTER_LOG"

for regime in $REGIMES; do
  for task in $TASKS; do
    suite=${ENV_BASE}_${regime}
    outdir=$OUT_ROOT/noskill_${short}_${regime}_t${task}
    mlog=/tmp/noskill_${short}_${regime}_t${task}.log
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
echo "[$(date +%T)] ==== ALL NO-SKILL GROUPS DONE ====" | tee -a "$OUTER_LOG"
