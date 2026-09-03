#!/usr/bin/env bash

set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RPENT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
REPRO_ROOT="${RPENT_REPRO_ROOT:-${RPENT_ROOT}/.behavior-runtime}"
RLINF_ROOT="${RLINF_ROOT:-${REPRO_ROOT}/RLinf}"
RPENT_VENV="${RPENT_VENV:-${REPRO_ROOT}/venvs/rpent}"
BEHAVIOR_VENV="${BEHAVIOR_VENV:-${REPRO_ROOT}/venvs/behavior}"

: "${OMNIGIBSON_DATA_PATH:?Set OMNIGIBSON_DATA_PATH}"
: "${PI05_CHECKPOINT_PATH:?Set PI05_CHECKPOINT_PATH}"
: "${DINOV2_SOURCE_ARCHIVE:?Set DINOV2_SOURCE_ARCHIVE}"
: "${DINOV2_WEIGHTS:?Set DINOV2_WEIGHTS}"

"${SCRIPT_DIR}/verify_behavior_assets.sh"

TASK_NAME="${TASK_NAME:-turning_on_radio}"
PUBLIC_SEED="${PUBLIC_SEED:-1}"
ENV_GPU="${BEHAVIOR_ENV_GPU:-0}"
MODEL_GPU="${BEHAVIOR_MODEL_GPU:-1}"
DASHBOARD_HOST="${DASHBOARD_HOST:-127.0.0.1}"
DASHBOARD_PORT="${DASHBOARD_PORT:-8765}"
DASHBOARD_LANGUAGE="${DASHBOARD_LANGUAGE:-zh-cn}"
PLANNER="${PLANNER:-codex}"
PLANNER_MODEL="${PLANNER_MODEL:-gpt-5.5}"
OUTPUT_DIR="${OUTPUT_DIR:-${REPRO_ROOT}/logs/dashboard-$(date -u +%Y%m%dT%H%M%SZ)}"
MEMORY_DIR="${BEHAVIOR_MEMORY_DIR:-${REPRO_ROOT}/memory/behavior}"
EPISODE_MEMORY_DIR="${BEHAVIOR_EPISODE_MEMORY_DIR:-}"

episode_memory_args=()
if [[ -n "${EPISODE_MEMORY_DIR}" ]]; then
    episode_memory_args=(--behavior-memory-dir "${EPISODE_MEMORY_DIR}")
fi

mkdir -p "${OUTPUT_DIR}" "${MEMORY_DIR}"
export OMNI_KIT_ACCEPT_EULA=YES
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
export RAY_ACCEL_ENV_VAR_OVERRIDE_ON_ZERO=0
export RPENT_RLINF_ROOT="${RLINF_ROOT}"
export RPENT_BEHAVIOR_PYTHON="${BEHAVIOR_VENV}/bin/python"

cd "${RPENT_ROOT}"
exec "${RPENT_VENV}/bin/rpent" \
    --robot behavior \
    --dashboard \
    --dashboard-host "${DASHBOARD_HOST}" \
    --dashboard-port "${DASHBOARD_PORT}" \
    --dashboard-language "${DASHBOARD_LANGUAGE}" \
    --task-name "${TASK_NAME}" \
    --public-seed "${PUBLIC_SEED}" \
    --behavior-mode eval \
    --max-episode-steps "${MAX_EPISODE_STEPS:-43200}" \
    --planner "${PLANNER}" \
    --model "${PLANNER_MODEL}" \
    --reasoning-effort "${REASONING_EFFORT:-xhigh}" \
    --max-turns "${MAX_TURNS:-60}" \
    --planner-timeout-s "${PLANNER_TIMEOUT_S:-3600}" \
    --memory-profile local \
    --memory-dir "${MEMORY_DIR}" \
    "${episode_memory_args[@]}" \
    --output-dir "${OUTPUT_DIR}" \
    --behavior-repo "${RLINF_ROOT}" \
    --behavior-python "${BEHAVIOR_VENV}/bin/python" \
    --activity-instance-dir "${OMNIGIBSON_DATA_PATH}/2025-challenge-task-instances" \
    --policy-checkpoint "${PI05_CHECKPOINT_PATH}" \
    --behavior-env-cuda-device "${ENV_GPU}" \
    --behavior-model-cuda-device "${MODEL_GPU}" \
    --dino-source-archive "${DINOV2_SOURCE_ARCHIVE}" \
    --dino-weights "${DINOV2_WEIGHTS}" \
    --dino-cache-dir "${REPRO_ROOT}/cache/dinov2" \
    --vla-ready-timeout-s "${VLA_READY_TIMEOUT_S:-600}"
