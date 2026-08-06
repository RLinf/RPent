#!/usr/bin/env bash
# ============================================
# PhysicalAgent — 运行单个 RoboCasa 任务
# 直接调用 scripts/robocasa_run_explore.sh
# ============================================
# Usage: ./run_robocasa.sh <TASK> <GPU> <SEED> [AGENT_TIMEOUT_S]
# Examples:
#   ./run_robocasa.sh OpenDrawer 0 0 1800
#   ./run_robocasa.sh PickPlaceCounterToCabinet 1 42 600
# ============================================
set -u

TASK=${1:?Usage: ./run_robocasa.sh <TASK> <GPU> <SEED> [TIMEOUT]}
GPU=${2:?Usage: ./run_robocasa.sh <TASK> <GPU> <SEED> [TIMEOUT]}
SEED=${3:-0}
TIMEOUT=${4:-1800}

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PARENT="$(cd "$REPO_ROOT/.." && pwd)"

# 路径覆盖 (不修改原始代码, 通过环境变量注入)
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export MUJOCO_GL="${MUJOCO_GL:-egl}"
export PYOPENGL_PLATFORM="${PYOPENGL_PLATFORM:-egl}"
export NO_ALBUMENTATIONS_UPDATE=1
export RLDX_ATTN_IMPL="${RLDX_ATTN_IMPL:-sdpa}"
export UV_CACHE_DIR="$PARENT/.uv_cache"
export OUT_BASE="$PARENT/logs"

# 视频录制: 让 VLA rollout 的每步画面录成 mp4 (rldx_skill 内, 复用 obs 渲染帧)。
export RLDX_VIDEO="${RLDX_VIDEO:-1}"
# 每次运行的输出子目录格式 = 日期-任务 (e.g. 20260705-161230-SteamInMicrowave)。
# 所有产物 (audit / run_logs / 图片 / 视频) 落在 $OUT_BASE/$RUN_SUBDIR/ 下。
export RUN_SUBDIR="${RUN_SUBDIR:-$(date +%Y%m%d-%H%M%S)-${TASK}}"

echo "============================================"
echo "[run] Task:    $TASK"
echo "[run] GPU:     $GPU"
echo "[run] Seed:    $SEED"
echo "[run] Timeout: ${TIMEOUT}s"
echo "[run] Output:  $OUT_BASE/$RUN_SUBDIR"
echo "[run] Video:   RLDX_VIDEO=$RLDX_VIDEO"
echo "============================================"

cd "$REPO_ROOT"

# robocasa_run_explore.sh 在 scripts/ 下
bash "$REPO_ROOT/scripts/robocasa_run_explore.sh" "$TASK" "$GPU" "$SEED" "$TIMEOUT"
