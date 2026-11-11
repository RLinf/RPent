#!/usr/bin/env bash
# One shared-workflow simulator rollout against a served Qwen policy.
#
# RPENT_OBSERVATION_MODE (adaptive only) and
# RPENT_PROMPT (simple|base|base_gripper|base_gripper_camera, default simple)
# select the interface; --suite, --task, --seed and --cuda-device come from
# the caller. Extra arguments go to rpent unchanged.
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/env.sh"
source "$EVAL_DIR/planner.sh"
cd "$RPENT_REPO_ROOT"
mode="${RPENT_OBSERVATION_MODE:-adaptive}"
prompt="${RPENT_PROMPT:-simple}"
runtime_args=()
if [[ -n "${RPENT_SAM3_ENDPOINT:-}" ]]; then
  runtime_args+=(--sam3-endpoint "$RPENT_SAM3_ENDPOINT")
fi
output="${RPENT_OUTPUT_DIR:-$RPENT_REPO_ROOT/logs/rl-$mode-$prompt-$(date +%Y%m%d-%H%M%S)}"
exec python -m rpent.cli.main --robot rl \
  --observation-mode "$mode" --prompt "$prompt" \
  "${runtime_args[@]}" \
  "${planner_args[@]}" \
  --image_size "${RPENT_IMAGE_SIZE:-1024}" \
  --max-turns "${RPENT_MAX_TURNS:-100}" \
  --max-episode-steps "${RPENT_MAX_EPISODE_STEPS:-10000}" \
  --planner-timeout-s "${RPENT_PLANNER_TIMEOUT_S:-7200}" \
  --output-dir "$output" "$@"
