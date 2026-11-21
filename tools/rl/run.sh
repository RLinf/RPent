#!/usr/bin/env bash
# 修改下方参数区即可配置单条运行和批处理默认值。
# 优先级：显式命令行参数 > RPENT_* 环境变量 > 下方默认值。
set -euo pipefail
effort="${RPENT_REASONING_EFFORT:-xhigh}"
if [[ -n "${RPENT_MODEL:-}" && "${RPENT_MODEL,,}" != *qwen3.8* && -z "${RPENT_REASONING_EFFORT:-}" ]]; then
  effort=none
fi
run_args=(
  --sam3-timeout-s "${RPENT_SAM3_TIMEOUT_S:-120}"                 # SAM3 排队加推理 RPC 时限，不延长 planner 总预算。
  --workflow "${RPENT_WORKFLOW:-episode}"                         # 工作流：episode、episode_image、pear 或 zap_*。
  --world-cord "${RPENT_WORLD_CORD:-base}"                         # base 为仿真坐标；general 为基座坐标和指垫 TCP。
  --perception-mode "${RPENT_PERCEPTION_MODE:-single}"           # 感知接口：single；batch 使用 observe，仅支持 episode。
  --max-observe-calls "${RPENT_MAX_OBSERVE_CALLS:-8}"               # 一次 observe 中子操作总数上限，至少 1。
  --observe-concurrency "${RPENT_OBSERVE_CONCURRENCY:-8}"           # 独立子操作并发上限；不修改 SAM3 副本的并发限制。
  --action-mode "${RPENT_ACTION_MODE:-single}"                    # 动作接口：single；batch 仅支持 episode/episode_image。
  --max-actions-per-call "${RPENT_MAX_ACTIONS_PER_CALL:-3}"        # batch 每次最多执行的动作数量。
  --reasoning-history "${RPENT_REASONING_HISTORY:--1}"            # 保留最近几轮推理：-1 全部、0 删除、正数 N 轮。
  --image-history "${RPENT_IMAGE_HISTORY:--1}"                    # 保留最近几组图片：双相机算一组，-1 全部、0 无图。
  --image_size "${RPENT_IMAGE_SIZE:-1024}"                         # 发给 VLM 的图片边长：1024、512 或 256。
  --max-turns "${RPENT_MAX_TURNS:-80}"                            # 每条轨迹最多模型请求次数，包括无效回复。
  --max-tokens "${RPENT_MAX_TOKENS:-32768}"                        # 每次回复的总输出上限，包含 reasoning 和工具协议。
  --context-window "${RPENT_CONTEXT_WINDOW:-262144}"              # 服务模型的上下文容量；不会修改服务端启动配置。
  --compaction-fraction "${RPENT_COMPACTION_FRACTION:-0.7}"        # 当前输入超过容量的该比例时触发压缩。
  --compaction-max-messages "${RPENT_COMPACTION_MAX_MESSAGES:-80}" # 消息数（通常每轮2条）超过此值即触发；0 关闭此规则。
  --compaction-keep-messages "${RPENT_COMPACTION_KEEP_MESSAGES:-10}" # 压缩后保留的尾部消息数，另保留开场和完整工具配对。
  --thinking                                                     # 开启本轮思考；改为 --no-thinking 可关闭。
  --reasoning-effort "$effort"                                    # 27B 思考强度：xhigh、medium、low；none 只省略该字段。
  --temperature "${RPENT_TEMPERATURE:-1.0}"                        # 采样温度；27B 官方 thinking 默认 1.0。
  --top-p "${RPENT_TOP_P:-0.95}"                                  # 核采样的累计概率；官方 thinking 默认 0.95。
  --top-k "${RPENT_TOP_K:-20}"                                    # 候选 token 上限；官方默认 20，-1 禁用。
  --min-p "${RPENT_MIN_P:-0.0}"                                   # 相对最高概率的最低概率门槛；官方默认 0。
  --presence-penalty "${RPENT_PRESENCE_PENALTY:-0.0}"               # 已出现 token 的存在惩罚；官方 thinking 默认 0。
  --repetition-penalty "${RPENT_REPETITION_PENALTY:-1.0}"           # 重复惩罚系数；官方默认 1（不额外惩罚）。
  --max-episode-steps "${RPENT_MAX_EPISODE_STEPS:-10000}"           # 每条轨迹最多物理环境步数。
  --planner-timeout-s "${RPENT_PLANNER_TIMEOUT_S:-7200}"            # planner 整条轨迹的墙钟时限，单位秒。
  --observation-mode "${RPENT_OBSERVATION_MODE:-adaptive}"         # 观测接口，当前支持 adaptive。
  --prompt "${RPENT_PROMPT:-simple}"                              # 提示词变体，默认 simple。
)
# 仅打印 argv，不初始化模型或仿真；批处理用它读取同一套默认值。
if [[ "${1:-}" == --print-defaults ]]; then
  printf '%s\n' "${run_args[@]}"
  exit 0
fi
source "$(dirname "${BASH_SOURCE[0]}")/env.sh"
source "$EVAL_DIR/planner.sh"
cd "$RPENT_REPO_ROOT"
runtime_args=()
if [[ -n "${RPENT_SAM3_ENDPOINT:-}" ]]; then
  runtime_args+=(--sam3-endpoint "$RPENT_SAM3_ENDPOINT")
fi
output="${RPENT_OUTPUT_DIR:-$RPENT_REPO_ROOT/logs/rl-$(date +%Y%m%d-%H%M%S)}"
exec python -m rpent.cli.main --robot rl \
  "${planner_args[@]}" "${run_args[@]}" "${runtime_args[@]}" \
  --output-dir "$output" "$@"
