# Sourced by run.sh: the api planner flags for a served Qwen model.
#
# RPENT_MODEL names the served model with the provider prefix, for example
# openai-chat:qwen3.8-27b or openai-chat:qwen3.5-4b, and RPENT_BASE_URL its
# endpoint. The 27B template has reasoning-effort tiers and defaults to xhigh;
# the 4B template has none, so no tier is sent to it and thinking stays on
# through the planner's explicit chat_template_kwargs. Strict tool definitions
# are off because SGLang constrains the whole tool set once any tool is strict.
: "${RPENT_MODEL:?RPENT_MODEL is required, e.g. openai-chat:qwen3.5-4b}"
: "${RPENT_BASE_URL:?RPENT_BASE_URL is required, e.g. http://127.0.0.1:30020/v1}"
export RPENT_PRESENCE_PENALTY="${RPENT_PRESENCE_PENALTY:-0}"
export OPENAI_API_KEY="${OPENAI_API_KEY:-local}"
if [[ -z "${RPENT_REASONING_EFFORT:-}" ]]; then
  case "${RPENT_MODEL,,}" in
    *qwen3.8*) RPENT_REASONING_EFFORT=xhigh ;;
    *) RPENT_REASONING_EFFORT=none ;;
  esac
fi
planner_args=(
  --planner api --model "$RPENT_MODEL" --base-url "$RPENT_BASE_URL"
  --reasoning-effort "$RPENT_REASONING_EFFORT"
  --no-strict-tools
  --max-tokens "${RPENT_MAX_TOKENS:-32768}"
  --context-window "${RPENT_CONTEXT_WINDOW:-262144}"
  --compaction-fraction "${RPENT_COMPACTION_FRACTION:-0.7}"
)
