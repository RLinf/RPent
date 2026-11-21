# Sourced by run.sh: model/endpoint selection. Hyperparameters live in run.sh.
: "${RPENT_MODEL:?RPENT_MODEL is required, e.g. openai-chat:qwen3.8-27b}"
: "${RPENT_BASE_URL:?RPENT_BASE_URL is required, e.g. http://localhost:30000/v1}"
export OPENAI_API_KEY="${OPENAI_API_KEY:-local}"
planner_args=(
  --planner api --model "$RPENT_MODEL" --base-url "$RPENT_BASE_URL"
  --no-strict-tools
)
