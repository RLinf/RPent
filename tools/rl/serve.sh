#!/usr/bin/env bash
# Launch a local SGLang policy; SAM3 uses its existing Python module.
set -euo pipefail
: "${MODEL_PATH:?Set MODEL_PATH to the local checkpoint directory}"
python="${RPENT_SGLANG_PYTHON:-python}"
export CUDA_VISIBLE_DEVICES="${SERVE_GPUS:-2,3}"
IFS=, read -r -a devices <<< "$CUDA_VISIBLE_DEVICES"
args=(--model-path "$MODEL_PATH" --served-model-name "${MODEL_NAME:-qwen3.8-27b}"
  --tp "${#devices[@]}" --host 127.0.0.1 --port "${SERVE_PORT:-30080}"
  --context-length 262144 --mem-fraction-static "${SERVE_MEM_FRACTION:-0.8}"
  --max-running-requests "${SERVE_CONCURRENCY:-8}" --tool-call-parser qwen3_coder --reasoning-parser qwen3
  --enable-cache-report)
if [[ ${#devices[@]} -gt 1 ]]; then args+=(--disable-custom-all-reduce); fi
if [[ ${ENABLE_MTP:-1} == 1 ]]; then
  args+=(--speculative-algorithm EAGLE --speculative-num-steps 3 --speculative-eagle-topk 1 --speculative-num-draft-tokens 4)
fi
exec "$python" -m sglang.launch_server "${args[@]}" "$@"
