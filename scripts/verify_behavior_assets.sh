#!/usr/bin/env bash

set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RPENT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
REPRO_ROOT="${RPENT_REPRO_ROOT:-${RPENT_ROOT}/.behavior-runtime}"
RPENT_VENV="${RPENT_VENV:-${REPRO_ROOT}/venvs/rpent}"

: "${OMNIGIBSON_DATA_PATH:?Set OMNIGIBSON_DATA_PATH to the complete BEHAVIOR data root}"
: "${PI05_CHECKPOINT_PATH:?Set PI05_CHECKPOINT_PATH to the downloaded Pi0.5 checkpoint}"

required_directories=(
    "${OMNIGIBSON_DATA_PATH}/behavior-1k-assets/scenes"
    "${OMNIGIBSON_DATA_PATH}/omnigibson-robot-assets"
    "${OMNIGIBSON_DATA_PATH}/2025-challenge-task-instances"
)
required_files=(
    "${OMNIGIBSON_DATA_PATH}/omnigibson.key"
    "${PI05_CHECKPOINT_PATH}/model.safetensors"
    "${PI05_CHECKPOINT_PATH}/assets/behavior-1k/2025-challenge-demos/norm_stats.json"
)

for path in "${required_directories[@]}"; do
    if [[ ! -d "${path}" ]]; then
        echo "Missing required directory: ${path}" >&2
        exit 1
    fi
done
for path in "${required_files[@]}"; do
    if [[ ! -f "${path}" ]]; then
        echo "Missing required file: ${path}" >&2
        exit 1
    fi
done

if [[ ! -x "${RPENT_VENV}/bin/python" ]]; then
    echo "Missing RPent Python: ${RPENT_VENV}/bin/python" >&2
    exit 1
fi
cd "${RPENT_ROOT}"
"${RPENT_VENV}/bin/python" - "${PI05_CHECKPOINT_PATH}" <<'PY'
from pathlib import Path
import sys

from robots.behavior.policy_checkpoint import validate_policy_checkpoint

checkpoint = Path(sys.argv[1]).resolve()
validate_policy_checkpoint(checkpoint)
print(f"Policy checkpoint contract: OK ({checkpoint})")
PY

echo "BEHAVIOR assets: OK"
