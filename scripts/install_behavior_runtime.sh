#!/usr/bin/env bash

set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RPENT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
REPRO_ROOT="${RPENT_REPRO_ROOT:-${RPENT_ROOT}/.behavior-runtime}"
RLINF_ROOT="${RLINF_ROOT:-${REPRO_ROOT}/RLinf}"
RPENT_VENV="${RPENT_VENV:-${REPRO_ROOT}/venvs/rpent}"
BEHAVIOR_VENV="${BEHAVIOR_VENV:-${REPRO_ROOT}/venvs/behavior}"
LOG_DIR="${LOG_DIR:-${REPRO_ROOT}/logs/install}"
TOOLS_DIR="${TOOLS_DIR:-${REPRO_ROOT}/tools}"
UV_VERSION="${UV_VERSION:-0.12.7}"
PYTHON_VERSION="${PYTHON_VERSION:-3.10}"
RLINF_REPO_URL="${RLINF_REPO_URL:-https://github.com/RLinf/RLinf.git}"
RLINF_COMMIT="${RLINF_COMMIT:-dd92c62857da4c67aa5e7c36f731c0d6a121f6d7}"

mkdir -p "${LOG_DIR}" "${TOOLS_DIR}" "$(dirname "${RPENT_VENV}")"
LOG_FILE="${LOG_DIR}/install-$(date -u +%Y%m%dT%H%M%SZ).log"
exec > >(tee -a "${LOG_FILE}") 2>&1

trap 'echo "ERROR: command failed at line ${LINENO}. See ${LOG_FILE}" >&2' ERR

require_command() {
    if ! command -v "$1" >/dev/null 2>&1; then
        echo "Missing required host command: $1" >&2
        exit 1
    fi
}

for command_name in bash curl git nvidia-smi sha256sum; do
    require_command "${command_name}"
done

echo "RPent root: ${RPENT_ROOT}"
echo "RLinf root: ${RLINF_ROOT}"
echo "RPent venv: ${RPENT_VENV}"
echo "BEHAVIOR venv: ${BEHAVIOR_VENV}"
echo "Install log: ${LOG_FILE}"

UV_BIN="${TOOLS_DIR}/uv"
if [[ ! -x "${UV_BIN}" ]] || [[ "$("${UV_BIN}" --version 2>/dev/null || true)" != "uv ${UV_VERSION}" ]]; then
    UV_INSTALLER="${TOOLS_DIR}/uv-install-${UV_VERSION}.sh"
    curl -LsSf "https://astral.sh/uv/${UV_VERSION}/install.sh" -o "${UV_INSTALLER}"
    env UV_UNMANAGED_INSTALL="${TOOLS_DIR}" sh "${UV_INSTALLER}"
fi
if [[ "$("${UV_BIN}" --version)" != "uv ${UV_VERSION}" ]]; then
    echo "Expected uv ${UV_VERSION}, got $("${UV_BIN}" --version)." >&2
    exit 1
fi
export PATH="${TOOLS_DIR}:${PATH}"

if [[ ! -d "${RLINF_ROOT}/.git" ]]; then
    mkdir -p "$(dirname "${RLINF_ROOT}")"
    git clone "${RLINF_REPO_URL}" "${RLINF_ROOT}"
    git -C "${RLINF_ROOT}" checkout --detach "${RLINF_COMMIT}"
else
    ACTUAL_RLINF_COMMIT="$(git -C "${RLINF_ROOT}" rev-parse HEAD)"
    if [[ "${ACTUAL_RLINF_COMMIT}" != "${RLINF_COMMIT}" ]]; then
        echo "Existing RLinf checkout is ${ACTUAL_RLINF_COMMIT}, expected ${RLINF_COMMIT}." >&2
        echo "Use a new RLINF_ROOT; this script will not overwrite a checkout." >&2
        exit 1
    fi
    if [[ -n "$(git -C "${RLINF_ROOT}" status --porcelain)" ]]; then
        echo "Existing RLinf checkout is dirty; refusing to run its installer." >&2
        exit 1
    fi
fi

if [[ ! -x "${RPENT_VENV}/bin/python" ]]; then
    "${UV_BIN}" venv --python "${PYTHON_VERSION}" "${RPENT_VENV}"
fi
"${UV_BIN}" pip install --python "${RPENT_VENV}/bin/python" -e "${RPENT_ROOT}"

export UV_TORCH_BACKEND=cu124
bash "${RLINF_ROOT}/requirements/install.sh" embodied \
    --model openpi \
    --env behavior \
    --venv "${BEHAVIOR_VENV}" \
    --install-rlinf \
    --no-flash-attn \
    --no-root

BEHAVIOR_PYTHON="${BEHAVIOR_VENV}/bin/python"
if [[ ! -x "${BEHAVIOR_PYTHON}" ]]; then
    echo "RLinf installer did not create ${BEHAVIOR_PYTHON}." >&2
    exit 1
fi

# Install RPent and all HTTP sidecar dependencies before the final compatibility pins.
# Planner packages in this environment are metadata-only for the sidecars; the
# separate RPent venv remains the canonical planner and Dashboard environment.
"${UV_BIN}" pip install --python "${BEHAVIOR_PYTHON}" -e "${RPENT_ROOT}"

# The official BEHAVIOR runtime is validated against CUDA 12.4 and torch 2.5.1.
"${UV_BIN}" pip install --python "${BEHAVIOR_PYTHON}" --reinstall \
    --index-url https://download.pytorch.org/whl/cu124 \
    'torch==2.5.1+cu124' \
    'torchvision==0.20.1+cu124' \
    'torchaudio==2.5.1+cu124'

# Final compatibility repin. This intentionally runs after every dependency installer.
# --no-deps prevents a late resolver pass from silently changing Isaac/OpenPI versions.
FINAL_PINS=(
    'numpy==1.26.4'
    'protobuf==6.33.0'
    'ml-dtypes==0.5.3'
    'click==8.2.1'
    'llvmlite==0.48.0'
    'numba==0.66.0'
    'fastapi==0.110.0'
    'starlette==0.36.3'
    'pydantic==2.9.2'
    'pydantic-core==2.23.4'
    'uvicorn==0.52.4'
    'ray==2.55.1'
    'tensorflow-metadata==1.21.0'
    'typeguard==4.5.2'
    'gdown==6.1.0'
    'pymunk==7.3.0'
    'zarr==3.0.0a5'
    'google-api-core==2.30.3'
    'googleapis-common-protos==1.75.0'
    'proto-plus==1.28.0'
    'beautifulsoup4==4.15.0'
    'soupsieve==2.8.4'
    'asciitree==0.3.3'
    'crc32c==2.8'
    'donfig==0.8.1.post1'
    'numcodecs==0.13.1'
    'torchcodec==0.2.0'
    'rlinf-openpi==0.1.1'
    'rlinf-transformer-openpi==4.53.2'
    'lerobot==0.3.3'
    'openpi-client==0.1.2'
)
"${UV_BIN}" pip install --python "${BEHAVIOR_PYTHON}" --no-deps "${FINAL_PINS[@]}"

# OpenPI's replacement files must match the final transformers build. No package
# installation is allowed after this block.
"${UV_BIN}" pip install --python "${BEHAVIOR_PYTHON}" --no-deps \
    'transformers==4.53.2' 'tokenizers==0.21.4' 'huggingface-hub==0.36.2'
SITE_PACKAGES="$("${BEHAVIOR_PYTHON}" -c 'import site; print(site.getsitepackages()[0])')"
TRANSFORMERS_REPLACE="${SITE_PACKAGES}/openpi/models_pytorch/transformers_replace"
if [[ ! -d "${TRANSFORMERS_REPLACE}" ]]; then
    echo "Missing OpenPI transformer replacement directory: ${TRANSFORMERS_REPLACE}" >&2
    exit 1
fi
cp -a "${TRANSFORMERS_REPLACE}/." "${SITE_PACKAGES}/transformers/"

"${BEHAVIOR_PYTHON}" - <<'PY'
from importlib.metadata import version
from pathlib import Path
import hashlib
import site

expected = {
    "torch": "2.5.1+cu124",
    "torchvision": "0.20.1+cu124",
    "torchaudio": "2.5.1+cu124",
    "numpy": "1.26.4",
    "fastapi": "0.110.0",
    "starlette": "0.36.3",
    "pydantic": "2.9.2",
    "transformers": "4.53.2",
    "tokenizers": "0.21.4",
    "rlinf-openpi": "0.1.1",
    "rlinf-transformer-openpi": "4.53.2",
    "lerobot": "0.3.3",
}
for package, wanted in expected.items():
    actual = version(package)
    if actual != wanted:
        raise RuntimeError(f"{package}: expected {wanted}, got {actual}")

root = Path(site.getsitepackages()[0])
replacement = root / "openpi/models_pytorch/transformers_replace"
target = root / "transformers"
files = sorted(path.relative_to(replacement) for path in replacement.rglob("*.py"))
if not files:
    raise RuntimeError("OpenPI transformer replacement contains no Python files")
for relative in files:
    source_hash = hashlib.sha256((replacement / relative).read_bytes()).digest()
    target_hash = hashlib.sha256((target / relative).read_bytes()).digest()
    if source_hash != target_hash:
        raise RuntimeError(f"transformer replacement mismatch: {relative}")

import torch
import omnigibson
import ray
import rlinf
import openpi

if not torch.cuda.is_available():
    raise RuntimeError("torch.cuda.is_available() is false")
tensor = torch.ones(1, device="cuda")
print("CUDA smoke:", tensor, torch.cuda.get_device_name(0))
print("Critical imports and transformer replacement: OK")
PY

MANIFEST_DIR="${REPRO_ROOT}/manifests"
mkdir -p "${MANIFEST_DIR}"
"${UV_BIN}" pip freeze --python "${RPENT_VENV}/bin/python" \
    > "${MANIFEST_DIR}/rpent-venv.freeze.txt"
"${UV_BIN}" pip freeze --python "${BEHAVIOR_PYTHON}" \
    > "${MANIFEST_DIR}/behavior-venv.freeze.txt"
{
    echo "generated_at_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
    echo "rpent_commit=$(git -C "${RPENT_ROOT}" rev-parse HEAD)"
    echo "rpent_dirty=$(test -n "$(git -C "${RPENT_ROOT}" status --porcelain)" && echo true || echo false)"
    echo "rlinf_commit=$(git -C "${RLINF_ROOT}" rev-parse HEAD)"
    echo "uv_version=$("${UV_BIN}" --version)"
    echo "python_version=$("${BEHAVIOR_PYTHON}" --version 2>&1)"
} > "${MANIFEST_DIR}/source-versions.txt"

echo "Running uv dependency metadata check (report-only for upstream pin conflicts)."
"${UV_BIN}" pip check --python "${BEHAVIOR_PYTHON}" \
    > "${MANIFEST_DIR}/behavior-pip-check.txt" 2>&1 || true
cat "${MANIFEST_DIR}/behavior-pip-check.txt"

cd "${RPENT_ROOT}"
"${RPENT_VENV}/bin/python" -m robots.behavior.selfcheck

echo "Installation complete."
echo "Behavior Python: ${BEHAVIOR_PYTHON}"
echo "Version manifests: ${MANIFEST_DIR}"
echo "Next: export the asset variables and run scripts/verify_behavior_assets.sh"
