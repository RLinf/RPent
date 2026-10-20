#!/usr/bin/env bash

set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RPENT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
REPRO_ROOT="${RPENT_REPRO_ROOT:-${RPENT_ROOT}/.behavior-runtime}"
RLINF_ROOT="${RLINF_ROOT:-${REPRO_ROOT}/RLinf}"
RPENT_VENV="${RPENT_VENV:-${REPRO_ROOT}/venvs/rpent}"
BEHAVIOR_VENV="${BEHAVIOR_VENV:-${REPRO_ROOT}/venvs/behavior}"
LOG_DIR="${LOG_DIR:-${REPRO_ROOT}/logs/install}"
TOOLS_DIR="${TOOLS_DIR:-${REPRO_ROOT}/tools}"
export DOWNLOAD_DIR="${DOWNLOAD_DIR:-${REPRO_ROOT}/downloads}"
UV_VERSION="${UV_VERSION:-0.12.7}"
PYTHON_VERSION="${PYTHON_VERSION:-3.10}"
RLINF_REPO_URL="${RLINF_REPO_URL:-https://github.com/RLinf/RLinf.git}"
RLINF_COMMIT="${RLINF_COMMIT:-91a302ec50ddacd1fdabc5859b85e3661d768236}"

mkdir -p "${LOG_DIR}" "${TOOLS_DIR}" "${DOWNLOAD_DIR}" "$(dirname "${RPENT_VENV}")"
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
echo "BEHAVIOR source plugin: ${RPENT_ROOT}/robots/behavior"

UV_BIN="${TOOLS_DIR}/uv"
UV_VERSION_OUTPUT="$("${UV_BIN}" --version 2>/dev/null || true)"
if [[ ! -x "${UV_BIN}" ]] || {
    [[ "${UV_VERSION_OUTPUT}" != "uv ${UV_VERSION}" ]] &&
        [[ "${UV_VERSION_OUTPUT}" != "uv ${UV_VERSION} "* ]]
}; then
    UV_INSTALLER="${TOOLS_DIR}/uv-install-${UV_VERSION}.sh"
    curl -LsSf "https://astral.sh/uv/${UV_VERSION}/install.sh" -o "${UV_INSTALLER}"
    env UV_UNMANAGED_INSTALL="${TOOLS_DIR}" sh "${UV_INSTALLER}"
fi
UV_VERSION_OUTPUT="$("${UV_BIN}" --version)"
if [[ "${UV_VERSION_OUTPUT}" != "uv ${UV_VERSION}" ]] &&
    [[ "${UV_VERSION_OUTPUT}" != "uv ${UV_VERSION} "* ]]; then
    echo "Expected uv ${UV_VERSION}, got ${UV_VERSION_OUTPUT}." >&2
    exit 1
fi
export PATH="${TOOLS_DIR}:${PATH}"

if [[ ! -e "${RLINF_ROOT}/.git" ]]; then
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
# This installer uses the same checkout for the planner and simulator sidecars.
# Keep the planner venv editable; simulator sidecars run from this checkout
# without pulling planner dependencies into their separate venv.
"${UV_BIN}" pip install --python "${RPENT_VENV}/bin/python" -e "${RPENT_ROOT}[behavior]"
"${UV_BIN}" pip check --python "${RPENT_VENV}/bin/python"

export UV_TORCH_BACKEND=cu124
RLINF_INSTALL_ARGS=()
if [[ "${RLINF_NO_ROOT:-0}" == 1 ]]; then
    RLINF_INSTALL_ARGS+=(--no-root)
fi
if [[ "${RLINF_USE_MIRROR:-0}" == 1 ]]; then
    RLINF_INSTALL_ARGS+=(--use-mirror)
fi
(
    cd "${RLINF_ROOT}"
    bash requirements/install.sh embodied \
        --model openpi \
        --env behavior \
        --python "${PYTHON_VERSION}" \
        --venv "${BEHAVIOR_VENV}" \
        --install-rlinf \
        --no-flash-attn \
        "${RLINF_INSTALL_ARGS[@]}"
)

BEHAVIOR_PYTHON="${BEHAVIOR_VENV}/bin/python"
if [[ ! -x "${BEHAVIOR_PYTHON}" ]]; then
    echo "RLinf installer did not create ${BEHAVIOR_PYTHON}." >&2
    exit 1
fi

# ENV, VLA, and DINO run from this source checkout. Installing RPent here would
# pull planner dependencies into the simulator environment and conflict with its pins.

MANIFEST_DIR="${REPRO_ROOT}/manifests"
mkdir -p "${MANIFEST_DIR}"
RLINF_FREEZE="${MANIFEST_DIR}/rlinf-venv.freeze.txt"
"${UV_BIN}" pip freeze --exclude-editable --python "${BEHAVIOR_PYTHON}" > "${RLINF_FREEZE}"
RLINF_PIP_CHECK="${MANIFEST_DIR}/rlinf-pip-check.txt"
RLINF_METADATA_CONFLICT=0
KNOWN_RLINF_MISMATCH='The package `rlinf-openpi` requires `torch==2.7.1`, but `2.5.1+cu124` is installed'
KNOWN_CHECK_RESULT="$(printf 'Found 1 incompatibility\n%s' "${KNOWN_RLINF_MISMATCH}")"
if ! "${UV_BIN}" pip check --python "${BEHAVIOR_PYTHON}" > "${RLINF_PIP_CHECK}" 2>&1; then
    CHECK_RESULT="$(sed -E '/^Using Python .* environment at: /d; /^Checked [0-9]+ packages in /d' "${RLINF_PIP_CHECK}")"
    if [[ "${CHECK_RESULT}" != "${KNOWN_CHECK_RESULT}" ]]; then
        cat "${RLINF_PIP_CHECK}" >&2
        echo "Unexpected dependency incompatibility in the RLinf recipe." >&2
        exit 1
    fi
    RLINF_METADATA_CONFLICT=1
fi
cat "${RLINF_PIP_CHECK}"

# Resolve cuRobo's real dependencies without changing RLinf's installed versions.
CUROBO_COMMIT=4ea77366ca48ee453e7df139e39fa6532af49f3b # v0.8.0
CUROBO_CONSTRAINTS="${MANIFEST_DIR}/rlinf-installed-constraints.txt"
awk -F '==' 'NF == 2 && $1 ~ /^[A-Za-z0-9_.-]+$/ {print}' \
    "${RLINF_FREEZE}" > "${CUROBO_CONSTRAINTS}"
"${UV_BIN}" pip install --python "${BEHAVIOR_PYTHON}" \
    --constraint "${CUROBO_CONSTRAINTS}" \
    "nvidia-curobo[cu12-torch] @ git+https://github.com/NVlabs/curobo.git@${CUROBO_COMMIT}"

"${BEHAVIOR_PYTHON}" - "${CUROBO_CONSTRAINTS}" <<'PY'
from importlib.metadata import version
from pathlib import Path
import sys

for line in Path(sys.argv[1]).read_text().splitlines():
    package, expected = line.split("==", 1)
    actual = version(package)
    if actual != expected:
        raise RuntimeError(f"RLinf package changed while adding cuRobo: {package} {expected} -> {actual}")
PY

# RLinf owns OpenPI's Transformers replacement; verify rather than recopy it.
SITE_PACKAGES="$("${BEHAVIOR_PYTHON}" -c 'import site; print(site.getsitepackages()[0])')"
TRANSFORMERS_REPLACE="${SITE_PACKAGES}/openpi/models_pytorch/transformers_replace"
if [[ ! -d "${TRANSFORMERS_REPLACE}" ]]; then
    echo "Missing OpenPI transformer replacement directory: ${TRANSFORMERS_REPLACE}" >&2
    exit 1
fi

"${BEHAVIOR_PYTHON}" - <<'PY'
from importlib.metadata import version
from pathlib import Path
import hashlib
import site

if version("rlinf-openpi") != "0.1.1" or not version("torch").startswith("2.5.1"):
    raise RuntimeError("RLinf BEHAVIOR/OpenPI versions differ from its pinned recipe")

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
import curobo

if not torch.cuda.is_available():
    raise RuntimeError("torch.cuda.is_available() is false")
tensor = torch.ones(1, device="cuda")
print("CUDA smoke:", tensor, torch.cuda.get_device_name(0))
print("Critical imports and transformer replacement: OK")
PY

(
    cd "${RPENT_ROOT}"
    "${BEHAVIOR_PYTHON}" -c \
        'import robots.behavior.env_server, robots.behavior.vla_server, robots.behavior.dino_v2.server'
)

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

echo "Checking simulator dependency metadata."
PIP_CHECK_REPORT="${MANIFEST_DIR}/behavior-pip-check.txt"
if ! "${UV_BIN}" pip check --python "${BEHAVIOR_PYTHON}" > "${PIP_CHECK_REPORT}" 2>&1; then
    CHECK_RESULT="$(sed -E '/^Using Python .* environment at: /d; /^Checked [0-9]+ packages in /d' "${PIP_CHECK_REPORT}")"
    if [[ "${RLINF_METADATA_CONFLICT}" != 1 ]] || [[ "${CHECK_RESULT}" != "${KNOWN_CHECK_RESULT}" ]]; then
        cat "${PIP_CHECK_REPORT}" >&2
        echo "Unexpected simulator dependency incompatibility." >&2
        exit 1
    fi
fi
cat "${PIP_CHECK_REPORT}"
echo "RLinf and final dependency reports: ${RLINF_PIP_CHECK}, ${PIP_CHECK_REPORT}"

cd "${RPENT_ROOT}"
"${RPENT_VENV}/bin/python" -m robots.behavior.selfcheck

if [[ "${RLINF_METADATA_CONFLICT}" == 1 ]]; then
    echo "Runtime installed; uv pip check still reports the known RLinf/OpenPI metadata conflict."
else
    echo "Installation complete; dependency metadata check passed."
fi
echo "Behavior Python: ${BEHAVIOR_PYTHON}"
echo "Version manifests: ${MANIFEST_DIR}"
echo "Next: export the asset variables and run:"
echo "  cd ${RPENT_ROOT} && ${RPENT_VENV}/bin/python -m robots.behavior.assets_cli --verify"
