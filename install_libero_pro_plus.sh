#!/usr/bin/env bash
# Install LIBERO-Pro and/or LIBERO-Plus into an existing Python environment.
#
# This is intentionally a repository-root entry point. It keeps the project
# patch path repo-relative and avoids machine-specific defaults where possible.
#
# Usage:
#   bash install_libero_pro_plus.sh
#   bash install_libero_pro_plus.sh --only pro
#   bash install_libero_pro_plus.sh --only plus
#   bash install_libero_pro_plus.sh --venv /path/to/venv
#
# Useful environment overrides:
#   VENV_PY             Python executable in the target env.
#   PYTHON_BIN          Fallback Python executable if VENV_PY is unset.
#   LIBERO_PRO_PATH     Local LIBERO-PRO checkout.
#   LIBERO_PLUS_PATH    Local LIBERO-plus checkout.
#   LIBERO_PRO_HF_DIR   Snapshot with bddl_files/ and init_files/.
#   USE_MIRROR=1        Clone through ghfast.top.

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEFAULT_PATCH="$ROOT_DIR/physicalagent/primitives/workspace_pro/liberopro_register_perturbations.patch"

ONLY=""
PATCH_FILE="${PATCH_FILE:-$DEFAULT_PATCH}"

usage() {
        cat <<'HELP'
Install LIBERO-Pro and/or LIBERO-Plus into an existing Python environment.

Usage:
    bash install_libero_pro_plus.sh
    bash install_libero_pro_plus.sh --only pro
    bash install_libero_pro_plus.sh --only plus
    bash install_libero_pro_plus.sh --venv /path/to/venv

Useful environment overrides:
    VENV_PY             Python executable in the target env.
    PYTHON_BIN          Fallback Python executable if VENV_PY is unset.
    LIBERO_PRO_PATH     Local LIBERO-PRO checkout.
    LIBERO_PLUS_PATH    Local LIBERO-plus checkout.
    LIBERO_PRO_HF_DIR   Snapshot with bddl_files/ and init_files/.
    USE_MIRROR=1        Clone through ghfast.top.

Options:
  --only pro|plus       Install only one package family.
  --venv PATH           Use PATH/bin/python as the target interpreter.
  --python PATH         Use PATH as the target interpreter.
  --pro-path PATH       Clone/reuse LIBERO-PRO at PATH.
  --plus-path PATH      Clone/reuse LIBERO-plus at PATH.
  --hf-dir PATH         LIBERO-Pro HF perturbation snapshot.
  --patch PATH          Patch file for LIBERO-Pro suite registration.
  --mirror              Use the ghfast.top GitHub mirror.
  -h, --help            Show this help.
HELP
}

while [ $# -gt 0 ]; do
    case "$1" in
        --only) ONLY="${2:?missing value for --only}"; shift 2 ;;
        --venv) VENV_PY="${2:?missing value for --venv}/bin/python"; shift 2 ;;
        --python) VENV_PY="${2:?missing value for --python}"; shift 2 ;;
        --pro-path) LIBERO_PRO_PATH="${2:?missing value for --pro-path}"; shift 2 ;;
        --plus-path) LIBERO_PLUS_PATH="${2:?missing value for --plus-path}"; shift 2 ;;
        --hf-dir) LIBERO_PRO_HF_DIR="${2:?missing value for --hf-dir}"; shift 2 ;;
        --patch) PATCH_FILE="${2:?missing value for --patch}"; shift 2 ;;
        --mirror) USE_MIRROR=1; shift ;;
        -h|--help) usage; exit 0 ;;
        *) echo "unknown arg: $1" >&2; usage >&2; exit 1 ;;
    esac
done

if [ -z "${VENV_PY:-}" ]; then
    if [ -n "${PYTHON_BIN:-}" ]; then
        VENV_PY="$PYTHON_BIN"
    else
        VENV_PY="$(command -v python)"
    fi
fi

if [ ! -x "$VENV_PY" ]; then
    echo "[ERROR] Python executable not found or not executable: $VENV_PY" >&2
    echo "        Pass --venv /path/to/venv or set VENV_PY." >&2
    exit 1
fi

VENV_DIR="$($VENV_PY - <<'PY'
import sys
print(sys.prefix)
PY
)"
PIP=("$VENV_PY" -m pip)

LIBERO_PRO_PATH="${LIBERO_PRO_PATH:-$VENV_DIR/libero_pro}"
LIBERO_PLUS_PATH="${LIBERO_PLUS_PATH:-$VENV_DIR/libero_plus}"
LIBERO_PRO_HF_DIR="${LIBERO_PRO_HF_DIR:-${XDG_CACHE_HOME:-$HOME/.cache}/physicalagent/liberopro_hf}"

if [ "${USE_MIRROR:-0}" = "1" ]; then
    GH="https://ghfast.top/https://github.com"
else
    GH="https://github.com"
fi

echo "[install] repo root       = $ROOT_DIR"
echo "[install] VENV_PY         = $VENV_PY"
echo "[install] sys.prefix      = $VENV_DIR"
echo "[install] LIBERO_PRO      = $LIBERO_PRO_PATH"
echo "[install] LIBERO_PLUS     = $LIBERO_PLUS_PATH"
echo "[install] HF snapshot     = $LIBERO_PRO_HF_DIR"
echo "[install] patch           = $PATCH_FILE"
echo "[install] github prefix   = $GH"
[ -n "$ONLY" ] && echo "[install] only            = $ONLY"

clone_or_reuse() {
    local target_dir="$1" repo_url="$2"
    if [ -d "$target_dir/.git" ]; then
        echo "[clone_or_reuse] reusing $target_dir"
    else
        echo "[clone_or_reuse] cloning $repo_url -> $target_dir"
        mkdir -p "$(dirname "$target_dir")"
        git clone "$repo_url" "$target_dir" || {
            echo "[clone_or_reuse] clone failed; retry with --mirror if GitHub TLS is unstable" >&2
            return 1
        }
    fi
}

sync_dir() {
    local src="$1" dst="$2" pattern="$3"
    [ -d "$src" ] || { echo "[sync] missing $src; skipping"; return; }
    [ -d "$dst" ] || { echo "[sync] missing $dst; skipping"; return; }
    local n=0
    for suite_dir in "$src"/*/; do
        [ -d "$suite_dir" ] || continue
        local name
        name="$(basename "$suite_dir")"
        mkdir -p "$dst/$name"
        for f in "$suite_dir"$pattern; do
            [ -f "$f" ] || continue
            cp -f "$f" "$dst/$name/"
            n=$((n + 1))
        done
    done
    echo "[sync] copied $n files from $src/* into $dst/*"
}

site_packages() {
    "$VENV_PY" - <<'PY'
import site
print(site.getsitepackages()[0])
PY
}

install_editable_or_pth() {
    local package_name="$1" source_dir="$2" pth_name="$3"
    echo "[$package_name] pip install -e $source_dir"
    if "${PIP[@]}" install -e "$source_dir" --no-build-isolation; then
        return 0
    fi
    echo "[$package_name] WARN: pip install failed; falling back to .pth"
    local sp
    sp="$(site_packages)"
    echo "$source_dir" > "$sp/$pth_name"
    echo "[$package_name] wrote $sp/$pth_name"
}

install_libero_pro() {
    echo
    echo "================ LIBERO-PRO ================"
    clone_or_reuse "$LIBERO_PRO_PATH" "$GH/RLinf/LIBERO-PRO.git"

    local already_at target_at
    already_at="$($VENV_PY -c "import liberopro, os; print(os.path.realpath(os.path.dirname(liberopro.__file__)))" 2>/dev/null || true)"
    target_at="$(realpath "$LIBERO_PRO_PATH/liberopro" 2>/dev/null || echo "")"
    if [ -n "$already_at" ] && [ "$already_at" = "$target_at" ]; then
        echo "[pro] liberopro already editable at $LIBERO_PRO_PATH"
    else
        install_editable_or_pth pro "$LIBERO_PRO_PATH" liberopro.pth
    fi

    if [ ! -f "$PATCH_FILE" ]; then
        echo "[pro] WARN: patch file missing: $PATCH_FILE"
    else
        pushd "$LIBERO_PRO_PATH" >/dev/null
        if git apply --check "$PATCH_FILE" 2>/dev/null; then
            echo "[pro] applying perturbation registration patch"
            git apply "$PATCH_FILE"
        elif git apply --reverse --check "$PATCH_FILE" 2>/dev/null; then
            echo "[pro] patch already applied"
        else
            echo "[pro] WARN: patch applies neither forward nor reverse; inspect $PATCH_FILE"
        fi
        popd >/dev/null
    fi

    if [ -d "$LIBERO_PRO_HF_DIR" ]; then
        local dest="$LIBERO_PRO_PATH/liberopro/liberopro"
        sync_dir "$LIBERO_PRO_HF_DIR/bddl_files" "$dest/bddl_files" "*.bddl"
        sync_dir "$LIBERO_PRO_HF_DIR/init_files" "$dest/init_files" "*.pruned_init"
    else
        echo "[pro] WARN: HF snapshot dir missing: $LIBERO_PRO_HF_DIR"
        cat <<EOF
[pro] Download it with:
$VENV_PY - <<'PY'
from huggingface_hub import snapshot_download
snapshot_download(
    repo_id='zhouxueyang/LIBERO-Pro', repo_type='dataset',
    local_dir='$LIBERO_PRO_HF_DIR',
    allow_patterns=['bddl_files/**', 'init_files/**'],
)
PY
EOF
    fi

    LIBERO_TYPE=pro "$VENV_PY" - <<'PY'
import os
os.environ.setdefault("LIBERO_TYPE", "pro")
import liberopro.liberopro.benchmark as bench

suites = [
    "libero_spatial_task", "libero_spatial_swap", "libero_spatial_lan",
    "libero_10_task", "libero_10_swap", "libero_10_lan",
    "libero_goal_task", "libero_goal_swap", "libero_goal_lan",
    "libero_object_task", "libero_object_swap", "libero_object_lan",
]
known_empty = {"libero_spatial_swap"}
bad = []
for suite_name in suites:
    try:
        suite = bench.get_benchmark(suite_name)()
        ntrials = len(suite.get_task_init_states(0))
        language = suite.get_task(0).language[:60]
        status = "OK" if ntrials > 0 else "EMPTY_INIT"
        print(f"  {suite_name:25s} t0 trials={ntrials:>3}  {language!r}  [{status}]")
        if ntrials == 0 and suite_name not in known_empty:
            bad.append(suite_name)
    except Exception as exc:
        print(f"  {suite_name:25s} ERROR {type(exc).__name__}: {exc}")
        if suite_name not in known_empty:
            bad.append(suite_name)
if bad:
    raise SystemExit(f"[verify] broken suites, excluding known-empty: {bad}")
print(f"[verify] all probed suites are usable; skipped known-empty: {sorted(known_empty)}")
PY
    echo "[pro] OK"
}

install_libero_plus() {
    echo
    echo "================ LIBERO-PLUS ================"

    if command -v apt-get >/dev/null 2>&1; then
        local packages=(libexpat1 libfontconfig1-dev libpython3-stdlib libmagickwand-dev)
        if dpkg -s "${packages[@]}" >/dev/null 2>&1; then
            echo "[plus] apt deps already installed"
        else
            echo "[plus] installing apt deps: ${packages[*]}"
            if [ "$(id -u)" -eq 0 ]; then
                apt-get update -y && apt-get install -y "${packages[@]}" || \
                    echo "[plus] WARN: apt install failed"
            elif command -v sudo >/dev/null 2>&1; then
                sudo apt-get update -y && sudo apt-get install -y "${packages[@]}" || \
                    echo "[plus] WARN: sudo apt install failed"
            else
                echo "[plus] WARN: apt deps need root/sudo; install manually if needed"
            fi
        fi
    else
        echo "[plus] no apt-get; skipping system deps"
    fi

    clone_or_reuse "$LIBERO_PLUS_PATH" "$GH/sylvestf/LIBERO-plus.git"

    local already_at target_at
    already_at="$($VENV_PY -c "import liberoplus, os; print(os.path.realpath(os.path.dirname(liberoplus.__file__)))" 2>/dev/null || true)"
    target_at="$(realpath "$LIBERO_PLUS_PATH/liberoplus" 2>/dev/null || echo "")"
    if [ -n "$already_at" ] && [ "$already_at" = "$target_at" ]; then
        echo "[plus] liberoplus already editable at $LIBERO_PLUS_PATH"
    else
        if [ -f "$LIBERO_PLUS_PATH/extra_requirements.txt" ]; then
            "${PIP[@]}" install -r "$LIBERO_PLUS_PATH/extra_requirements.txt" --no-build-isolation || \
                echo "[plus] WARN: extra_requirements install failed"
        fi
        install_editable_or_pth plus "$LIBERO_PLUS_PATH" liberoplus.pth
    fi

    "$VENV_PY" - <<'PY'
import importlib
import os

module = importlib.import_module("liberoplus")
print(f"[verify] liberoplus imported from {module.__file__}")
assets = os.path.join(os.path.dirname(module.__file__), "liberoplus", "assets")
if os.path.isdir(assets):
    print(f"[verify] assets dir at {assets} has {sum(1 for _ in os.scandir(assets))} entries")
else:
    print(f"[verify] WARN: assets dir missing at {assets}")
PY
    echo "[plus] OK"
}

case "$ONLY" in
    pro) install_libero_pro ;;
    plus) install_libero_plus ;;
    "") install_libero_pro; install_libero_plus ;;
    *) echo "unknown --only target: $ONLY" >&2; exit 1 ;;
esac

echo
echo "[install] DONE."