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
#
# Useful environment overrides:
#   LIBERO_PRO_PATH     Local LIBERO-PRO checkout.
#   LIBERO_PLUS_PATH    Local LIBERO-plus checkout.
#   LIBERO_PRO_HF_DIR   Snapshot with bddl_files/ and init_files/.
#   LIBERO_PRO_CONFIG_PATH   LIBERO_CONFIG_PATH to write/use for LIBERO-Pro.
#   LIBERO_PLUS_CONFIG_PATH  LIBERO_CONFIG_PATH to write/use for LIBERO-Plus.
#   USE_MIRROR=1        Clone through ghfast.top.

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

ONLY=""
PATCH_FILE="${PATCH_FILE:-"$ROOT_DIR/liberopro_register_perturbations.patch"}"

usage() {
        cat <<'HELP'
Install LIBERO-Pro and/or LIBERO-Plus into an existing Python environment.

Usage:
    bash install_libero_pro_plus.sh
    bash install_libero_pro_plus.sh --only pro
    bash install_libero_pro_plus.sh --only plus

Useful environment overrides:
    LIBERO_PRO_PATH     Local LIBERO-PRO checkout.
    LIBERO_PLUS_PATH    Local LIBERO-plus checkout.
    LIBERO_PRO_HF_DIR   Snapshot with bddl_files/ and init_files/.
    LIBERO_PRO_CONFIG_PATH   LIBERO_CONFIG_PATH to write/use for LIBERO-Pro.
    LIBERO_PLUS_CONFIG_PATH  LIBERO_CONFIG_PATH to write/use for LIBERO-Plus.
    USE_MIRROR=1        Clone through ghfast.top.

Options:
  --only pro|plus       Install only one package family.
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
        --pro-path) LIBERO_PRO_PATH="${2:?missing value for --pro-path}"; shift 2 ;;
        --plus-path) LIBERO_PLUS_PATH="${2:?missing value for --plus-path}"; shift 2 ;;
        --hf-dir) LIBERO_PRO_HF_DIR="${2:?missing value for --hf-dir}"; shift 2 ;;
        --patch) PATCH_FILE="${2:?missing value for --patch}"; shift 2 ;;
        --mirror) USE_MIRROR=1; shift ;;
        -h|--help) usage; exit 0 ;;
        *) echo "unknown arg: $1" >&2; usage >&2; exit 1 ;;
    esac
done

PYTHON="$(command -v python || true)"
if [ -z "$PYTHON" ] || [ ! -x "$PYTHON" ]; then
    echo "[ERROR] python not found on PATH; activate the target environment first." >&2
    exit 1
fi

VENV_DIR="$($PYTHON - <<'PY'
import sys
print(sys.prefix)
PY
)"
PIP=("$PYTHON" -m pip)

LIBERO_PRO_PATH="${LIBERO_PRO_PATH:-$VENV_DIR/libero_pro}"
LIBERO_PLUS_PATH="${LIBERO_PLUS_PATH:-$VENV_DIR/libero_plus}"
LIBERO_PRO_HF_DIR="${LIBERO_PRO_HF_DIR:-${XDG_CACHE_HOME:-$HOME/.cache}/rpent/liberopro_hf}"
LIBERO_PRO_CONFIG_PATH="${LIBERO_PRO_CONFIG_PATH:-${LIBERO_CONFIG_PATH:-$HOME/.liberopro}}"
if [ -z "${LIBERO_PLUS_CONFIG_PATH:-}" ]; then
    if [ "$ONLY" = "plus" ] && [ -n "${LIBERO_CONFIG_PATH:-}" ]; then
        LIBERO_PLUS_CONFIG_PATH="$LIBERO_CONFIG_PATH"
    else
        LIBERO_PLUS_CONFIG_PATH="$HOME/.libero"
    fi
fi

if [ "${USE_MIRROR:-0}" = "1" ]; then
    GH="https://ghfast.top/https://github.com"
else
    GH="https://github.com"
fi

echo "[install] repo root       = $ROOT_DIR"
echo "[install] python          = $PYTHON"
echo "[install] sys.prefix      = $VENV_DIR"
echo "[install] LIBERO_PRO      = $LIBERO_PRO_PATH"
echo "[install] LIBERO_PLUS     = $LIBERO_PLUS_PATH"
echo "[install] HF snapshot     = $LIBERO_PRO_HF_DIR"
echo "[install] PRO config      = $LIBERO_PRO_CONFIG_PATH"
echo "[install] PLUS config     = $LIBERO_PLUS_CONFIG_PATH"
echo "[install] patch           = $PATCH_FILE"
echo "[install] github prefix   = $GH"
[ -n "$ONLY" ] && echo "[install] only            = $ONLY"

python_clean() {
    PYTHONNOUSERSITE=1 PYTHONPATH= "$PYTHON" "$@"
}

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
    python_clean - <<'PY'
import site
print(site.getsitepackages()[0])
PY
}

write_pth() {
    local package_name="$1" source_dir="$2" pth_name="$3"
    local sp
    sp="$(site_packages)"
    echo "$source_dir" > "$sp/$pth_name"
    echo "[$package_name] wrote $sp/$pth_name"
}

write_libero_config() {
    local label="$1" config_dir="$2" benchmark_root="$3"
    benchmark_root="$(realpath "$benchmark_root")"
    mkdir -p "$config_dir"
    cat > "$config_dir/config.yaml" <<EOF
assets: $benchmark_root/./assets
bddl_files: $benchmark_root/./bddl_files
benchmark_root: $benchmark_root
datasets: $benchmark_root/../datasets
init_states: $benchmark_root/./init_files
EOF
    echo "[$label] wrote LIBERO config: $config_dir/config.yaml"
}

install_editable_or_pth() {
    local package_name="$1" source_dir="$2" pth_name="$3"
    echo "[$package_name] pip install -e $source_dir"
    if "${PIP[@]}" install -e "$source_dir" --no-build-isolation; then
        return 0
    fi
    echo "[$package_name] WARN: pip install failed; falling back to .pth"
    write_pth "$package_name" "$source_dir" "$pth_name"
}

install_libero_pro() {
    echo
    echo "================ LIBERO-PRO ================"
    clone_or_reuse "$LIBERO_PRO_PATH" "$GH/RLinf/LIBERO-PRO.git"

    local already_at target_at
    already_at="$(python_clean -c "import liberopro, os; print(os.path.realpath(os.path.dirname(liberopro.__file__)))" 2>/dev/null || true)"
    target_at="$(realpath "$LIBERO_PRO_PATH/liberopro" 2>/dev/null || echo "")"
    if [ -n "$already_at" ] && [ "$already_at" = "$target_at" ]; then
        echo "[pro] liberopro already editable at $LIBERO_PRO_PATH"
    else
        install_editable_or_pth pro "$LIBERO_PRO_PATH" liberopro.pth
    fi

    if [ ! -f "$PATCH_FILE" ]; then
        echo "[pro] patch file missing: $PATCH_FILE" >&2
        return 1
    fi

    pushd "$LIBERO_PRO_PATH" >/dev/null
    if git apply --check "$PATCH_FILE" 2>/dev/null; then
        git apply "$PATCH_FILE"
        echo "[pro] applied patch: $PATCH_FILE"
    elif git apply --reverse --check "$PATCH_FILE" 2>/dev/null; then
        echo "[pro] patch already applied: $PATCH_FILE"
    else
        echo "[pro] patch applies neither forward nor reverse; use a clean LIBERO-PRO checkout or repair the existing install manually" >&2
        popd >/dev/null
        return 1
    fi
    popd >/dev/null

    local dest="$LIBERO_PRO_PATH/liberopro/liberopro"
    write_libero_config pro "$LIBERO_PRO_CONFIG_PATH" "$dest"

    if ! python_clean -c "import huggingface_hub" >/dev/null 2>&1; then
        echo "[pro] installing Python package: huggingface_hub"
        "${PIP[@]}" install huggingface_hub
    fi
    echo "[pro] downloading HF perturbation assets to $LIBERO_PRO_HF_DIR"
    python_clean - "$LIBERO_PRO_HF_DIR" <<'PY'
from pathlib import Path
import sys

from huggingface_hub import snapshot_download

local_dir = Path(sys.argv[1]).expanduser()
snapshot_download(
    repo_id="zhouxueyang/LIBERO-Pro",
    repo_type="dataset",
    local_dir=str(local_dir),
    allow_patterns=["bddl_files/**", "init_files/**"],
)
PY
    local pro_perturbation_suites=(
        "libero_spatial_task" "libero_spatial_swap" "libero_spatial_lan"
        "libero_10_task" "libero_10_swap" "libero_10_lan"
        "libero_goal_task" "libero_goal_swap" "libero_goal_lan"
        "libero_object_task" "libero_object_swap" "libero_object_lan"
    )
    local suite_name
    for suite_name in "${pro_perturbation_suites[@]}"; do
        if ! compgen -G "$LIBERO_PRO_HF_DIR/bddl_files/$suite_name/*.bddl" >/dev/null; then
            echo "[pro] downloaded HF snapshot but missing bddl_files/$suite_name/*.bddl" >&2
            return 1
        fi
        if ! compgen -G "$LIBERO_PRO_HF_DIR/init_files/$suite_name/*.pruned_init" >/dev/null; then
            echo "[pro] downloaded HF snapshot but missing init_files/$suite_name/*.pruned_init" >&2
            return 1
        fi
    done
    sync_dir "$LIBERO_PRO_HF_DIR/bddl_files" "$dest/bddl_files" "*.bddl"
    sync_dir "$LIBERO_PRO_HF_DIR/init_files" "$dest/init_files" "*.pruned_init"

    PYTHONNOUSERSITE=1 PYTHONPATH= LIBERO_CONFIG_PATH="$LIBERO_PRO_CONFIG_PATH" LIBERO_TYPE=pro "$PYTHON" - "${pro_perturbation_suites[@]}" <<'PY'
import os
import sys
os.environ.setdefault("LIBERO_TYPE", "pro")
import liberopro.liberopro.benchmark as bench

suites = sys.argv[1:]
bad = []
for suite_name in suites:
    try:
        suite = bench.get_benchmark(suite_name)()
        ntrials = len(suite.get_task_init_states(0))
        language = suite.get_task(0).language[:60]
        status = "OK" if ntrials > 0 else "EMPTY_INIT"
        print(f"  {suite_name:25s} t0 trials={ntrials:>3}  {language!r}  [{status}]")
        if ntrials == 0:
            bad.append(suite_name)
    except Exception as exc:
        print(f"  {suite_name:25s} ERROR {type(exc).__name__}: {exc}")
        bad.append(suite_name)
if bad:
    raise SystemExit(f"[verify] broken suites: {bad}")
print("[verify] all probed suites are usable")
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

    local already_at target_at plus_pythonpath plus_package_dir
    plus_pythonpath="$LIBERO_PLUS_PATH/libero"
    plus_package_dir="$plus_pythonpath/libero"
    already_at="$(python_clean -c "import libero, os; print(os.path.realpath(os.path.dirname(libero.__file__)))" 2>/dev/null || true)"
    target_at="$(realpath "$plus_package_dir" 2>/dev/null || echo "")"
    if [ -n "$already_at" ] && [ "$already_at" = "$target_at" ]; then
        echo "[plus] libero already editable at $LIBERO_PLUS_PATH"
    else
        if [ -f "$LIBERO_PLUS_PATH/extra_requirements.txt" ]; then
            "${PIP[@]}" install -r "$LIBERO_PLUS_PATH/extra_requirements.txt" --no-build-isolation || \
                echo "[plus] WARN: extra_requirements install failed"
        fi
        install_editable_or_pth plus "$LIBERO_PLUS_PATH" liberoplus.pth
    fi

    if ! python_clean -c "import libero" >/dev/null 2>&1; then
        echo "[plus] pip editable metadata did not expose import 'libero'; using nested-package .pth"
        write_pth plus "$plus_pythonpath" liberoplus.pth
    fi

    write_libero_config plus "$LIBERO_PLUS_CONFIG_PATH" "$plus_package_dir"

    PYTHONNOUSERSITE=1 PYTHONPATH= LIBERO_CONFIG_PATH="$LIBERO_PLUS_CONFIG_PATH" "$PYTHON" - <<'PY'
import importlib
import os

module = importlib.import_module("libero")
print(f"[verify] libero imported from {module.__file__}")
root = os.path.dirname(module.__file__)
for name in ("bddl_files", "init_files"):
    path = os.path.join(root, name)
    if not os.path.isdir(path):
        raise SystemExit(f"[verify] missing {name} dir at {path}")
    print(f"[verify] {name} dir at {path} has {sum(1 for _ in os.scandir(path))} entries")
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
