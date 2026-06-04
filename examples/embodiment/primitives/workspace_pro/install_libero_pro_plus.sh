#!/usr/bin/env bash
# install_libero_pro_plus.sh
#
# Reproduce a working LIBERO-Pro + LIBERO-Plus install on top of an EXISTING
# venv that already has openpi + base libero (no fresh venv, no torch reinstall).
# Tailored for the layout on this machine where:
#   - openpi venv at /opt/venv/openpi
#   - base libero already installed editable at /opt/venv/openpi/libero
#   - HF perturbation snapshot at /mnt/public2/zhangyixian/datasets/liberopro_hf
#
# What this script does (all idempotent):
#   1. clone (or reuse) LIBERO-PRO and LIBERO-plus repos
#   2. pip install -e each into the openpi venv
#   3. apply liberopro_register_perturbations.patch (registers the 16
#      perturbation suites + overrides Task.language from BDDL)
#   4. sync the HF perturbation snapshot (bddl_files + init_files) over
#      the upstream liberopro install directory — upstream ships broken
#      / empty .pruned_init for several suites
#   5. install LIBERO-plus extra apt packages + extra_requirements.txt
#   6. verify both packages import and benchmarks list cleanly
#
# Usage:
#   bash install_libero_pro_plus.sh                  # both, default paths
#   bash install_libero_pro_plus.sh --only pro       # only LIBERO-Pro
#   bash install_libero_pro_plus.sh --only plus      # only LIBERO-Plus
#   bash install_libero_pro_plus.sh --venv /path/to/venv
#
# Env vars (override defaults):
#   VENV_PY            python in target venv (default /opt/venv/openpi/bin/python)
#   LIBERO_PRO_PATH    local LIBERO-PRO checkout (default $VENV_DIR/libero_pro)
#   LIBERO_PLUS_PATH   local LIBERO-plus checkout (default /mnt/public2/zhangyixian/LIBERO-plus)
#   LIBERO_PRO_HF_DIR  HF snapshot of perturbation BDDL+init files
#                      (default /mnt/public2/zhangyixian/datasets/liberopro_hf)
#   USE_MIRROR=1       use ghfast.top mirror for github clones (TLS-stable)

set -euo pipefail

# -----------------------------------------------------------------------------
# defaults
VENV_PY="${VENV_PY:-/opt/venv/openpi/bin/python}"
VENV_DIR="$(dirname "$(dirname "$VENV_PY")")"
PIP=("$VENV_PY" -m pip)

LIBERO_PRO_PATH="${LIBERO_PRO_PATH:-$VENV_DIR/libero_pro}"
LIBERO_PLUS_PATH="${LIBERO_PLUS_PATH:-/mnt/public2/zhangyixian/LIBERO-plus}"
LIBERO_PRO_HF_DIR="${LIBERO_PRO_HF_DIR:-/mnt/public2/zhangyixian/datasets/liberopro_hf}"

THIS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PATCH_FILE="$THIS_DIR/liberopro_register_perturbations.patch"

if [ "${USE_MIRROR:-0}" = "1" ]; then
    GH="https://ghfast.top/https://github.com"
else
    GH="https://github.com"
fi

ONLY=""
while [ $# -gt 0 ]; do
    case "$1" in
        --only) ONLY="$2"; shift 2 ;;
        --venv) VENV_PY="$2/bin/python"; VENV_DIR="$2"; PIP=("$VENV_PY" -m pip); shift 2 ;;
        -h|--help) sed -n '2,30p' "$0"; exit 0 ;;
        *) echo "unknown arg: $1" >&2; exit 1 ;;
    esac
done

echo "[install] VENV_PY        = $VENV_PY"
echo "[install] LIBERO_PRO     = $LIBERO_PRO_PATH"
echo "[install] LIBERO_PLUS    = $LIBERO_PLUS_PATH"
echo "[install] HF snapshot    = $LIBERO_PRO_HF_DIR"
echo "[install] github prefix  = $GH"
[ -n "$ONLY" ] && echo "[install] only          = $ONLY"

if [ ! -x "$VENV_PY" ]; then
    echo "[ERROR] $VENV_PY not found. Set VENV_PY to your existing python." >&2
    exit 1
fi

# -----------------------------------------------------------------------------
# helpers

clone_or_reuse() {
    local target_dir="$1" repo_url="$2"
    if [ -d "$target_dir/.git" ]; then
        echo "[clone_or_reuse] reusing existing checkout at $target_dir"
    else
        echo "[clone_or_reuse] cloning $repo_url -> $target_dir"
        mkdir -p "$(dirname "$target_dir")"
        git clone "$repo_url" "$target_dir" || {
            echo "[clone_or_reuse] git clone failed; if TLS is unstable try USE_MIRROR=1" >&2
            return 1
        }
    fi
}

sync_dir() {
    # Copy files from src/ to dst/ for matching suite subdirs.
    # dst must already exist (we never create new suite dirs).
    local src="$1" dst="$2" pattern="$3"
    [ -d "$src" ] || { echo "[sync] $src missing — skip"; return; }
    [ -d "$dst" ] || { echo "[sync] $dst missing — skip"; return; }
    local n=0
    for suite_dir in "$src"/*/; do
        local name; name=$(basename "$suite_dir")
        mkdir -p "$dst/$name"
        for f in "$suite_dir"$pattern; do
            [ -f "$f" ] || continue
            cp -f "$f" "$dst/$name/" && n=$((n+1))
        done
    done
    echo "[sync] copied $n files from $src/* into $dst/*"
}

# =============================================================================
# 1. LIBERO-PRO
install_libero_pro() {
    echo
    echo "================ LIBERO-PRO ================"

    # 1a. ensure repo present
    clone_or_reuse "$LIBERO_PRO_PATH" "$GH/RLinf/LIBERO-PRO.git"

    # 1b. editable install (uses venv's existing torch/etc cache)
    # Skip if already importable AND points at the same source dir — avoids
    # network roundtrip when the mirror is flaky.
    local already_at
    already_at=$("$VENV_PY" -c "import liberopro, os; print(os.path.realpath(os.path.dirname(liberopro.__file__)))" 2>/dev/null || true)
    local target_at; target_at=$(realpath "$LIBERO_PRO_PATH/liberopro" 2>/dev/null || echo "")
    if [ -n "$already_at" ] && [ "$already_at" = "$target_at" ]; then
        echo "[pro] liberopro already installed editable at $LIBERO_PRO_PATH — skip pip install"
    else
        echo "[pro] running pip install -e $LIBERO_PRO_PATH (use --no-build-isolation to avoid setuptools download)"
        "${PIP[@]}" install -e "$LIBERO_PRO_PATH" --no-build-isolation || {
            echo "[pro] WARN: pip install failed (network?). Falling back to manual editable install via .pth"
            local sp; sp=$("$VENV_PY" -c "import site; print(site.getsitepackages()[0])")
            echo "$LIBERO_PRO_PATH" > "$sp/liberopro.pth"
            echo "[pro] wrote $sp/liberopro.pth"
        }
    fi

    # 1c. apply perturbation-registration patch (idempotent: check before apply)
    if [ ! -f "$PATCH_FILE" ]; then
        echo "[pro] WARN: patch file missing at $PATCH_FILE — skipping registration patch"
    else
        pushd "$LIBERO_PRO_PATH" >/dev/null
        if git apply --check "$PATCH_FILE" 2>/dev/null; then
            echo "[pro] applying perturbation registration patch"
            git apply "$PATCH_FILE"
        elif git apply --reverse --check "$PATCH_FILE" 2>/dev/null; then
            echo "[pro] patch already applied — skip"
        else
            echo "[pro] WARN: patch applies neither forward nor reverse — possible conflict."
            echo "         Inspect manually: $PATCH_FILE"
        fi
        popd >/dev/null
    fi

    # 1d. sync HF perturbation snapshot (bddl_files + init_files) over the
    # upstream install dir; upstream ships several .pruned_init at 0 bytes
    # or 364 bytes (broken) instead of the correct multi-KB files.
    if [ -d "$LIBERO_PRO_HF_DIR" ]; then
        local dest="$LIBERO_PRO_PATH/liberopro/liberopro"
        sync_dir "$LIBERO_PRO_HF_DIR/bddl_files" "$dest/bddl_files"  "*.bddl"
        sync_dir "$LIBERO_PRO_HF_DIR/init_files" "$dest/init_files"  "*.pruned_init"
    else
        echo "[pro] WARN: HF snapshot dir $LIBERO_PRO_HF_DIR missing."
        echo "         Re-download with:"
        echo "         $VENV_PY -c \"from huggingface_hub import snapshot_download; "
        echo "             snapshot_download(repo_id='zhouxueyang/LIBERO-Pro', repo_type='dataset',"
        echo "                 local_dir='$LIBERO_PRO_HF_DIR',"
        echo "                 allow_patterns=['bddl_files/**','init_files/**'])\""
    fi

    # 1e. verify perturbation suites resolve + trials are nonzero
    LIBERO_TYPE=pro "$VENV_PY" - <<'PYEOF'
import os
os.environ.setdefault("LIBERO_TYPE", "pro")
import liberopro.liberopro.benchmark as bench
suites = ["libero_spatial_task","libero_spatial_swap","libero_spatial_lan",
          "libero_10_task","libero_10_swap","libero_10_lan",
          "libero_goal_task","libero_goal_swap","libero_goal_lan",
          "libero_object_task","libero_object_swap","libero_object_lan"]
# Known-broken on upstream + HF — P2 swap doesn't apply to spatial-relation
# tasks (per LIBERO-Pro paper). Treat as expected-empty, not an error.
KNOWN_EMPTY = {"libero_spatial_swap"}
bad = []
for s in suites:
    try:
        b = bench.get_benchmark(s)()
        ntr = len(b.get_task_init_states(0))
        lang = b.get_task(0).language[:60]
        status = "OK" if ntr > 0 else "EMPTY_INIT"
        print(f"  {s:25s} t0 trials={ntr:>3}  '{lang}'  [{status}]")
        if ntr == 0 and s not in KNOWN_EMPTY: bad.append(s)
    except Exception as e:
        msg = f"ERROR {type(e).__name__}: {e}"
        print(f"  {s:25s} {msg}")
        if s not in KNOWN_EMPTY: bad.append(s)
if bad:
    raise SystemExit(f"[verify] {len(bad)} suites broken (excl. known-empty): {bad}")
print(f"[verify] all probed suites have nonzero trials "
      f"(skipped known-empty: {sorted(KNOWN_EMPTY)})")
PYEOF
    echo "[pro] OK"
}

# =============================================================================
# 2. LIBERO-PLUS
install_libero_plus() {
    echo
    echo "================ LIBERO-PLUS ================"

    # 2a. apt system deps (best-effort; need sudo)
    if command -v apt-get >/dev/null 2>&1; then
        local PKGS=(libexpat1 libfontconfig1-dev libpython3-stdlib libmagickwand-dev)
        if dpkg -s "${PKGS[@]}" >/dev/null 2>&1; then
            echo "[plus] apt deps already installed"
        else
            echo "[plus] installing apt deps: ${PKGS[*]}"
            if [ "$(id -u)" -eq 0 ]; then
                apt-get update -y && apt-get install -y "${PKGS[@]}" || \
                    echo "[plus] WARN: apt install failed; install manually if you see runtime errors"
            else
                sudo apt-get update -y && sudo apt-get install -y "${PKGS[@]}" || \
                    echo "[plus] WARN: sudo apt install failed; install manually if needed"
            fi
        fi
    else
        echo "[plus] no apt-get; skip system-deps step"
    fi

    # 2b. ensure repo present
    if [ ! -d "$LIBERO_PLUS_PATH/.git" ]; then
        clone_or_reuse "$LIBERO_PLUS_PATH" "$GH/sylvestf/LIBERO-plus.git"
    else
        echo "[plus] reusing existing checkout at $LIBERO_PLUS_PATH"
    fi

    # 2c. extra_requirements.txt + editable install (skip if already in place)
    local already_at
    already_at=$("$VENV_PY" -c "import liberoplus, os; print(os.path.realpath(os.path.dirname(liberoplus.__file__)))" 2>/dev/null || true)
    local target_at; target_at=$(realpath "$LIBERO_PLUS_PATH/liberoplus" 2>/dev/null || echo "")
    if [ -n "$already_at" ] && [ "$already_at" = "$target_at" ]; then
        echo "[plus] liberoplus already installed editable at $LIBERO_PLUS_PATH — skip pip install"
    else
        if [ -f "$LIBERO_PLUS_PATH/extra_requirements.txt" ]; then
            "${PIP[@]}" install -r "$LIBERO_PLUS_PATH/extra_requirements.txt" --no-build-isolation || \
                echo "[plus] WARN: extra_requirements install failed; some optional deps may be missing"
        fi
        "${PIP[@]}" install -e "$LIBERO_PLUS_PATH" --no-build-isolation || {
            echo "[plus] WARN: pip install failed; falling back to .pth"
            local sp; sp=$("$VENV_PY" -c "import site; print(site.getsitepackages()[0])")
            echo "$LIBERO_PLUS_PATH" > "$sp/liberoplus.pth"
        }
    fi

    # 2d. verify
    "$VENV_PY" - <<'PYEOF'
import importlib
m = importlib.import_module("liberoplus")
print(f"[verify] liberoplus imported from {m.__file__}")
# assets sanity-check
import os
assets = os.path.join(os.path.dirname(m.__file__), "liberoplus", "assets")
if os.path.isdir(assets):
    n = sum(1 for _ in os.scandir(assets))
    print(f"[verify] assets dir at {assets} has {n} entries")
else:
    print(f"[verify] WARN: assets dir missing at {assets} — download from "
          "https://huggingface.co/datasets/Sylvest/LIBERO-plus/tree/main "
          "and unzip to <repo>/libero/libero/assets/ (see LIBERO-plus README)")
PYEOF
    echo "[plus] OK"
}

# =============================================================================
case "$ONLY" in
    pro)    install_libero_pro ;;
    plus)   install_libero_plus ;;
    "")     install_libero_pro; install_libero_plus ;;
    *)      echo "unknown --only target: $ONLY" >&2; exit 1 ;;
esac

echo
echo "[install] DONE."
