# Source after activating the RPent Python environment.
EVAL_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export RPENT_REPO_ROOT="$(cd "$EVAL_DIR/../.." && pwd)"
command -v rpent >/dev/null || { echo "Activate the RPent environment first." >&2; return 1; }
export PYTHONPATH="$RPENT_REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}"
export RPENT_DATA_ROOT="${RPENT_DATA_ROOT:-${XDG_DATA_HOME:-$HOME/.local/share}/rpent}"
export SAM3_CHECKPOINT_PATH="${SAM3_CHECKPOINT_PATH:-$RPENT_DATA_ROOT/models/sam3/sam3.pt}"
export LIBERO_CONFIG_PATH="${LIBERO_CONFIG_PATH:-$RPENT_DATA_ROOT/libero-config}"
export LIBERO_PRO_ASSET_PATH="${LIBERO_PRO_ASSET_PATH:-$RPENT_DATA_ROOT/libero-pro/assets}"
export LIBERO_TYPE="${LIBERO_TYPE:-pro}"
export MUJOCO_GL="${MUJOCO_GL:-egl}"
export PYOPENGL_PLATFORM="${PYOPENGL_PLATFORM:-egl}"
