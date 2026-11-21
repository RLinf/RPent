# Copy to .rl-env.sh and source it from the checkout root after activating a venv.
# This file contains paths and service settings, never credentials.
export RPENT_DATA_ROOT="${RPENT_DATA_ROOT:-${XDG_DATA_HOME:-$HOME/.local/share}/rpent}"
export MODEL_PATH="$RPENT_DATA_ROOT/models/Qwen3.8-27B"
export SAM3_CHECKPOINT_PATH="$RPENT_DATA_ROOT/models/sam3/sam3.pt"
export LIBERO_CONFIG_PATH="$RPENT_DATA_ROOT/libero-config"
export LIBERO_PRO_ASSET_PATH="$RPENT_DATA_ROOT/libero-pro/assets"
export LIBERO_TYPE=pro
export ROBOTWIN_ASSETS_PATH="$RPENT_DATA_ROOT/robotwin"
export RPENT_ROBOCASA_PYTHON="$PWD/.venv-robocasa/bin/python"
export RPENT_ROBOTWIN_PYTHON="$PWD/.venv-robotwin/bin/python"
export RPENT_MODEL=openai-chat:qwen3.8-27b
export RPENT_BASE_URL=http://127.0.0.1:30080/v1
export RPENT_SAM3_ENDPOINT=http://127.0.0.1:8112
# These are explicit example settings, not launcher defaults. When unset,
# presence/repetition penalties fall back to the model server's defaults.
export RPENT_PRESENCE_PENALTY=0
export RPENT_REPETITION_PENALTY=1.0
