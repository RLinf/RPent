#!/usr/bin/env bash
# ============================================
# PhysicalAgent 一键部署脚本
# ============================================
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PARENT="$(cd "$REPO_ROOT/.." && pwd)"

MODEL_DIR="$PARENT/hf_models"

# 依赖源码 (symlink 进 pa_robocasa_work)
ROBOSUITE_SRC="$PARENT/robosuite"
ROBOCASA_SRC="$PARENT/robocasa"
ROBOCASA_ASSETS="$PARENT/hf_datasets/robocasa_kitchen_assets_extracted/robocasa/robocasa/models/assets"

VENV_DIR="$PARENT/.venv_pa_robocasa"

# 国内镜像加速 (pip/uv/hf)
export UV_DEFAULT_INDEX="https://mirrors.aliyun.com/pypi/simple/"
export UV_INDEX_URL="https://mirrors.aliyun.com/pypi/simple/"
export PIP_INDEX_URL="https://mirrors.aliyun.com/pypi/simple/"
export HF_ENDPOINT="https://hf-mirror.com"
export NO_PROXY="localhost,127.0.0.1,mirrors.aliyun.com,mirrors.ustc.edu.cn,hf-mirror.com,.aliyun.com,.aliyuncs.com"
export no_proxy="$NO_PROXY"
# uv link mode: copy avoids hardlink cross-fs warnings
export UV_LINK_MODE=copy
export UV_CACHE_DIR="$PARENT/.uv_cache"

# 确保 uv 可用
if ! command -v uv >/dev/null 2>&1; then
    echo "ERROR: uv not found. Install with: pip install uv (or use system uv)"
    # 尝试从系统 PATH 找 uv
    export PATH="/usr/local/bin:/usr/bin:$PATH"
    if ! command -v uv >/dev/null 2>&1; then
        echo "ERROR: uv not found anywhere. Install with: pip install uv"
        exit 1
    fi
fi

cd "$REPO_ROOT"

echo "============================================"
echo " PhysicalAgent Deployment (self-contained)"
echo " REPO:          $REPO_ROOT"
echo " MODELS:        $MODEL_DIR"
echo " ROBOSUITE_SRC: $ROBOSUITE_SRC"
echo " ROBOCASA_SRC:  $ROBOCASA_SRC"
echo "============================================"

# ---- 0. 前置检查: 依赖源码必须存在于 zhuchunyang_rl 下 ----
echo "[0/6] Checking self-contained sources..."
for d in "$ROBOSUITE_SRC" "$ROBOCASA_SRC" "$ROBOCASA_ASSETS"; do
    if [ ! -d "$d" ]; then
        echo "  ERROR: required source missing: $d"
        echo "  (所有依赖必须复制到 /mnt/public/zhuchunyang_rl/ 下, 禁止使用 /mnt/public/tgy/)"
        exit 1
    fi
    echo "  ok: $d"
done

# ---- 1. 修复内部 symlink ----
echo "[1/6] Fixing internal symlinks..."

rm -f "$REPO_ROOT/checkpoints"
ln -sf "$MODEL_DIR" "$REPO_ROOT/checkpoints"
echo "  checkpoints -> $MODEL_DIR"

# robocasa assets 在 editable 安装之后再挂载 (见 [4c] 后)。
# 原因: robocasa setup.py 用 include_package_data + MANIFEST.in 递归收集
# *.xml/*.png/*.obj/*.stl 等, 若此时 assets(19G) 已挂载, editable build 会
# 遍历海量文件卡死数十分钟。build 时保持 assets 目录不存在 -> 秒级完成。
rm -f "$ROBOCASA_SRC/robocasa/models/assets"   # 确保 build 期间不存在
echo "  (robocasa assets 延迟到 editable 安装后挂载)"

rm -rf "$REPO_ROOT/external_dependencies"
mkdir -p "$REPO_ROOT/external_dependencies"
ln -sf "$ROBOCASA_SRC" "$REPO_ROOT/external_dependencies/robocasa365"
echo "  external_dependencies/robocasa365 -> $ROBOCASA_SRC"

# ---- 3. 创建 Python 3.10 venv ----
echo "[3/6] Creating Python 3.10 venv..."
rm -rf "$VENV_DIR"

PY310=$(which python3.10 2>/dev/null || echo /usr/bin/python3.10)
if [ ! -x "$PY310" ]; then
    echo "  ERROR: python3.10 not found!"
    exit 1
fi

uv venv "$VENV_DIR" --python "$PY310"
source "$VENV_DIR/bin/activate"
echo "  Python: $(python --version)"

# ---- 4. 安装 Python 依赖 ----
echo "[4/6] Installing Python dependencies (this may take a while)..."
echo "  (using Aliyun PyPI mirror for speed)"

# 基础
uv pip install setuptools wheel pip

# PyTorch (CUDA)
echo "  [4a] PyTorch..."
uv pip install torch==2.7.0 torchvision==0.22.0

# robosuite - 从 zhuchunyang_rl 本地路径安装 (--no-deps: 依赖在 [4e] 显式安装)
echo "  [4b] robosuite (from $ROBOSUITE_SRC)..."
uv pip install -e "$ROBOSUITE_SRC" --config-settings editable_mode=compat --no-deps

# robocasa365 - 从 zhuchunyang_rl 本地路径安装 (--no-deps: 依赖在 [4e] 显式装齐,
# 跳过 lerobot/tianshou 等重依赖树解析; 这些仅数据集脚本用到, 仿真运行不需要)
echo "  [4c] robocasa365 (from $ROBOCASA_SRC)..."
uv pip install -e "$REPO_ROOT/external_dependencies/robocasa365" --config-settings editable_mode=compat --no-deps

# editable 安装完成 -> 现在挂载 assets (build 已结束, 不再扫描)
ln -sf "$ROBOCASA_ASSETS" "$ROBOCASA_SRC/robocasa/models/assets"
echo "  robocasa/models/assets -> $ROBOCASA_ASSETS (挂载完成)"

# rldx - 从 GitHub 克隆 RLDX-1 (https://github.com/RLWRLD/RLDX-1)
echo "  [4d] rldx (from https://github.com/RLWRLD/RLDX-1)..."
RLDX_SRC="$PARENT/rldx_pure"
if [ ! -d "$RLDX_SRC/rldx" ]; then
    echo "    克隆 RLDX-1 到 $RLDX_SRC ..."
    rm -rf "$RLDX_SRC"
    git clone --depth 1 https://ghfast.top/https://github.com/RLWRLD/RLDX-1.git "$RLDX_SRC"
fi
uv pip install -e "$RLDX_SRC" --config-settings editable_mode=compat --no-deps

# 核心依赖
echo "  [4e] Core dependencies..."
uv pip install \
    transformers==4.57.0 \
    diffusers==0.35.1 \
    accelerate \
    peft==0.17.1 \
    gymnasium==1.2.2 \
    numpy==2.2.5 \
    pandas==2.2.3 \
    einops==0.8.1 \
    scipy \
    omegaconf \
    safetensors \
    huggingface-hub \
    datasets \
    wandb \
    deepspeed==0.17.6 \
    tyro==0.9.17 \
    click \
    tqdm \
    Pillow \
    PyYAML \
    requests \
    packaging \
    termcolor \
    opencv-contrib-python-headless \
    websockets \
    pyzmq \
    albumentations \
    dm-tree \
    av \
    msgpack \
    msgpack-numpy \
    lmdb \
    moviepy \
    torchcodec \
    numba==0.61.2 \
    mujoco==3.3.1 \
    pygame \
    pynput \
    imageio \
    h5py \
    lxml \
    hidapi

# ---- 5. .venv symlink ----
echo "[5/6] Setting .venv symlink..."
rm -f "$REPO_ROOT/.venv"
ln -sf "$VENV_DIR" "$REPO_ROOT/.venv"
echo "  .venv and nested SIM_PY linked"

# ---- 6. 验证 ----
echo "[6/6] Verifying..."

SIM_PY="$VENV_DIR/bin/python"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl

OK=0
"$SIM_PY" -c "
import robosuite; print('[ok] robosuite:', robosuite.__version__, '->', robosuite.__file__)
import robocasa;  print('[ok] robocasa ->', robocasa.__path__[0])
import rldx;      print('[ok] rldx')
import gymnasium; print('[ok] gymnasium')
print('ALL IMPORTS PASSED')
" && OK=1 || OK=0

# 验证没有任何包指向 /mnt/public/tgy/
echo ""
TGY_REFS=$("$SIM_PY" -c "import robosuite, robocasa; print(robosuite.__file__); print(robocasa.__path__[0])" 2>/dev/null | grep "/mnt/public/tgy" || true)
if [ -n "$TGY_REFS" ]; then
    echo "[FAIL] Some packages still reference /mnt/public/tgy: $TGY_REFS"
    OK=0
else
    echo "[ok] No /mnt/public/tgy references"
fi

# 验证 symlink 链
ls "$REPO_ROOT/scripts/run_robocasa.sh" >/dev/null 2>&1 && echo "[ok] Launcher present" || echo "[FAIL] Launcher missing"
ls "$REPO_ROOT/checkpoints/RLDX-1-FT-RC365/config.json" >/dev/null 2>&1 && echo "[ok] Model path resolves" || echo "[FAIL] Model path broken"

echo ""
echo "============================================"
if [ "$OK" = "1" ]; then
    echo " DEPLOYMENT SUCCESSFUL!"
else
    echo " Some checks failed - review output above"
fi
echo ""
echo " Run a task:"
echo "   bash $REPO_ROOT/scripts/run_robocasa.sh <TASK> <GPU> <SEED> [MODEL] [TIMEOUT]"
echo ""
echo " Example:"
echo "   bash $REPO_ROOT/scripts/run_robocasa.sh SteamInMicrowave 0 3"
echo "============================================"
