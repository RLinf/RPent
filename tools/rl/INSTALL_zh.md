# 安装共享 RL 工作流

[运行任务](README_zh.md) · [English](INSTALL.md)

本指南从新的 Linux checkout 开始：先安装公共 planner、SAM3 与 LIBERO 环境，
再按需安装隔离的 RoboCasa、RoboTwin 环境。所有路径由自己的 checkout 与数据
目录推导，不依赖集群激活脚本、已有 checkpoint 或 VLA 权重。除特别说明外，
所有命令均在 checkout 根目录执行。

## 1. 系统条件与源码

需要 Linux x86_64、Git、[uv](https://docs.astral.sh/uv/getting-started/installation/)
和正常工作的 NVIDIA 驱动，先确认 `nvidia-smi` 可用。示例 SGLang wheels 使用
CUDA 13，需要兼容驱动；仿真环境使用 CUDA 12.8。激活 Python 环境不会安装系统
驱动，也不会清理继承的 CUDA 库路径。RoboTwin 还需要 NVIDIA Vulkan/EGL、
CUDA 12.8 toolkit（含 `nvcc`）及编译 cuRobo 的 C++ 编译器。
Ubuntu 上安装其余系统依赖：

```bash
sudo apt-get update
sudo apt-get install -y git build-essential ffmpeg libegl1 libopengl0 libgl1 libglib2.0-0 libvulkan1 vulkan-tools
# 如果尚未安装 uv，先按上方官方链接安装。
uv python install 3.12 3.11
git clone --branch rpent-rl-workflow https://github.com/zhexuanxu/RPent.git
cd RPent
cp tools/rl/config.example.sh .rl-env.sh
```

如果开发分支尚未发布，使用本次改动提供的源码包，替代上面的 Git clone 步骤：

```bash
tar -xzf rpent-rl-workflows-source.tar.gz
cd RPent
cp tools/rl/config.example.sh .rl-env.sh
```

使用包含 `robots/rl` 的版本，并随实验保存 `git rev-parse HEAD`。编辑 `.rl-env.sh`
选择磁盘、模型/SAM3 位置和服务地址。默认数据根目录为
`${XDG_DATA_HOME:-$HOME/.local/share}/rpent`。分别打开新的终端并进入 checkout，
再激活对应环境；不要继承 RL/SFT 任务的 `PYTHONPATH` 或 CUDA 库覆盖设置。

## 2. Planner、SAM3 与 LIBERO

公共环境负责 planner、几何计算、SAM3 和 LIBERO。RoboCasa、RoboTwin 使用独立
子进程与 Python 环境，避免 robosuite、NumPy、SciPy、Torch 的版本冲突。

```bash
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python --prerelease=allow --torch-backend cu128 -r requirements/libero-rl.txt
source .venv/bin/activate
source .rl-env.sh
source tools/rl/env.sh
uv pip check --python .venv/bin/python
rpent --help
```

依赖文件选择匹配的 RLinf `rpent-ppo` runtime、SAM3 和公开 LIBERO forks，
不单独使用旧版 `rpent[rlinf]` extra，也不需要 OpenPi/Pi0.5。验证过的主版本为
Python 3.12、Torch 2.11.0/CUDA 12.8、torchvision 0.26.0、Transformers 4.57.6、
MuJoCo 3.3.0、robosuite 1.5.2、NumPy 1.26.4。依赖文件不是完整传递依赖锁文件，
部署时还应保存 `uv pip freeze`。

先在 [facebook/sam3](https://huggingface.co/facebook/sam3) 获得访问权限，再登录
下载。Qwen 需下载完整目录，包括 tokenizer、processor 和 chat template。
下载需要网络及足够的磁盘空间：

```bash
unset HF_HUB_OFFLINE HF_DATASETS_OFFLINE TRANSFORMERS_OFFLINE
hf auth login
hf download Qwen/Qwen3.8-27B --local-dir "$MODEL_PATH"
hf download facebook/sam3 sam3.pt --local-dir "$(dirname "$SAM3_CHECKPOINT_PATH")"
mkdir -p "$LIBERO_CONFIG_PATH" "$LIBERO_PRO_ASSET_PATH"
env -u LIBERO_PRO_ASSET_PATH liberopro-download-assets \
  --assets-dir "$LIBERO_PRO_ASSET_PATH" --skip-existing --no-cache
```

临时 `env -u` 避免下载器选用已有资产目录的 fallback。网络需要 HF 镜像时，可在
下载前设置 `HF_ENDPOINT`；代理按自己的网络配置，无需私有下载脚本。
运行任务不需要演示数据集。BDDL 与初始化状态随 `rpent-liberopro` 安装，
完成安装及资产下载后生成本地配置：

```bash
python - <<'PY'
import os
from pathlib import Path
import yaml
import liberopro.liberopro as libero
config = libero.get_default_path_dict()
config["assets"] = str(Path(os.environ["LIBERO_PRO_ASSET_PATH"]).resolve())
config["datasets"] = str(Path(os.environ["RPENT_DATA_ROOT"]) / "libero-pro/datasets")
for key in ("benchmark_root", "bddl_files", "init_states", "assets"):
    assert Path(config[key]).is_dir(), (key, config[key])
path = Path(os.environ["LIBERO_CONFIG_PATH"]) / "config.yaml"
path.write_text(yaml.safe_dump(config))
print(path.read_text())
PY
```

`benchmark_root` 保留安装包的实际路径。YAML 中包含绝对路径，移动环境或资产后
须重新生成。本指南使用 LIBERO-PRO（`LIBERO_TYPE=pro`）；standard/plus 需安装
对应包并配置对应资产及配置文件。

## 3. 模型服务

在 checkout 根目录的新终端中单独安装 SGLang：

```bash
uv venv --python 3.12 .venv-sglang
uv pip install --python .venv-sglang/bin/python --prerelease=allow "sglang==0.5.23" "torchvision==0.29.2"
uv pip check --python .venv-sglang/bin/python
```

验证过的服务栈为 SGLang 0.5.23、sgl-kernel 0.4.10、Torch 2.14.1/CUDA 13.0、
Transformers 5.19.0；与仿真环境的 CUDA 12.8 wheels 分开。
[运行指南](README_zh.md#启动公共服务)给出模型启动、`/v1/models` 检查、SAM3
启动及第一个 LIBERO 任务。已有兼容远程模型端点时，可跳过本地模型安装。
checkpoint 不含 MTP 权重时，运行 `serve.sh` 设置 `ENABLE_MTP=0`。

## 4. RoboCasa 环境与资产

在 checkout 根目录创建隔离的 Python 3.11 环境。固定的 robosuite revision 提供
`load_model_on_init`，不能用 LIBERO 环境中的 PyPI 1.5.2 替代：

```bash
uv venv --python 3.11 .venv-robocasa
uv pip install --python .venv-robocasa/bin/python -r requirements/rl-robocasa.txt
uv pip check --python .venv-robocasa/bin/python
.venv-robocasa/bin/python -c "import robocasa, robosuite; print('RoboCasa imports OK')"
```

用仿真环境的下载器准备资产。此版本将资产写入安装包的
`robocasa/models/assets`，最简单的方式是在空间充足的磁盘创建 venv：

```bash
.venv-robocasa/bin/python -m robocasa.scripts.download_kitchen_assets --type all
```

按下载器提示确认资产下载，不需要演示/轨迹数据集。若希望放到其他磁盘，须在
**下载前**复制包中已有的资产目录，再将包目录替换为软链接。仅设置
`ROBOCASA_ASSETS_PATH` 不会改变此版本下载器的位置；[参考文档](REFERENCE_zh.md#迁移-robocasa-资产)
提供可选的迁移命令。

`.rl-env.sh` 示例已将 `RPENT_ROBOCASA_PYTHON` 指向此 venv。运行轨迹时激活公共
`.venv`，`--simulator robocasa` 会用隔离解释器启动原生服务，无需 RLDX 权重。

## 5. RoboTwin 环境与资产

先检查 `nvcc --version` 与 Torch 构建使用的 toolkit 一致（这里为 CUDA 12.8），
且已安装 NVIDIA Vulkan ICD。无显示器的机器上，`vulkaninfo --summary` 也应列出
NVIDIA GPU。然后安装：

```bash
uv venv --python 3.11 .venv-robotwin
uv pip install --python .venv-robotwin/bin/python torch==2.8.0+cu128 --index-url https://download.pytorch.org/whl/cu128
uv pip install --python .venv-robotwin/bin/python -r requirements/rl-robotwin.txt
uv pip install --python .venv-robotwin/bin/python ninja wheel 'setuptools>=69.5.1,<75.9'
uv pip install --python .venv-robotwin/bin/python --no-build-isolation 'git+https://github.com/NVlabs/curobo.git@v0.7.8'
```

RoboTwin 还依赖 RLinf 环境适配器。把源码路径写入仿真 venv，不必安装整套
RLinf 训练依赖：

```bash
mkdir -p "$RPENT_DATA_ROOT/src"
git clone --branch rpent-ppo https://github.com/zhexuanxu/RLinf.git "$RPENT_DATA_ROOT/src/RLinf"
git -C "$RPENT_DATA_ROOT/src/RLinf" checkout f351c57e62818b18315ed4b2194312a44cafb697
RLINF_ROOT="$RPENT_DATA_ROOT/src/RLinf" .venv-robotwin/bin/python - <<'PY'
import os
import sysconfig
from pathlib import Path
root = Path(os.environ["RLINF_ROOT"]).resolve()
assert (root / "rlinf/envs/sim/robotwin/robotwin_env.py").is_file()
Path(sysconfig.get_path("purelib"), "rpent_robotwin_rlinf_source.pth").write_text(str(root) + "\n")
PY
uv pip check --python .venv-robotwin/bin/python
.venv-robotwin/bin/python -c "from robots.robotwin.rlinf_env import RoboTwinAgentEnv; print('RoboTwin adapter imports OK')"
.venv-robotwin/bin/robotwin-download-assets --output "$ROBOTWIN_ASSETS_PATH"
```

`ROBOTWIN_ASSETS_PATH` 指向包含 `.robotwin-assets.json` 和 `assets/` 的快照
**根目录**，不是内部的 `assets/`。固定版本 runtime 会验证资产快照。
默认 preset 为 `demo_randomized`，seed 应从该任务/preset 公开的有效列表选择。
激活公共 planner `.venv` 来启动任务；配置示例通过 `RPENT_ROBOTWIN_PYTHON`
选择 `.venv-robotwin/bin/python`，无需 LingBot/VLA 权重。

## 6. 验证与排错

依次运行[指南中的四条单任务命令](README_zh.md#运行一个任务)，不要同时执行。
它们分别覆盖 LIBERO episode/ZAP、RoboCasa episode、RoboTwin episode。
检查原生成功标记及最终工具结果；正常未成功的任务仍可能是有效 episode。

| 现象 | 优先检查 |
| --- | --- |
| Planner 连不上模型 | 从 planner 主机检查 `/v1/models`、模型名、API key 和端点 |
| SAM3 加载失败 | 模型授权、完整权重路径、显存和服务日志 |
| LIBERO 缺 XML/BDDL/init states | 资产清单和 `config.yaml`；移动目录后重新生成 |
| RoboCasa 缺 `load_model_on_init` | `RPENT_ROBOCASA_PYTHON` 是否选择固定 robosuite 的隔离环境 |
| RoboTwin 找不到 `rlinf` | 仿真 venv 的 `.pth` 文件及对应 adapter 源码目录 |
| cuRobo 编译/导入失败 | Torch/toolkit 匹配、C++ 编译器、先安装构建依赖 |
| EGL/Vulkan create-device 失败 | NVIDIA EGL/Vulkan 库、ICD 和真实 GPU 分配；不能只凭此消息判定 CUDA OOM |
| CUDA 库符号不匹配 | 打开干净终端，使用隔离解释器，移除继承的冲突 Torch/NVIDIA 库路径 |
| 未知任务或无效 seed | 原生任务名、preset、RoboTwin 公布的 seed 表 |

如果平台将驱动放在动态加载器搜索范围之外，应按**该平台实际安装的驱动**设置
`LD_LIBRARY_PATH`、`VK_ICD_FILENAMES`，不要复制其他机器的路径。CUDA/Vulkan
设备编号可能不同，首次单环境运行时用 `nvidia-smi` 核对实际分配。测量场景驻留
显存与冷启动开销后，再增加并发。
