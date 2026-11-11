# RL 工作流参考

[运行第一个任务](README_zh.md) · [安装配置](INSTALL_zh.md) · [English](REFERENCE.md)

## 关键参数

`serve.sh` 可以在环境变量设置之后追加 SGLang 参数，服务监听本机
`127.0.0.1`。规划器和模型服务默认上下文都为 262144 token。

| 服务变量 | 默认值 | 含义 |
| --- | --- | --- |
| `RPENT_SGLANG_PYTHON` | `python` | 已激活 SGLang 环境的解释器；不负责切换 CUDA 库 |
| `SERVE_GPUS` | `2,3` | 可见物理 GPU，数量决定 TP |
| `MODEL_PATH` | 必填 | 本地权重目录 |
| `MODEL_NAME` | `qwen3.8-27b` | 服务模型名称，须与 `RPENT_MODEL` 的 `openai-chat:` 后缀一致 |
| `SERVE_PORT` | `30080` | 本地模型端口 |
| `SERVE_MEM_FRACTION` | `0.8` | SGLang 静态显存比例 |
| `SERVE_CONCURRENCY` | `8` | 最大同时运行的模型请求数 |
| `ENABLE_MTP` | `1` | EAGLE/MTP：3 steps、top-k 1、4 draft tokens；没有 MTP 权重的模型设为 0 |

| 轨迹变量 / 参数 | 默认值 | 含义 |
| --- | --- | --- |
| `RPENT_MODEL`、`RPENT_BASE_URL` | 必填 | 服务模型和 `/v1` endpoint |
| `RPENT_SAM3_ENDPOINT` | 未设置 | 已启动的共享 SAM3 endpoint |
| `RPENT_PROMPT` | `simple` | `simple`、`base`、`base_gripper`、`base_gripper_camera` |
| `RPENT_REASONING_EFFORT` | Qwen3.8 为 `xhigh`，其他模型为 `none` | `none` 不发送 effort 等级，仍启用 thinking |
| `RPENT_MAX_TURNS` | `100` | 单条轨迹的模型请求数 |
| `RPENT_MAX_EPISODE_STEPS` | `10000` | 仿真步预算 |
| `RPENT_MAX_TOKENS` | `32768` | 每次请求的输出预算，包含 thinking |
| `RPENT_CONTEXT_WINDOW` | `262144` | 规划器上下文预算 |
| `RPENT_COMPACTION_FRACTION` | `0.7` | 历史压缩阈值 |
| `RPENT_PLANNER_TIMEOUT_S` | `7200` | 规划器总超时；需要其他限制时显式设置 |
| `RPENT_PRESENCE_PENALTY`、`RPENT_REPETITION_PENALTY` | 未设置 | 可显式覆盖的采样参数 |
| `RPENT_OUTPUT_DIR` | `logs/rl-adaptive-<prompt>-<time>` | 单条轨迹输出目录 |
| `--suite`、`--task`、`--seed` | suite/task 必填，seed 0 | LIBERO 任务及初始状态 seed |
| `--cuda-device` | 未设置 | 仿真 GPU，与共享 SAM3 的 GPU 独立 |

Thinking 采样参数为 temperature 1.0、top-p 0.95、top-k 20。复现 SFT checkpoint
评测的 penalty 时，显式设置 `RPENT_PRESENCE_PENALTY=0` 和
`RPENT_REPETITION_PENALTY=1.0`。`--observation-mode adaptive` 仍可使用，
image/text 模式会被拒绝。`run.sh` 的其他参数会直接传给 `rpent`。

## 批量采集、保存与恢复

沿用上面的模型和 SAM3 环境变量，启动一个支持恢复的任务队列：

```bash
python tools/rl/run_batch.py run "$RPENT_DATA_ROOT/rollouts/new-run" --jobs tools/rl/jobs/panel40.txt --seeds 3-66 --gpus 1 --per-gpu 4
```

清单每行形如 `adaptive libero_object 0 0`。`panel40.txt` 包含
`libero_object`、`libero_goal`、`libero_spatial`、`libero_10` 各自的 task 0–9。
环境并发应结合仿真显存及初始化负载设置，模型服务并发是独立的限制。

| Batch 参数 | 默认值 | 含义 |
| --- | --- | --- |
| `--jobs` | 必填 | JSON 任务清单或 LIBERO 四字段清单 |
| `--seeds` | 使用清单的 seed | 逗号分隔的 seed 或区间，例如 `0,3-66` |
| `--gpus`、`--per-gpu` | GPU 必填，每 GPU 1 个 worker | 环境 GPU 和每卡 worker 数 |
| `--prompt` | `simple` | 整批使用一个 prompt 变体 |
| `--workflow` | `episode` | episode / episode_image / zap_turn / zap_episode / zap_subtask |
| `--no-thinking` | 关闭 | 整批禁用 Qwen thinking，并将该设置写入 manifest |
| `--image_size` | `1024` | VLM 图片边长：1024 / 512 / 256；原始感知始终1024 |
| `--start-barrier` | 关闭 | 所有待跑环境驻留后统一放行首个请求 |
| `--max-initializing-per-gpu` | `8` | 每卡同时冷启动数，独立于轨迹槽位 |
| `--endpoints` | `RPENT_BASE_URL` | 逗号分隔的策略 `/v1` endpoint |
| `--start-interval` | `1` 秒 | 初次启动 worker 的间隔 |
| `--episode-timeout` | `14400` 秒 | 单条轨迹的外层进程超时 |
| `--infrastructure-retries` | `1` | 每次启动中，错误或审计失败后的额外尝试数 |

每个任务独立保存 `attempt-NNN`。审计会检查 native success、完整 HTTP/原生响应、
工具调用和 PNG SHA-256，通过后写入 `COMPLETED.json` 并原子更新
`valid-index.json`。正常完成但没有成功的任务仍是有效数据；异常尝试保留
`INVALID.json`，中断尝试保留 `INTERRUPTED.json`。

正常达到输出、上下文或单条超时限制的失败写入 `EVALUATED.json`，在
`outcomes-index.json` 中计数一次，不当作基础设施故障重试，也不加入训练索引。
中断或损坏的尝试单独保留；不完整响应不会标成完整训练输出。
`valid-index.json` 只包含通过审计的记录。

SIGINT/SIGTERM 会终止本批次拥有的轨迹进程组并保留所有尝试。重复相同命令即可
恢复，已封存轨迹会重新审计后跳过。目录锁防止重复采集器，manifest 会拒绝任务、
模型、prompt 或生成参数变化。只读 summary 会重新审计已封存数据，不覆盖运行中
采集器的索引：

```bash
python tools/rl/run_batch.py summary "$RPENT_DATA_ROOT/rollouts/new-run"
```

单条轨迹包含 `result.json`、`states.json`、transcript、视频和 `api_trace/`。
Trace 保存压缩后实际请求、原始 PNG 字节、原生消息、完整流式响应和压缩事件。
通用读取模块为 `rpent.evaluation.trajectory`，数据编码及 SFT 入口位于
`ms-swift/examples/rpent`。

### 对比 thinking 设置

Qwen3.8 批量任务设置 `RPENT_REASONING_EFFORT=low` 或 `medium`，即可在启用
thinking 时选择对应等级。关闭 thinking 时，设置 `RPENT_REASONING_EFFORT=none`，
并在 `run_batch.py run` 中加上 `--no-thinking`；单条 `run.sh` 也支持该参数。
单独设置 `none` 只是不发送 effort 等级，仍然启用 thinking。每组设置使用新的
批次目录，恢复时不允许改变 thinking 设置。规划器保持采样参数不变，关闭
thinking 不会隐式切换到另一套采样配置。

## 离线安装

在可联网的 Linux x86_64 / Python 3.12 机器上，为两个环境分别准备“完整”
wheelhouse。先按前文完成在线安装，然后在各自环境中构建对应依赖：

```bash
source .venv/bin/activate
uv pip install pip wheel
python -m pip wheel --pre --wheel-dir wheelhouse-rpent --extra-index-url https://download.pytorch.org/whl/cu128 -r requirements/libero-rl.txt setuptools wheel
source .venv-sglang/bin/activate
uv pip install pip wheel
python -m pip wheel --pre --wheel-dir wheelhouse-sglang "sglang==0.5.23" "torchvision==0.29.2"
```

将仓库、两套 wheelhouse、模型目录和仿真资源传到离线机器。
操作系统、CPU 架构、Python ABI 和 CUDA 驱动须匹配；没有预编译 wheel
的扩展还需要在准备机器上安装构建工具链。少量缓存 wheel 不能重建完整环境。
离线机器上分别创建新 venv 并安装各自完整依赖；`--no-deps` 避免包内 Git URL
元数据触发联网，因此所有依赖 wheel 都必须已经准备好。

```bash
uv venv --python 3.12 .venv
source .venv/bin/activate
uv pip install --no-index --no-deps wheelhouse-rpent/*.whl
uv pip install --no-index --no-deps --no-build-isolation -e .
uv venv --python 3.12 .venv-sglang
source .venv-sglang/bin/activate
uv pip install --no-index --no-deps wheelhouse-sglang/*.whl
```

离线执行 `uv venv` 前，机器上须已安装对应 Python 解释器。
在两个环境中分别运行 `uv pip check`。迁移仿真器后重新执行配置生成步骤，
因为 YAML 中保存的是绝对路径。全部资源就绪后，可在运行终端设置
`HF_HUB_OFFLINE=1` 和 `TRANSFORMERS_OFFLINE=1`。
环境安装和维护须与正在运行的训练进程分开。

## 将 planner 与仿真器部署在不同机器

Adaptive episode 支持让 planner 与模型服务部署在一起，将原生仿真器、
动作内部控制循环和共享 SAM3 留在后端机器。两端须使用相同的控制和工具源码、
任务 prompt，并均安装 planner 环境。后端额外准备独立的原生仿真环境和资产，
按前文设置 `RPENT_ROBOCASA_PYTHON`、`RPENT_ROBOTWIN_PYTHON`、
`ROBOTWIN_ASSETS_PATH` 和 `RPENT_RLINF_ROOT`。

后端先启动共享 SAM3，再启动工具管理器。GPU 容量包含初始化中的 episode，
须根据该机器的真实驻留容量测试设置：

```bash
python -m robots.rl.remote_manager \
  --output-dir /path/to/new-backend-run \
  --gpu-slots 0:4,1:6 --max-initializing-per-gpu 2 \
  --sam3-endpoint http://127.0.0.1:8112 --port 28090
```

服务默认只监听 loopback。先建立私网路由或 SSH 转发，再在 planner 机器运行：

```bash
export RPENT_REMOTE_TOOLKIT_ENDPOINT=http://127.0.0.1:28090
unset RPENT_SAM3_ENDPOINT
bash tools/rl/run.sh --simulator robocasa --task-name PickPlaceCounterToCabinet \
  --split target --seed 1 --workflow episode --cuda-device 1
```

这里的 `--cuda-device` 指向后端 GPU。后端负责环境和 SAM3 连接，不要再传
`--env-endpoint` 或 `--sam3-endpoint`。批量入口继承同一远程端点，仍通过
`--endpoints` 分配模型请求。此远程接口只支持 adaptive episode。
握手会拒绝源码摘要、工具声明、任务或预算不一致的连接。

每条后端 episode 有独立且归管理器所有的 worker，保留配置、日志、原生状态、
图片和视频。变化的产物通过 SHA-256 校验后，以原始字节同步到 planner 目录；
相关文件写入后才发布 `states.json`。`remote-runtime.json` 记录 worker
身份和源码，`remote-transfer.jsonl` 记录文件哈希和传输量。Planner transcript
和 wire trace 留在 planner 机器。传输失败会终止 episode，不重放结果不明的
动作。模型推理期间由心跳续租；planner 失联后，已就绪的后端 worker 会在
五分钟后释放，文件继续保留。

## RLinf 与 SFT

RLinf 的独立环境和 `run_train.sh` 配置见它自己的
`examples/agent/rpent/README.md`。将 `tools.rpent.python` 指向这个 RPent
venv 的解释器，并将相同 LIBERO/SAM3 路径传给 episode 子进程。
子进程还需要 RPent 对应的 CUDA/EGL 环境，不能使用 RL 环境的 `PYTHONPATH`
或 Torch/NVIDIA 库路径替代。SGLang 始终在独立环境中运行。

轨迹编码和多模态 SFT 见 `ms-swift/examples/rpent/README.md`。
训练数据须保留压缩后实际请求、原始图片字节及顺序、可见历史 thinking，
以及指定 assistant 输出的 loss mask。

使用当前工作区的 RLinf 配置须将 episode 模块指向 `robots.rl.rl_episode`；
`robots.libero_rl.rl_episode` 属于旧工作区。迁移时保留原生任务字段。

## 迁移 RoboCasa 资产

RoboCasa 通过安装包内的 `robocasa/models/assets` 读取资产。需要共享存储时，
先复制包内自带文件，再将该环境的资产目录改为符号链接，然后下载：

```bash
export ROBOCASA_ASSETS_PATH="$RPENT_DATA_ROOT/robocasa/assets"
.venv-robocasa/bin/python - <<'PY'
import importlib.util, os, shutil
from pathlib import Path
source = Path(importlib.util.find_spec('robocasa').origin).parent / 'models/assets'
target = Path(os.environ['ROBOCASA_ASSETS_PATH']).expanduser().resolve()
if source.resolve() != target:
    shutil.copytree(source, target, dirs_exist_ok=True)
    source.rename(source.with_name('assets.bundled'))
    source.symlink_to(target, target_is_directory=True)
PY
.venv-robocasa/bin/python -m robocasa.scripts.download_kitchen_assets --type all
```

原生下载器会在下载前询问；不需要下载 demonstration 数据集。迁移资产时保留
包内原有场景和配置文件。仅设置 `ROBOCASA_ASSETS_PATH` 不会重定向该版本的读取路径。

五种工作流、SAM3 池与固定任务对比见 [工作流指南](WORKFLOWS_zh.md)。

模型启动器启用缓存上报。八副本 SAM3、单副本等待指标和长历史缓存调优见[工作流性能配置](WORKFLOWS_zh.md#模型前缀缓存诊断)。
