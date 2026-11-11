# Adaptive VLM 机器人工作流

[English](README.md) · [从零安装](INSTALL_zh.md) · [参数与部署参考](REFERENCE_zh.md)

VLM 通过末端控制和视觉工具直接操作机器人。统一入口为 `--robot rl`，
显式选择仿真器和工作流：

| 仿真器 | `--simulator` | 工作流 |
| --- | --- | --- |
| LIBERO / LIBERO-PRO | `libero` | `episode`、`zap_turn`、`zap_episode`、`zap_subtask` |
| RoboCasa365 | `robocasa` | `episode` |
| RoboTwin | `robotwin` | `episode` |

四种模式使用 API planner、原生任务指令和实测机器人状态，成功由仿真器判定。
[工作流指南](WORKFLOWS_zh.md) 详细说明工具、上下文边界、SAM3 池和固定任务对比。

## 安装与配置

先完成 [INSTALL_zh.md](INSTALL_zh.md)：包括系统条件、隔离 Python 环境、
模型/SAM3 下载和各仿真器资产。复制公共配置，再按自己的磁盘和服务地址修改：

```bash
# 在 checkout 根目录执行。
cp tools/rl/config.example.sh .rl-env.sh
```

下面的示例布局是 GPU 0 放 SAM3，GPU 1 放一个仿真环境，GPU 2–3 放 Qwen3.8-27B
TP2。这些只是设备编号，不要求特定机器。已有的兼容模型服务可以替代本地部署。
显存需求取决于模型、场景和并发数；请从一个 episode 开始。

## 启动公共服务

分别打开终端，工作目录都设为 checkout 根目录。先启动模型：

```bash
source .venv-sglang/bin/activate
source .rl-env.sh
SERVE_GPUS=2,3 bash tools/rl/serve.sh
```

等待 `curl --fail http://127.0.0.1:30080/v1/models` 列出 `qwen3.8-27b`。
第二个终端启动唯一的共享 SAM3：

```bash
source .venv/bin/activate
source .rl-env.sh
source tools/rl/env.sh
python -m rpent.robots.components.sam3_server --cuda-device 0 --port 8112
```

等日志打印监听地址后，在第三个终端激活 planner，加载相同配置：

```bash
source .venv/bin/activate
source .rl-env.sh
source tools/rl/env.sh
```

设置 `RPENT_SAM3_ENDPOINT` 后，episode 共用该服务；如果不设置，每条 episode
会在自己的仿真 GPU 上启动一个 SAM3。

## 运行一个任务

以下命令分别执行，不要同时启动。`--cuda-device` 指定仿真 GPU，模型和 SAM3
分别使用配置中指定的服务。

LIBERO-PRO episode：

```bash
RPENT_OUTPUT_DIR=logs/libero-episode bash tools/rl/run.sh \
  --simulator libero --workflow episode \
  --suite libero_object --task 0 --seed 1 --cuda-device 1
```

RoboCasa Target50 episode，需要先安装 RoboCasa：

```bash
RPENT_OUTPUT_DIR=logs/robocasa-episode bash tools/rl/run.sh \
  --simulator robocasa --workflow episode \
  --task-name PickPlaceCounterToCabinet --split target --seed 1 --cuda-device 1
```

RoboTwin episode，需要先安装 RoboTwin：

```bash
RPENT_OUTPUT_DIR=logs/robotwin-episode bash tools/rl/run.sh \
  --simulator robotwin --workflow episode \
  --task-name click_alarmclock --task-config demo_randomized \
  --seed 100000 --cuda-device 1
```

RoboTwin 的任务、preset、seed 组合须在公开评测清单中有效；
`robots/robotwin/eval/demo_randomized.json` 保存各任务的有效 seed。
`tools/rl/jobs/robocasa-robotwin100.json` 也列出 RoboCasa/RoboTwin 示例，
它只是任务清单，并不表示需要并发启动 100 个环境。LIBERO 的 task 索引从 0 开始，
属于指定 suite；`panel40.txt` 和 `libero-dev10.json` 是示例任务清单。

LIBERO-PRO ZAP 使用相同服务和任务参数：

```bash
RPENT_OUTPUT_DIR=logs/libero-zap bash tools/rl/run.sh \
  --simulator libero --workflow zap_turn \
  --suite libero_object --task 0 --seed 1 --cuda-device 1
```

每次运行使用新的输出目录。默认预算为 100 次模型请求、10000 仿真步、7200 秒
planner 超时，单次最多输出 32768 tokens，上下文为 262144 tokens。批量启动器
另有 14400 秒的 episode 超时。完整覆盖项见[参数参考](REFERENCE_zh.md#关键参数)。
启动器显式默认发送 presence penalty 0；示例配置设置 repetition penalty 1.0。
覆盖值会记录在实际模型请求中。

## 工作流与工具

`episode` 显式调用 `view_env`、`segment`、`back_project`。LIBERO 的三个
`zap_*` 模式自动提供标注图像和深度几何，分别在每轮、整条任务或子任务范围内
保留上下文。`zap_episode` 没有 todo；`zap_subtask` 每次有效更新结构化 todo 后
清空对话。[工作流指南](WORKFLOWS_zh.md) 列出完整工具、初始化规则、压缩边界
和仅影响 VLM 图片的 `--image_size`。

RoboCasa 增加局部平面速度与角速度工具 `move_base`；RoboTwin 使用解剖学
`arm` 参数、左右臂图像标识和可选双臂目标。四个 LIBERO 模式共用机器人、坐标和
夹爪说明；可选 `--prompt` 为各模式增加相同细节。任务指令直接来自
`get_task_language()`，没有逐任务 prompt 文件。

## 查看结果

`result.json` 记录原生成功标记、错误、轮数和 tokens；`states.json` 保存实测状态
及原生终止标记。进程正常退出不等于任务成功。每条轨迹还保留相机/深度数据、
`episode.mp4`、`rollout.json`、原生对话和 `api_trace/`，包括实际请求、原始图片字节
与完整模型响应。遇到异常先看 `result.json`、`env_server.log`、`sam3_server.log`
（episode 自己启动 SAM3 时）及 transcript 中最后一次工具结果。

需要可审计、可恢复的批量采集时，先从每 GPU 一个驻留环境开始：

```bash
python tools/rl/run_batch.py run "$RPENT_DATA_ROOT/rollouts/libero-dev10" \
  --jobs tools/rl/jobs/libero-dev10.json --gpus 1 --per-gpu 1 \
  --max-initializing-per-gpu 1 --workflow episode
python tools/rl/run_batch.py summary "$RPENT_DATA_ROOT/rollouts/libero-dev10"
```

ZAP 使用仅含 LIBERO 的清单，并改为 `--workflow zap_turn`。重试、审计、断点恢复、
[离线安装](REFERENCE_zh.md#离线安装)、[远程仿真器](REFERENCE_zh.md#将-planner-与仿真器部署在不同机器)
和 [RLinf/SFT 集成](REFERENCE_zh.md#rlinf-与-sft) 见参考文档。

## 代码结构

| 模块 | 职责 |
| --- | --- |
| `robots/rl/robot_spec.py` | CLI、配置、服务生命周期、planner 选择 |
| `robots/rl/prompt_bundle.py`、`prompts/*.md` | 支持组合与可复用的机器人/工作流约定 |
| `robots/rl/toolkit.py` | 工具注册、原生任务文本、episode 产物 |
| `robots/rl/tools.py`、`geometry.py` | LIBERO 动作原语及共享视觉几何 |
| `robots/rl/backends.py`、`server.py` | RoboCasa/RoboTwin 动作适配与隔离原生服务 |
| `robots/rl/workflow.py`、`snapshot.py`、`images.py` | 公共循环的上下文策略、可见 todo/感知和显示坐标 |
| `robots/rl/remote.py`、`remote_manager.py` | 远程 episode worker 与文件传输 |
| `robots/rl/rl_episode.py` | RL trainer 的单 episode 接口 |
| `tools/rl/run.sh`、`run_batch.py` | 单条轨迹及可审计批量启动器 |

原生仿真 facade 保留在 `robots/libero`、`robots/robocasa`、`robots/robotwin`，
共享工作流复用它们，而不加载各自的 VLA toolkit。扩展新仿真器时增加原生适配、
机器人约定和明确的工作流支持，无需再复制一套 planner。CPU 回归测试位于
`tests/unit_tests/robots/rl`。
