# BEHAVIOR

本接入让标准 RPent Planner 和 Toolkit 能够在官方 BEHAVIOR-1K 测评环境中运行。
Planner 负责选择工具；官方 evaluator 接收机器人动作、推进仿真，并负责重置、终止和评分。

[English](README.md)

## 支持范围

- 每个 Env 服务连接一个 R1Pro 环境和一个官方 evaluator 会话。
- 支持挑战赛的 100 个任务 ID（`0..99`），启动时须提供匹配的完整 main-task 指令。
  能识别任务 ID 不代表已在该任务上取得成功。
- 观测按 `head`、`left_wrist`、`right_wrist` 顺序提供三视角 RGB 和公开机器人状态；
  几何工具还需要对应的深度与相机标定。
- 原生动作有 23 个通道。RLinf Pi0.5 预设使用 `abs_joint`、BF16、5 步去噪和
  32 动作的 chunk，运行期间保持完整 main-task 指令不变。
- 各服务分别启动，RPent CLI 连接并借用它们。结束 Planner 会关闭客户端连接，
  不会关闭外部服务。本接入尚不支持 Dashboard 启动和多个 evaluator 会话。

## 安装与配置

按官方 [BEHAVIOR-1K 安装说明](https://github.com/StanfordVL/BEHAVIOR-1K)
准备仿真器、挑战赛资源和 evaluator。仿真环境应与 Planner、模型环境分开。

在 RPent 源码目录内安装轻量的 Planner 和 Env 服务依赖：

```bash
python -m pip install -e '.[behavior]'
```

VLA 服务还需要支持 BEHAVIOR 的 RLinf 安装、模型依赖、checkpoint、归一化统计和
tokenizer 资源。在该模型环境中安装 `.[behavior-vla]`。RLinf 必须提供
`rlinf.models.embodiment.openpi_pytorch.get_model`、`pi05_behavior` 配置，以及
`predict_action_batch(..., mode="eval")` 接口。RPent 不会隐式下载权重或安装某个
特定 RLinf fork。图像变换和动作归一化由所安装的 RLinf 实现负责；如果 tokenizer
资源单独存放，在启动模型服务前按 RLinf 的要求设置 `OPENPI_DATA_HOME`。

完整 Agent 还需要 RPent 共享的 SAM3 服务（在独立环境中安装 `.[sam3]` 并准备权重），
以及用于 `move_to` 的 CuRobo v0.7.8 和兼容的 PyTorch/Warp/CUDA 环境。
仅启用 VLA 工具时不需要 SAM3 和 CuRobo。机器人发现和 `--help` 不会导入仿真器、
模型或 GPU 库。

在源码目录外选择一个新的运行目录。将以下路径替换为实际安装位置，并在各终端中
设置相同的变量：

```bash
export RPENT_REPO=/path/to/RPent
export RPENT_RUN=/path/to/behavior-runs/run-name
export VLA_PYTHON=/path/to/vla-env/bin/python
export ENV_PYTHON=/path/to/env-service-env/bin/python
export SIM_PYTHON=/path/to/simulator-env/bin/python
export PLANNER_PYTHON=/path/to/planner-env/bin/python
export VLA_CHECKPOINT=/path/to/checkpoint
export PLANNER_MODEL=provider:vision-model-name
export TASK_ID=40
export TASK_NAME=make_microwave_popcorn
export TASK_LANGUAGE='In the kitchen, take the popcorn bag from the countertop, put it into the microwave, and heat it until the popcorn is cooked so the cooked popcorn ends up inside the bag.'
cd "$RPENT_REPO"
```

这里的任务仅作启动示例。更换任务时，同时修改三个任务变量；[`tasks.py`](../tasks.py) 列出官方
ID 与名称的对应关系。提交到仓库的 YAML 不应包含主机地址、凭据、权重或运行结果。

## 运行一条轨迹

每条长期运行的命令使用一个独立终端。首次运行先使用仅启用 VLA 的工具配置，
因此以下流程不需要 SAM3 和 CuRobo。

**1. 启动 VLA 服务。** 该服务只预测动作，不推进仿真。

```bash
"$VLA_PYTHON" -m robots.behavior.vla_server \
  --checkpoint "$VLA_CHECKPOINT" --task-prompt "$TASK_LANGUAGE" \
  --host 127.0.0.1 --port 28011 --transport socket \
  --output-dir "$RPENT_RUN/vla"
```

**2. 模型就绪后，启动 Env 桥接服务。** RPent 内部 RPC 和官方策略 WebSocket
使用不同的端口。

```bash
"$ENV_PYTHON" -m robots.behavior.env_server \
  --task-id "$TASK_ID" --task-language "$TASK_LANGUAGE" \
  --vla-endpoint socket://127.0.0.1:28011 \
  --host 127.0.0.1 --port 28010 --transport socket \
  --policy-host 127.0.0.1 --policy-port 28012
```

**3. 启动官方 evaluator。** `public_test` 的下标 `0` 对应 instance `301`。
官方 evaluator 拥有仿真循环；任务测评保留其默认步数预算。桥接接口要求每一步交换
一次观测和动作，不使用 evaluator 客户端的动作 chunk 重放。

```bash
"$SIM_PYTHON" -m omnigibson.eval.eval \
  --task-name "$TASK_NAME" --mode public_test --instance-indices 0 \
  --num-rollouts 1 --num-envs 1 --host 127.0.0.1 --port 28012 \
  --write-video --video-fps 120 --output-dir "$RPENT_RUN/official"
```

**4. 首帧观测就绪后，启动标准 RPent CLI。** 按 RPent 的规划器配置方式准备支持图像
输入的模型和凭据，并设置 `PLANNER_MODEL`。模型服务商的地址不写入机器人模板。
这里的 `--model` 指 VLM Planner，不是 VLA 权重。

```bash
"$PLANNER_PYTHON" -m rpent.cli.main --robot behavior --planner api \
  --model "$PLANNER_MODEL" \
  --behavior-config robots/behavior/configs/vla_only.yaml \
  --task-id "$TASK_ID" --task-language "$TASK_LANGUAGE" \
  --env-endpoint socket://127.0.0.1:28010 \
  --vla-endpoint socket://127.0.0.1:28011 \
  --output-dir "$RPENT_RUN/agent"
```

`vla_only.yaml` 向 Planner 提供观测、VLA 和结束工具。VLM 仍负责选择何时调用它们，
因此该配置不能直接当作固定执行循环的纯 VLA baseline。

使用完整 `agent.yaml` 时，先用权重启动共享 SAM3 服务，为 Env 服务增加
`--motion-backend curobo`，为 Planner 命令增加
`--sam3-endpoint socket://127.0.0.1:28013`。SAM3 服务参数见
`python -m rpent.robots.components.sam3_server --help`。保持任务文字和模型权重一致。
轨迹结束后显式关闭分别启动的服务；运行下一条轨迹需要新的 Env 服务和输出目录。

## 工具与接口约定

`agent.yaml` 开放 12 个机器人工具。RPent 原有的文件／记忆工具仍然保留，
机器人的 allowlist 不会替换这些共享工具。

| 工具 | 用途 | 是否推进环境 |
| --- | --- | --- |
| `view_env_state` | 读取保存的三视角、实测状态和执行反馈 | 否 |
| `read_robot_state` | 读取保存记录中详细的关节实测值和速度 | 否 |
| `segment` | 通过文字或原图 `[行, 列]` 点进行 SAM3 分割 | 否 |
| `pixel_to_robot_base` | 将保存帧的像素／深度投影到该帧的机器人基座坐标系 | 否 |
| `vla_act` | 使用完整 main-task 指令预测并执行受限的动作 chunk | 是 |
| `set_gripper` | 控制选定夹爪，`val=0` 闭合、`val=1` 打开 | 是 |
| `move_to` | 用 CuRobo IK 求解，把单臂移动到附近位姿 | 是 |
| `move_eef_delta` | 通过 `move_to` 路径让单臂做小幅相对 xyz 位移 | 是 |
| `rotate_wrist` | 通过 `move_to` 路径让 EEF 绕机器人基座 Z 轴旋转 | 是 |
| `rotate_wrist_joint` | 按相对角度转动单个手腕关节（roll、pitch 或 yaw），不经过 IK | 是 |
| `move_base` | 执行受限的局部底盘速度动作，最后发送零底盘命令 | 是 |
| `finish` | 结束 Planner 会话并保存其判断 | 否 |

`release` 用于打开选定夹爪，可在 `tool_allowlist` 中按名称启用，不在
`agent.yaml` 内。

工具参数与限制定义在带类型标注的 `@tool` 函数中。所有工具使用原有
`ToolResult(data, images)` 返回结构。动作工具返回后，Toolkit 保存新观测，
并返回三视角、`state` 和 `log.result`。其中 `step` 是保存的观测记录编号，
不是环境动作数。已下发、已确认和未确认的动作分别计数；未知执行结果不补零，
超时的动作 chunk 不会自动重放。

几何坐标只对应所查询的保存帧，机器人移动后需要重新定位。`move_to`、
`move_eef_delta` 和 `rotate_wrist` 需要 Env 服务启用 CuRobo 运动后端。它们保持
躯干和另一臂不动，对目标位姿求解 IK 并对手臂关节插值；不对路径做场景碰撞检查，
也不提供通用抓取规划。无法保持目标朝向时，规划器只按位置重试，并在结果中报告
`orientation_relaxed` 及实际朝向误差。IK 失败时机器人不动，结果报告最近可达位置。
`move_base` 没有可靠里程计或到达保证。夹爪命令完成不代表已经抓住物体。工具成功和
Planner 的 `finish` 都不能代替官方评分。

`rotate_wrist(arm=..., delta_yaw_deg=..., settle_steps=0)` 让 EEF 绕机器人基座 Z 轴
旋转相对角度，范围 -180 至 180 度，可达性由 IK 决定。终点保持当前 EEF 的 xyz，
但中间路径及手指尖会扫过周围空间。它沿用 `move_to` 的静止规划起点；不支持转动
已抓住的物体或接触操作。

`rotate_wrist_joint(arm=..., joint=..., delta_deg=...)` 以线性关节斜坡把单个手腕
关节转动 -90 至 90 度的相对角度，然后保持最终目标。它不经过 IK，因此总会执行；
手的位置会绕手腕走一小段弧线。其余关节、躯干、底盘和双夹爪命令保持不变。结果报告
实测转动量，以及请求是否被关节限位截断。

手臂类工具调整左臂 7–13 或右臂 15–21 通道，底盘 0–2 为零速度，躯干、另一臂及
双夹爪命令保持。完成依据是动作确认和实测位姿误差，不要求末端速度为零。

## 经验库

Memory 是通过 RPent 共享的 `read_text_file` 和 `list_dir` 读取的历史经验，不是新的
机器人物理动作工具。用 `--memory-dir` 指定本地经验库目录；未指定时使用默认的
`memory/behavior/`。目录结构与其他接入一致：`MEMORY.md`、`global/`、`task-family/`。
Planner 先看初始观测，再读索引和最多三篇相关经验后行动。目标仍需根据当前观测
重新定位；历史经验不能证明当前坐标、抓取结果或官方任务成功。BEHAVIOR 评测以只读
方式打开经验库，不同步远程经验库。

## 文件结构与开发入口

主线是：**Planner 选择工具 → Toolkit 调用工具实现 → 生成动作 → Env 桥接层交给
官方 evaluator → 仿真执行并返回新观测 → Planner 再做决策。**

`primitives.py` 定义操作机器人本体的工具，`tools.py` 定义观测、感知、坐标查询
和会话工具，`toolkit.py` 统一注册并绑定运行资源。VLA 分支先请求模型预测动作，
手臂或夹爪分支按各自实现生成动作，最终都通过同一环境接口执行；感知和坐标查询
本身不产生动作。下面的目录树按真实文件位置列出入口：

```text
RPent/
├── rpent/                          原框架：CLI、Planner、会话与共享工具调度
├── robots/behavior/
│   ├── __init__.py                 机器人包入口
│   ├── tasks.py                    官方任务 ID 与名称
│   ├── robot_spec.py               配置、注册与外部服务连接
│   ├── guides/
│   │   ├── README.md               英文安装与使用说明
│   │   └── README.zh-CN.md         中文安装与开发导航
│   ├── configs/
│   │   ├── agent.yaml              完整工具配置
│   │   └── vla_only.yaml           Planner 仅调用 VLA 的工具配置
│   ├── prompts/evaluate.py         通用决策、工具选择与反馈规则
│   ├── prompt_bundle.py            按可用工具组装提示词
│   ├── toolkit.py                  工具注册、调用和观测保存
│   ├── tools.py                    观测、SAM3 分割及掩码／叠加图保存、坐标查询与 finish
│   ├── primitives.py               VLA、夹爪、手臂、底盘及动作计数
│   ├── motion.py                   环境侧 CuRobo IK、机器人映射和随包机器人模型
│   ├── assets/r1pro/               机器人模型、CuRobo 机器人配置和来源记录
│   ├── vla_client.py               请求 VLA 推理
│   ├── vla_server.py               VLA 服务入口
│   ├── vla_policy.py               RLinf 模型输入／输出适配
│   ├── env_client.py               Env 接口契约与数值请求兼容
│   ├── env_server.py               Env 服务入口
│   └── eval/evaluator.py           官方评测适配：通信、策略接口、动作／观测交接
└── tests/
    ├── unit_tests/robots/behavior/  离线接口与行为回归
    └── e2e_tests/behavior/          显式运行的真实服务检查
```

日常修改可按目标定位：

| 要改什么 | 优先阅读与修改 |
| --- | --- |
| 通用判断和工具选择策略 | `prompts/evaluate.py` |
| 哪些工具说明进入当前提示词 | `prompt_bundle.py` |
| 参数含义、单位、限制、返回结果说明 | 对应 `@tool` 函数的文档字符串 |
| 机器人本体动作 | `primitives.py` |
| 观测、分割和坐标计算 | `tools.py` |
| 工具注册与资源绑定 | `toolkit.py` |
| 环境侧 IK 求解和机器人模型 | `motion.py`、`assets/r1pro/` |
| 可用工具、任务或服务配置 | `configs/`、`robot_spec.py` |
| 公开观测解码 | `tools.py` 的 `snapshot_observation`、`extract_public_proprio` |
| 动作通信 | Env/VLA 对应服务模块 |

`primitives.py` 中的 `move_to` 通过 Env client 请求关节路径，再通过统一动作接口
执行。Env 服务持有 `motion.py` 中的 `R1ProMotionPlanner`：它加载机器人模型，用
CuRobo 对目标位姿求解 IK，再在各关节速度上限内对七个手臂关节做线性插值。它不对
路径做场景碰撞检查，仿真器的物理是接触的最终裁决。返回的关节路点供动作工具执行。
`motion.py` 本身不在 Toolkit 中注册。

`motion.py` 的来源是官方评测接口能提供什么。官方 evaluator 持有仿真器，只在策略
连接上交换观测、动作和分数，不向 Agent 侧提供可调用的手臂规划器或逆运动学服务。
因此"把手臂移动到指定位姿"这类工具必须在 Env 服务内部求解：需要机器人模型、
关节限位和机器人自碰撞球，才能把目标位姿变成关节路点。`R1ProMotionModel` 从随包
的 URDF 和 CuRobo 机器人配置中读取这些信息，`R1ProMotionPlanner` 用它们驱动
CuRobo。把控制器侧规划留在 `motion.py`，`primitives.py` 才能只保留"产出 23 维
动作"的工具。

`eval/evaluator.py` 包含完整的官方评测兼容层。文件内的三个类按消息从官方连接进入
RPent 工具的顺序分工：

| 类 | 职责 |
| --- | --- |
| `WebsocketPolicyServer` | 官方消息编解码与 WebSocket 连接 |
| `OfficialEvaluatorPolicy` | 实现单次官方评测会话的 `reset()` 和 `act(observation)` |
| `EvaluatorStepBridge` | 在官方评测与工具之间交接观测和待执行动作，通过下一帧观测确认动作已执行 |

官方 evaluator 负责推进仿真和计算分数。Env 服务的内部 RPC 接口与官方 policy
接口共享同一个桥接对象。`env_server.py` 创建桥接对象和策略对象，再调用
`create_official_policy_server()`；兼容层不另建一套仿真循环，通信依赖也只在
处理消息时加载。

任务目标保持动态输入。离线测试位于
`tests/unit_tests/robots/behavior/`；真实服务检查位于 `tests/e2e_tests/behavior/`，
具体见[测试指南](../../../tests/README.md)。日志、视频和 checkpoint 放在本包之外。

## 结果与排查

- `agent/`：RPent 对话、保存的观测和工具结果。
- `vla/`：模型设置、推理耗时和实际预测的动作 chunk。
- `official/`：官方 JSON 评分和视频，任务成功以此为准。

上述命令默认使用 120 FPS 编码视频；视频播放速度与机器人控制频率分别配置。
每次运行都需要新的输出目录。连接异常时，先检查服务地址和日志；
不要将 `ws://` 策略地址作为 `--env-endpoint`。发生超时或断连时，保留日志并核对
执行反馈，再决定如何启动下一条轨迹。
