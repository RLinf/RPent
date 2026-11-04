# EFE 兼容的工具驱动评测

[English](README.md) | [简体中文](README.zh-CN.md)

状态：实验性，**EFE-compatible / non-official**。
Harbor `0.23.0` 是 Adapter 支持的宿主版本。它要求 Python 3.12。
安装后的 RPent Runner 也使用 Python 3.12。它不安装 Robot extras。

## 范围

RPent 在 Harbor 的 Agent Container 内以该 Trial 唯一的 Agent 进程运行现有
`CodexPlanner`。`robots/efe` 把任务公开的 `rcb_client.Robot` 服务当作机器人，
并接入 RPent 的 Robot Registry。两组都调用 `efe_spec`、`efe_observe`、
`efe_status` 和 `efe_recover`。Raw 组另有 `efe_submit`，按能力 spec 中的
布局发送一个动作块。Skills 组改为 `move_to_xyz` 和 `set_gripper`：前者用
公开本体感受闭环求解 IK，后者在保持当前臂位姿的同时设置夹爪开口。
带 Reset 预算的任务还会得到 `efe_reset`；使用 agent-ended episode 的任务
还会得到 `efe_end_episode`。

Agent 只能通过该 Client 操作 Simulator。RPent 不启动 Simulator；仅当任务声明
Reset 预算且 Planner 请求时才 Reset；不读取特权状态，也不修改任务或评分。

Harbor 管理隔离和执行预算。EFE Verifier 独立评分。

`finish(status="completed")` 表示 Agent 完成了自己的工作。
它不表示任务成功，也不产生 Reward。`solved()` 报告服务自己的结束原因，
Verifier 仍然负责评分。CLI 的退出码只报告 Runner 是否出错。
第一阶段不支持在线 RL、Pi0.5、VLA 或修改 EFE 任务及评分。

## 任务与模式覆盖

EFE 共有 32 个 Benchmark 家族、2109 个任务目录，分为三种模式：1066 个
`standard`（`eai-standard/2.1`、`2.2`、`2.4`）、84 个 `standard-realtime`
（`eai-standard/2.3`）和 959 个 `privileged`。Adapter 从服务 spec 读取每个任务
的能力，没有针对具体任务的代码，因此覆盖全部 standard 与 standard-realtime
任务。standard-realtime 的时钟不会暂停，但服务在 Planner 思考期间保持上一次的
目标，所以同一套工具仍然适用。不支持 privileged：它与 Simulator 同容器，
交付物是轨迹或策略，而不是服务 Client。

## Raw 工具与 Skills

Raw 组直接暴露服务协议：Planner 读取能力 spec，选择关节目标，用 `efe_submit`
发送动作块。

Skills 组在同一 Client 之上以一层高级技能取代 `efe_submit`。`move_to_xyz`
把世界坐标目标解成 IK，提交短动作块，并重新读取本体感受直至工具坐标系到达
目标，因此 Planner 可以用笛卡尔空间推理，不必直接处理关节角。`set_gripper`
在保持当前臂位姿的同时设置夹爪开口。该组不提供关节级动作工具。

任务的动作布局不符合技能要求（运动学关节顺序加一个末尾夹爪列）时，工具集
不注册技能：保留 Raw `efe_submit` 路径，记录原因，并在 `session.json` 中
写入 `skills_error`。Prompt 只列出该 Session 实际注册的工具。因此两组的
差别只是可用工具集，模型、预算和任务完全一致。

## Trial 隔离

每次运行都建立独立状态：

- `memory/` 是该次 Trial 的空 `MemoryManager` 根目录。Runner 不下载或合并任何语料。
  上传的 `memory_source` 目录会替换该根目录。
- `codex-home/` 是该次 Trial 的 `CODEX_HOME`。Runner 不复制宿主的 History、Session 或认证文件。
- 模型与凭据只来自显式环境变量。Adapter 转发 `OPENAI_API_KEY`、`CODEX_API_KEY`、
  `OPENAI_BASE_URL`、`CODEX_BASE_URL` 和 Codex 预算变量。

没有语料时，测的是无历史经验下的 RPent 行为。
Runner 在自身进程中设置 `CODEX_HOME`。每个 Agent 进程只运行一个 Trial。

## 安装与入口

宿主需要 EFE 官方要求的 Docker 环境、云端模型凭据和 EFE checkout。
按 EFE 的说明准备宿主和任务；Simulator 的资源要求由任务决定。
首次运行仍需下载容器和任务资源。
保留 EFE 原始任务、Simulator、Verifier 和资源限制。

Adapter 从当前 checkout 打包 `rpent/`、`robots/` 的 Python 源码、
`pyproject.toml` 声明的 package data 和安装元数据。
它将源码上传到容器的 `/opt/rpent-efe/source`，再用任务自带的 Python 创建 `venv`。
任务镜像提供 Runner 使用的 Python。容器不需要预装 `uv`。
它不会将宿主的 `.git`、测试、模型凭据文件或运行日志放入源码包。
`robots/` 是现有 RPent distribution 所需的包，不会启动任何 Robot Runtime。
宿主的 `PYTHONPATH` 必须能导入本仓库中的 Adapter。
`source_root` 默认指向 Adapter 所在 checkout 的仓库根目录。
若 Adapter 从已安装的 Wheel 导入，必须显式传入 `source_root`。

```bash
export RPENT_ROOT=/absolute/path/to/RPent
export EFE_ROOT=/absolute/path/to/embodied-first-exam
export EFE_TASK="$EFE_ROOT/tasks/<family>/<task>"
export EFE_MODEL=openai/your-model
export OPENAI_API_KEY=your-key
```

不要将凭据写入仓库。Adapter 仅转发模型相关环境变量和显式的 Harbor `--agent-env`。
它将 `OPENAI_API_KEY`、`OPENAI_BASE_URL` 映射到 RPent 的 `CODEX_API_KEY`、`CODEX_BASE_URL`。
已提供的 `CODEX_*` 设置优先。
Adapter 不转发宿主的 `CODEX_HOME`。Runner 在 Agent 容器内为每次 Trial 新建该目录。

## Installed Agent 入口

```bash
PYTHONPATH="$RPENT_ROOT${PYTHONPATH:+:$PYTHONPATH}" \
uv run --project "$EFE_ROOT" harbor run \
  --path "$EFE_TASK" \
  --agent rpent.benchmark.efe.harbor_agent:RPentEfeAgent \
  --model "$EFE_MODEL" --n-concurrent 1 --n-attempts 1 \
  --agent-kwarg source_root="$RPENT_ROOT" \
  --agent-kwarg timeout_s=3500 --agent-kwarg max_turns=200 \
  --agent-kwarg reasoning_effort=none \
  --jobs-dir "$RPENT_ROOT/logs/efe-rpent"
```

默认工作目录是 `/app`。任务使用其它工作目录时，传入 `--agent-kwarg workdir=/path`。
Harbor 的任务预算始终是外层硬限制。`timeout_s` 是 RPent 内层限制。
它必须小于任务的 Agent 超时；两者相等时 Harbor 先超时，报告会丢失。
RPent 在 Trial 结束时写出 `session.json`，因此要留出余量。
`max_turns` 统计现有 Planner 的模型响应次数，不等于 Shell 命令数。

该自定义 Agent 不声明模型 Endpoint，Harbor 无法推断推理主机。
传入 `--allow-agent-host <model-host>`。显式名单会替换默认放行，
因此还需放行安装所需的包主机：
`--allow-agent-host pypi.org --allow-agent-host files.pythonhosted.org`。
首次容器安装运行 `pip`，Harbor 默认 Setup 超时为 360 秒；
冷安装较慢时加上 `--agent-setup-timeout-multiplier 4`。

替代入口用于已经准备好的 Agent Container 内部诊断：

```bash
rpent-efe --instruction-file /logs/agent/rpent-instruction.txt \
  --workdir /app --output-dir /logs/agent/manual-rpent \
  --model your-model --timeout-s 600 --max-turns 200
```

直接调用 CLI 不会创建 Harbor 容器，也不会启动 Simulator 或 Verifier。
完整评测使用 Installed Agent 入口。

## Raw 与 Skills 两组

两组使用同一个 Runner。`skills=false` 注册 Raw 工具；默认的 Skills 组用技能层
取代关节级提交工具。对照只改变工具集，不改变模型或预算。Runner 默认使用 Codex 的 `read-only` Sandbox：Shell 命令
无法访问 Simulator，因此只有 RPent 工具会作用于 Episode。传入
`--agent-kwarg sandbox=full-access` 可允许 Planner 直接编写脚本操作服务。
该模式测量 Agent 的自由行为；其 Shell 动作不计入 RPent 工具计数。
通用 `rpent --robot efe` 入口通过共享 Planner 构建，不传入 Sandbox 设置，
因此始终以 `full-access` 运行；需要让 Shell 无法访问 Simulator 时，
请使用 `rpent-efe` 或 Harbor Adapter。

```bash
# RPent-Raw
PYTHONPATH="$RPENT_ROOT${PYTHONPATH:+:$PYTHONPATH}" \
uv run --project "$EFE_ROOT" harbor run \
  --path "$EFE_TASK" \
  --agent rpent.benchmark.efe.harbor_agent:RPentEfeAgent \
  --model "$EFE_MODEL" --n-concurrent 1 --n-attempts 1 \
  --agent-kwarg source_root="$RPENT_ROOT" \
  --agent-kwarg timeout_s=3500 --agent-kwarg skills=false \
  --jobs-dir "$RPENT_ROOT/logs/efe-raw"

# RPent-Skills
PYTHONPATH="$RPENT_ROOT${PYTHONPATH:+:$PYTHONPATH}" \
uv run --project "$EFE_ROOT" harbor run \
  --path "$EFE_TASK" \
  --agent rpent.benchmark.efe.harbor_agent:RPentEfeAgent \
  --model "$EFE_MODEL" --n-concurrent 1 --n-attempts 1 \
  --agent-kwarg source_root="$RPENT_ROOT" \
  --jobs-dir "$RPENT_ROOT/logs/efe-skills"
```

用 Verifier Reward 与 RPent 计数汇总两组：

```bash
rpent-efe-report "$RPENT_ROOT/logs/efe-raw" "$RPENT_ROOT/logs/efe-skills"
```

报告读取 `result.json`（Verifier Reward、Token、总耗时）和
`agent/rpent/session.json`（工具调用、失败调用、技能调用、提交次数、控制步数）。
它按 `session.json` 的 `skills` 标记每组，因此两组可放在同一个运行根目录下；
分开 `--jobs-dir` 只是便于查看日志。`--json` 输出相同数据供脚本使用。
两组必须固定模型、端点、推理强度、任务版本、
时间上限和尝试次数。单次 Trial 不构成对照。

## 离线验证

```bash
pytest tests/unit_tests/rpent/benchmark/efe -q
pytest tests/integration/efe -q
```

基础离线测试不要求 Docker 或模型凭据。
Harbor 合约测试需要在 Python 3.12 环境安装 `harbor==0.23.0`。
真实 Harbor Smoke 需要显式提供任务和模型，见同目录测试中的 opt-in 条件。
该 Smoke 最多等待 4 小时，以包含容器构建、3500 秒 Agent 预算和任务的收尾回放。
离线通过仅证明接口、工具传输和产物契约，不证明 EFE 任务成功。
