# EFE 兼容评测

[English](README.md) | [简体中文](README.zh-CN.md)

状态：实验性，**EFE-compatible / non-official**。
Harbor `0.23.0` 是 Adapter 支持的宿主版本。它要求 Python 3.12。
安装后的 RPent Runner 也使用 Python 3.12。它不安装 Robot extras。

## 范围

RPent 在 Harbor 的 Agent Container 内运行现有 `CodexPlanner`，使用 RPent 的
Prompt bundle、可加载的 Memory 语料、RPent 共享文件与 Memory 工具及其授权规则，
并支持多 Session 交接。EFE 未提供 RobotSpec，因此该 Session 不加载机器人原语工具、
VLA 或 Robot Runtime。Agent 仍只能通过任务公开的 Robot Client 操作 Simulator。

Harbor 管理隔离和执行预算。EFE Verifier 独立评分。

`finish(status="completed")` 表示 Agent 完成了自己的工作。
它不表示任务成功，也不产生 Reward。CLI 的退出码只报告 Runner 是否出错。
第一阶段不支持在线 RL、Pi0.5、VLA 或修改 EFE 任务及评分。

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
CPU-only Simulator 不要求 GPU。首次运行仍需下载容器和任务资源。
按 EFE 的安装说明准备宿主及 `tasks/autobio/autobio-pickup-i00-standard/`。
该任务声明使用 CPU 物理模拟和 OSMesa 渲染。
其 Agent 容器使用 Python 3.12.14，工作目录为 `/app`，Client 位于 `/opt/tools`。
这些配置来自源码核对。真实运行状态见“验证状态”。
保留 EFE 原始任务、Simulator、Verifier 和资源限制。

Adapter 从当前 checkout 打包 `rpent/`、`robots/` 的 Python 源码、
`pyproject.toml` 声明的 package data 和安装元数据。
它将源码上传到容器的 `/opt/rpent-efe/source`，再用任务自带的 Python 创建 `venv`。
选定任务提供 Python 3.12。容器不需要预装 `uv`。
它不会将宿主的 `.git`、测试、模型凭据文件或运行日志放入源码包。
`robots/` 是现有 RPent distribution 所需的包，不会启动任何 Robot Runtime。
宿主的 `PYTHONPATH` 必须能导入本仓库中的 Adapter。
`source_root` 默认指向 Adapter 所在 checkout 的仓库根目录。
若 Adapter 从已安装的 Wheel 导入，必须显式传入 `source_root`。

```bash
export RPENT_ROOT=/absolute/path/to/RPent
export EFE_ROOT=/absolute/path/to/embodied-first-exam
export EFE_TASK="$EFE_ROOT/tasks/autobio/autobio-pickup-i00-standard"
export EFE_MODEL=openai/your-model
export OPENAI_API_KEY=your-key
```

不要将凭据写入仓库。Adapter 仅转发模型相关环境变量和显式的 Harbor `--agent-env`。
它将 `OPENAI_API_KEY`、`OPENAI_BASE_URL` 映射到 RPent 的 `CODEX_API_KEY`、`CODEX_BASE_URL`。
已提供的 `CODEX_*` 设置优先。
Adapter 不转发宿主的 `CODEX_HOME`。Runner 在 Agent 容器内为每次 Trial 新建该目录。

## P0：原生基线

先按 EFE 官方运行说明验证相同原始任务的原生 Codex 路径。
以下命令使用 Harbor 原生 Agent；它不加载 RPent：

```bash
uv run --project "$EFE_ROOT" harbor run \
  --path "$EFE_TASK" --agent codex --model "$EFE_MODEL" \
  --n-concurrent 1 --n-attempts 1 \
  --jobs-dir "$RPENT_ROOT/logs/efe-native"
```

检查 Trial 的 `result.json` 和 Verifier 输出。
容器启动、Simulator 交互和 Verifier 运行都需要真实证据。
原生 Agent 的正常退出不能替代任务成功结果。

## P1：RPent Installed Agent [推荐]

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
选定任务的原始 Agent 时间限制是 3600 秒，Adapter 默认使用 3500 秒，
以便 RPent 在 Harbor 取消 Agent 之前写出 `session.json`。
`timeout_s` 必须小于任务的 Agent 超时；两者相等时 Harbor 先超时，报告会丢失。
为对照实验使用相同的实际时间预算。
`max_turns` 统计现有 Planner 的模型响应次数，不等于 Shell 命令数。

替代入口用于已经准备好的 Agent Container 内部诊断：

```bash
rpent-efe --instruction-file /logs/agent/rpent-instruction.txt \
  --workdir /app --output-dir /logs/agent/manual-rpent \
  --model your-model --timeout-s 600 --max-turns 200
```

直接调用 CLI 不会创建 Harbor 容器，也不会启动 Simulator 或 Verifier。
完整评测使用 Installed Agent 入口。

## P2：先验经验与探索

加入 `memory_source` 以加载已有的 RPent 语料，加入 `explore_sessions` 以运行多个独立 Session。

```bash
PYTHONPATH="$RPENT_ROOT${PYTHONPATH:+:$PYTHONPATH}" \
uv run --project "$EFE_ROOT" harbor run \
  --path "$EFE_TASK" \
  --agent rpent.benchmark.efe.harbor_agent:RPentEfeAgent \
  --model "$EFE_MODEL" --n-concurrent 1 --n-attempts 1 \
  --agent-kwarg source_root="$RPENT_ROOT" \
  --agent-kwarg timeout_s=3500 --agent-kwarg max_turns=200 \
  --agent-kwarg reasoning_effort=none \
  --agent-kwarg memory_source="$RPENT_ROOT/memory/efe" \
  --agent-kwarg explore_sessions=3 \
  --jobs-dir "$RPENT_ROOT/logs/efe-rpent-exp"
```

`memory_source` 是宿主目录，Adapter 在运行前上传到 `/opt/rpent-efe/memory`。
省略它时，Session 从空 Memory 根目录开始。
RPent 文件工具在 `explore_sessions` 为 1 时把发布语料视为只读；
大于 1 时只允许写入当前 inbox。Codex 原生 shell 工具不受该策略约束，
但每次 Trial 都重新上传 `memory_source`，因此容器内写入不会回写到宿主。

`explore_sessions` 顺序运行独立的 Planner Session。每个 Session 使用新的 Codex 上下文，
并读取前序 Session 的笔记。`timeout_s` 是所有 Session 的总预算，每个 Session 获得剩余时间。
`max_turns` 按 Session 分别计数。

EFE Verifier 在 Agent Container 之外运行，因此没有 Session 会被提升进语料。
先验经验只来自 `memory_source`。

## P3：记录和对照

Harbor 同步每个 Trial 的 Agent 日志。RPent 的产物位于其 `agent/rpent/` 下：

| 文件 | 内容 |
| --- | --- |
| `session.json` | 原始任务、配置、Memory 模式、版本、各 Session 统计、Agent 自述、usage、运行时间和错误 |
| `codex.txt` | Planner 的可读记录 |
| `codex.txt.stream.jsonl` | 原始 Codex SDK 事件，包括原生工具调用 |
| `codex.txt.last` | 最后回复；启动失败时可能不存在 |
| `run.log` | RPent 运行日志 |

Agent 日志根目录的 `rpent-source.json` 保存上传源码包的 SHA-256。
如果 Harbor 在报告写入前终止 Agent，Adapter 会汇总每个 Session 轨迹中最后一个 usage 事件。
全零事件视为未观测，而不是已测量的用量。
此时 `usage-recovery.json` 标记 `complete=false`。这些计数只覆盖已写入轨迹的响应。
这些日志不包含 RPent 生成的最终 Reward。
Harbor 的 Trial `result.json` 保留 EFE Verifier 的评分和错误分类。

比较时固定任务及资源 revision、模型、endpoint、Codex 版本、reasoning effort、时间限制和重试策略。

| 配置 | 原生 `--agent codex` | RPent |
| --- | --- | --- |
| Codex 二进制 | 从 npm 安装 `@openai/codex@latest`，除非用 `--agent-kwarg version=` 固定 | 使用随包安装的 `openai-codex-cli-bin`；用 `--agent-env CODEX_BIN=...` 传入覆盖值 |
| Reasoning effort | 未设置，使用 Codex 或模型默认值 | `none` |
| System prompt | 无；Instruction 直接作为 user prompt | RPent system prompt |
| 项目文档与 Skills | Codex 默认值 | `project_doc_max_bytes=0`；禁用仓库 Skills |
| Context window 与压缩 | Codex 默认值 | 默认同 Codex；仅当设置 `CODEX_MODEL_CONTEXT_WINDOW` 或 `CODEX_AUTO_COMPACT_TOKEN_LIMIT` 时覆盖 |
| Turn 上限 | 无 | 每个 Session 的 `max_turns`，默认 200 |
| 时间限制 | 任务 Agent timeout | `timeout_s`，Adapter 默认 3500 |
| Sandbox 与审批 | `--dangerously-bypass-approvals-and-sandbox` | SDK `full_access` 与 `deny_all` 审批 |

原生 Agent 应加上 `--agent-kwarg reasoning_effort=none` 和
`--agent-kwarg version=<codex-version>`。RPent CLI 版本从 `session.json` 的
`versions` 读取。不要假设两边安装版本一致。
同时保留两边完整 Trial 结果和日志。不要只比较 `finish` 文本。

| 指标 | 读取与解释 |
| --- | --- |
| 任务结果 | 使用 Verifier Reward；另列基础设施或 Agent 错误 |
| Token | 比较 Harbor Context 中 input/cache/output；input 已包含 cache |
| 时间 | 分别比较 Agent 执行时间和包含安装的 Trial 总时间 |
| 工具行为 | 读取原始轨迹，区分 Shell、文件操作和 Robot Client 请求 |
| 成本 | 当前 Adapter 不估算 `cost_usd`；缺少测量时保留空值 |

Adapter 将实测 Token 总量回填 Harbor `AgentContext` 和 `model_usage`。
当前交付的是原始 SDK 轨迹，不是 ATIF 转换器。
在真实实验完成前，不声称 RPent 优于原生 Codex。

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

## 验证状态

2026-11-02 在 WSL Docker `29.9.2` 上运行了 1 次 Oracle、1 次原生 Codex
和 1 次 RPent Trial。三次 Verifier 均返回 `success=1`。
P0 与 P2 使用本地模型代理和 dummy key，不是公开 API key。
每条路径只运行一次，不构成统计对照。
RPent Trial 的 `memory/` 为空，`codex-home/` 中没有 `auth.json`。
这些 Trial 覆盖了 Harbor Adapter、源码上传和 usage 恢复。
RPent Trial 运行早于 Adapter 加载 RPent Prompt 与共享文件工具，
因此不覆盖当前的 Agent 配置。

由于 BuildKit 无法通过该宿主代理 clone，`sim` 镜像改用固定 AutoBio commit
在本地构建。任务与评分文件未修改。

## 参考

- [EFE setup and execution](https://github.com/mll-lab-nu/embodied-first-exam/blob/149a3836999cd0f4596b579e5e1a77bd28f3c478/docs/protocol/running.md)
- [Selected CPU task and limits](https://github.com/mll-lab-nu/embodied-first-exam/blob/149a3836999cd0f4596b579e5e1a77bd28f3c478/tasks/autobio/autobio-pickup-i00-standard/task.toml)
- [Harbor custom agent lifecycle](https://github.com/harbor-framework/harbor/blob/1e5c5c6db929a10a140d05e606882c671ae20729/docs-mintlify/agents/custom-agents.mdx)
- [Harbor usage fields](https://github.com/harbor-framework/harbor/blob/1e5c5c6db929a10a140d05e606882c671ae20729/src/harbor/models/agent/context.py)
