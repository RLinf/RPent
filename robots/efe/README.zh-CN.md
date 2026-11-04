# EFE 兼容评测

[English](README.md) | [简体中文](README.zh-CN.md)

状态：实验性，**EFE-compatible / non-official**。Harbor `0.23.0` 是 Adapter 支持的宿主版本，需要 Python 3.12。

RPent 通过 Harbor 的 Installed Agent 运行 EFE（Embodied First Exam）任务。Harbor 启动 Agent 容器（`main`）和 Simulator 容器（`sim`）；RPent 的 `CodexPlanner` 通过任务自带的 `rcb_client` 服务操作机器人，EFE 的 Verifier 独立评分。

- 支持：1150 个 `standard` 和 `standard-realtime` 任务。
- 暂不支持：959 个 `privileged` 任务，其交付物是可重放的 `trajectory.npz` 或策略程序/模型，而不是逐步控制。

安装、运行命令、工具集和结果位置：
<https://rpent.readthedocs.io/zh-cn/latest/rst_source/simulators/efe.html>

代码：`harbor_agent.py`（Harbor Adapter）、`runner.py`、`cli.py`；机器人工具在 `robots/efe/`。
