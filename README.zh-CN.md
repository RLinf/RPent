<div align="center">
  <h1>RPent</h1>
  <p><i>LLM 负责推理、VLA 负责执行，在仿真中形成闭环的具身智能体框架。</i></p>
</div>

<div align="center">

[![English](https://img.shields.io/badge/lang-English-blue.svg)](README.md)
[![简体中文](https://img.shields.io/badge/语言-简体中文-red.svg)](README.zh-CN.md)
[![GitHub](https://img.shields.io/badge/GitHub-RPent-181717?logo=github)](https://github.com/RLinf/RPent)

</div>

RPent 是一个把大语言模型放进「决策回路」的**具身智能体框架**。大模型负责高层推理并调用工具；一个视觉-语言-动作（VLA）策略——如 **Pi0.5** 或 **RLDX-1**——负责底层动作执行；仿真环境（**LIBERO** 或 **RoboCasa**）返回观测与渲染画面，闭合整个回路。推理、执行、仿真各自运行在独立进程中，重量级的 GPU 模型与物理引擎不会争抢同一个 Python 解释器。

<div align="center">
  <img src="docs/architecture.svg" alt="RPent 架构" width="960"/>
</div>

## 核心特性

- **LLM-in-the-loop 控制。** 大模型无需微调——它完全通过调用工具（`pi0_pick`、`move_to`、`rotate_wrist`、`back_project`、`finish` …）来驱动机器人。每次工具调用的结果都以多模态上下文（文本 + 渲染图像）回灌，让模型基于「它实际看到的画面」进行推理。
- **三进程架构。** **Agent 主进程**（LLM 决策大脑 + 工具容器，不加载 `torch`）、**env_server**（仿真器 + EGL 渲染）、**vla_server**（GPU 策略权重）彼此独立，用轻量 RPC 连接。两个重量级进程都可以独立重启、切换到另一块 GPU，或指向远程主机。
- **可插拔的决策大脑（planner）。** 用一个参数 `--planner {api, claude_code, codex}` 切换决策大脑，无需改动工具或提示词：
  - `api` —— 基于 [pydantic-ai](https://ai.pydantic.dev/) 的、与厂商无关的工具调用循环（Anthropic / OpenAI / OpenAI 兼容），带提示词缓存与历史图像裁剪。
  - `claude_code` —— 使用 [Claude Agent SDK](https://docs.claude.com/en/api/agent-sdk/overview)，把工具容器包装成进程内 MCP server。
  - `codex` —— 使用 OpenAI Codex SDK，通过 HTTP MCP server 桥接工具。
- **两套环境、两个 VLA、同一套契约。** LIBERO（Pi0.5 走 HTTP）与 RoboCasa（RLDX-1 走 socket-RPC）共用完全相同的 env/vla 进程拆分；仅传输编解码不同，按各自观测数据的形状而定。
- **实时 Dashboard。** 可选的 `--dashboard` 会启动一个本地 FastAPI 监控页，实时推送智能体的推理流、实时摄像头 / Pi0 视角画面、动作时间线与片段回放——并提供**中英双语界面**（`--dashboard-language {en, zh-cn}`）。
- **在磁盘上放一个包即可接入新环境。** 无需修改中心注册表——参见[接入新环境](https://rpent.readthedocs.io/zh-cn/latest/rst_source/extending/new_env.html)。

## 工作原理

一次运行就是一个 **LLM-in-the-loop** 循环：

1. 大模型对任务进行推理，并调用某个工具（例如 `pi0_pick`）。
2. 该工具的**原语驱动**向 `vla_server` 请求一段动作块（`predict` / `vla_infer`）。
3. `env_server` 执行这段动作块（LIBERO 用 `chunk_step`，RoboCasa 逐步 `step`）。
4. 环境渲染出新的观测与摄像头画面。
5. 结果被转成「文本 + 图像」内容块，回灌给大模型进入下一轮。

当大模型调用 `finish` 工具（`success` / `failure` / `stuck`），或达到 `--max-turns` / `--max-episode-steps` 上限时，循环结束。

## 支持的环境

<table style="width: 100%; table-layout: auto; border-collapse: collapse;">
  <thead align="center" valign="bottom">
    <tr>
      <th style="text-align: left;">仿真环境</th>
      <th>VLA 策略</th>
      <th>决策大脑</th>
    </tr>
  </thead>
  <tbody valign="top">
    <tr>
      <td style="text-align: left; padding-left: 8px;">
        <ul style="margin-left: 0; padding-left: 16px;">
          <li><b>LIBERO</b>（standard / pro / plus）✅</li>
          <ul>
            <li>libero_object · _task / _swap / _lan</li>
            <li>libero_goal · _task / _swap / _lan</li>
            <li>libero_spatial · _task / _lan</li>
            <li>libero_10 · _task / _swap / _lan</li>
          </ul>
          <li><b>RoboCasa</b>（厨房长程任务）✅</li>
          <ul>
            <li>PickPlace* · Open/Close* · TurnOn/Off* …</li>
          </ul>
        </ul>
      </td>
      <td>
        <ul style="margin-left: 0; padding-left: 16px;">
          <li><b>Pi0.5</b>（LIBERO，HTTP）✅</li>
          <li><b>RLDX-1</b>（RoboCasa，socket-RPC）✅</li>
        </ul>
      </td>
      <td>
        <ul style="margin-left: 0; padding-left: 16px;">
          <li><b>api</b> —— pydantic-ai ✅</li>
          <ul>
            <li>Anthropic（Claude）✅</li>
            <li>OpenAI（responses）✅</li>
            <li>OpenAI 兼容（chat）✅</li>
          </ul>
          <li><b>claude_code</b> —— Claude Agent SDK ✅</li>
          <li><b>codex</b> —— OpenAI Codex SDK ✅</li>
        </ul>
      </td>
    </tr>
  </tbody>
</table>

## 快速开始

**1. 用一条 `pip install` 安装 RPent。**

```bash
git clone https://github.com/RLinf/RPent rpent && cd rpent
pip install -e ".[full]"
```

`.[full]` 是默认的端到端组合（openpi Pi0.5 VLA + LIBERO-PRO 仿真器，运行在 RLinf 运行时之上）。如果不需要整套，可选择更小的 extra：

| Extra | 安装内容 |
| --- | --- |
| `.[full]` | `rlinf` + `openpi` + `libero-pro`——默认运行组合 |
| `.[libero-pro]` | 仅基础 LIBERO + LIBERO-PRO 仿真器 |
| `.[libero-plus]` | 基础 LIBERO + LIBERO-plus 仿真器 |
| `.[libero]` | 仅基础 LIBERO |
| `.[openpi]` | 仅 openpi VLA |
| `.[rlinf]` | 仅 RLinf 运行时 |

**2. 配置密钥与 checkpoint，然后运行。**

```bash
# 大模型 API 密钥（api 决策大脑）
export ANTHROPIC_BASE_URL=https://xxx
export ANTHROPIC_API_KEY=sk-xxx
export OPENAI_BASE_URL=https://xxx
export OPENAI_API_KEY=sk-xxx

# VLA checkpoint —— 从以下地址下载：
# https://huggingface.co/datasets/RLinf/rlinf-pi05-libero-130-fullshot-sft
export PI05_CHECKPOINT_PATH=/path/to/rlinf-pi05-libero-130-fullshot-sft
export LIBERO_TYPE=pro
export CUDA_VISIBLE_DEVICES=0

# 运行一个任务：libero_object_swap，task 2，seed 0，使用 api 决策大脑、
# 一个 Anthropic 模型，最大输出 8192 token。
#   • OpenAI 兼容 chat 端点：  --model openai-chat:glm-5.2
#   • OpenAI responses 端点：  --model openai:gpt-5.5
#   • claude_code / codex 大脑：无需 provider 前缀，如 --model claude-opus-4-8
rpent --env libero --suite libero_object_swap --task 2 --seed 0 \
  --planner api --model anthropic:claude-opus-4-8 --max-tokens 8192
```

### 实时 Dashboard

加上 `--dashboard` 即可为本次运行打开一个浏览器监控页。它会先展示一个启动屏让你选择配置，然后实时推送推理流、实时画面与动作时间线。用 `--dashboard-language zh-cn` 切换到中文界面。

```bash
rpent --env libero --dashboard --dashboard-language zh-cn \
  --suite libero_goal_task --task 1 --seed 0 --planner claude_code
```

## 主要命令行参数

| 参数 | 默认值 | 说明 |
| --- | --- | --- |
| `--env` | —（必填） | 环境后端。当前支持 `libero`。 |
| `--suite` | —（必填） | 任务集，如 `libero_object_task`、`libero_spatial_swap` |
| `--task` | —（必填） | 任务集内的任务编号 |
| `--seed` | `0` | 随机种子 |
| `--planner` | `api` | 决策大脑：`api` \| `claude_code` \| `codex` |
| `--model` | — | 模型 id；`api` 需带 provider 前缀（`anthropic:…`、`openai:…`、`openai-chat:…`） |
| `--max-turns` | `100` | 智能体最大轮数 |
| `--max-tokens` | `8192` | 单次回复最大 token |
| `--max-episode-steps` | `10000` | 环境最大步数 |
| `--libero-type` | `LIBERO_TYPE` 或 `pro` | LIBERO 类型：`standard` \| `pro` \| `plus` |
| `--cuda-device` | 继承当前环境 | env / vla server 可见的 GPU 设备 |
| `--dashboard` | 关 | 为本次运行启动本地 dashboard |
| `--dashboard-language` | `en` | Dashboard 界面语言：`en` \| `zh-cn` |
| `--env-endpoint` | —（新起进程） | 已在运行的 env_server 的 `[protocol://]host:port`（`protocol=http\|socket`，默认 `http`）。留空则本地起一个。 |
| `--vla-endpoint` | —（新起进程） | 已在运行的 vla_server 的 `[protocol://]host:port`（同上）。留空则本地起一个。 |

## 文档

- [接入新环境](https://rpent.readthedocs.io/zh-cn/latest/rst_source/extending/new_env.html) —— 把新的仿真器 / 机器人接入 runner（[English](https://rpent.readthedocs.io/en/latest/rst_source/extending/new_env.html)）。
- [`docs/`](docs/README.md) —— 本地 Sphinx 构建与预览说明。

## 致谢

RPent 构建于 [RLinf](https://github.com/RLinf/RLinf) 的仿真器、VLA 模型与训练基础设施之上，也得益于更广泛开源社区的 agent SDK —— [pydantic-ai](https://ai.pydantic.dev/)、[Claude Agent SDK](https://docs.claude.com/en/api/agent-sdk/overview) 与 OpenAI Codex SDK。感谢 LIBERO、RoboCasa、robosuite、MuJoCo、openpi 背后的团队。
