<div align="center">
  <img src="https://github.com/RLinf/misc/raw/main/pic/rpent_logo.png" alt="RPent-logo" width="520"/>
</div>

<div align="center">
<a href="https://arxiv.org/abs/2607.08448"><img src="https://img.shields.io/badge/arXiv-Paper-red?logo=arxiv"></a>
<a href="https://huggingface.co/RLinf"><img src="https://img.shields.io/badge/HuggingFace-yellow?logo=huggingface&logoColor=white" alt="Hugging Face"></a>
<a href="https://rpent.readthedocs.io/en/latest/"><img src="https://img.shields.io/badge/Documentation-Purple?color=8A2BE2&logo=readthedocs"></a>
<a href="https://rpent.readthedocs.io/zh-cn/latest/"><img src="https://img.shields.io/badge/中文文档-red?logo=readthedocs"></a>
<a href="https://github.com/RLinf/misc/blob/main/pic/rpent_wechat.png?raw=true"><img src="https://img.shields.io/badge/微信-green?logo=wechat&amp"></a>
</div>

<div align="center">

[![English](https://img.shields.io/badge/lang-English-blue.svg)](README.md)
[![简体中文](https://img.shields.io/badge/语言-简体中文-red.svg)](README.zh-CN.md)

</div>

<h1 align="center">
  <sub>RPent: 面向物理世界的智能体基础设施</sub>
</h1>

**RPent (Recursive Physical Agent)** 是一个用于构建具身智能体的开放框架，使智能体能够通过与物理世界的递归交互持续演化。RPent 并不预设单一基础模型，而是提供一个递归智能体框架，将感知、推理、记忆、执行与自我演化等异构智能统一到一个物理智能体中。通过持续交互、反思与适应，RPent 使物理智能体能够获得新的能力，并超越其初始设计不断演进。

RPent 建立在三项核心设计原则之上：**服务化、标准化和可组合**。RPent 支持将能力部署为可复用服务，通过统一接口连接，并灵活组合成多样化的物理智能体。这些原则使 RPent 能够超越传统机器人控制框架，建立面向物理世界的智能体基础设施；在其中，智能不仅被部署，也被持续构建、扩展与演化。

<div align="center">
  <img src="https://github.com/RLinf/misc/raw/main/pic/rpent_framework.png" alt="RPent framework"/>
</div>

## 最新动态

- [2026/07] 🔥 RPent 首篇论文 [Harness VLA: Steering Frozen VLAs into Reliable Manipulation Primitives via Memory-Guided Agents](https://arxiv.org/abs/2607.08448) 发布。

## 功能矩阵

<table width="100%">
  <thead align="center" valign="bottom">
    <tr>
      <th width="26%">智能体规划器</th>
      <th width="28%">动作原语</th>
      <th width="26%" align="left">仿真环境</th>
      <th width="20%">真实世界</th>
    </tr>
  </thead>
  <tbody valign="top">
    <tr>
      <td>
        <ul style="margin-left: 0; padding-left: 16px;">
          <li>Claude Code ✅</li>
          <li>Codex ✅</li>
          <li>Custom Planner ✅</li>
        </ul>
      </td>
      <td>
        <ul style="margin-left: 0; padding-left: 16px;">
          <li><b>VLA</b></li>
          <ul>
            <li>Pi0.5 ✅</li>
            <li>RLDX-1</li>
          </ul>
          <li><b>WAM</b></li>
          <ul>
            <li>DreamZero</li>
          </ul>
        </ul>
      </td>
      <td style="text-align: left; padding-left: 8px;">
        <ul style="margin-left: 0; padding-left: 16px;">
          <li>LIBERO-PRO ✅</li>
          <li>RoboCasa </li>
        </ul>
      </td>
      <td>
        <ul style="margin-left: 0; padding-left: 16px;">
          <li>Franka</li>
          <li>SO-101</li>
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

`.[full]` 是默认的端到端组合（openpi Pi0.5 VLA + LIBERO-PRO 仿真器，运行在 RLinf 运行时之上）。
如果不需要整套，可选择更小的 extra：

| Extra | 安装内容 |
| --- | --- |
| `.[full]` | `rlinf` + `openpi` + `libero-pro` — 默认运行组合 |
| `.[libero-pro]` | 基础 LIBERO + LIBERO-PRO 仿真器 |
| `.[libero-plus]` | 基础 LIBERO + LIBERO-plus 仿真器 |
| `.[libero]` | 仅基础 LIBERO |
| `.[openpi]` | 仅 openpi VLA |
| `.[rlinf]` | 仅 RLinf 运行时 |

**2. 配置密钥与 checkpoint，然后运行。**

```bash
# 大模型 API 密钥（api 规划器）
export ANTHROPIC_BASE_URL=https://xxx
export ANTHROPIC_API_KEY=sk-xxx
export OPENAI_BASE_URL=https://xxx
export OPENAI_API_KEY=sk-xxx

# VLA checkpoint — 从以下地址下载
# https://huggingface.co/datasets/RLinf/rlinf-pi05-libero-130-fullshot-sft
export PI05_CHECKPOINT_PATH=/path/to/rlinf-pi05-libero-130-fullshot-sft
export LIBERO_TYPE=pro
export CUDA_VISIBLE_DEVICES=0

# 运行一个任务：libero_object_swap，task 2，seed 0，使用 api 规划器
# 和 Anthropic 模型，最大输出 8192 token。
#   • OpenAI-compatible chat 端点：      --model openai-chat:glm-5.2
#   • OpenAI responses 端点：            --model openai:gpt-5.5
#   • claude_code / codex 规划器：       不需要 provider 前缀，如 --model claude-opus-4-8
rpent --suite libero_object_swap --task 2 --seed 0 \
  --planner api --model anthropic:claude-opus-4-8 --max-tokens 8192
```

### 实时 Dashboard

加上 `--dashboard` 即可为本次运行打开一个浏览器监控页。它会先展示一个启动屏让你选择配置，然后实时推送推理流、实时画面与动作时间线。用 `--dashboard-language zh-cn` 切换到中文界面。

```bash
rpent --dashboard --dashboard-language zh-cn \
  --suite libero_goal_task --task 1 --seed 0 --planner claude_code
```

### RoboCasa

RoboCasa 使用独立入口与安装指南。

```bash
bash scripts/setup_robocasa.sh                                # 一次性安装
bash scripts/run_robocasa.sh PickPlaceCounterToCabinet 0 0    # <任务> <GPU> <种子>
```

完整的 RoboCasa365 + RLDX-1 部署流程见 [SETUP_ROBOCASA.zh.md](docs/SETUP_ROBOCASA.zh.md)。

## 主要命令行参数

| 参数 | 默认值 | 说明 |
| --- | --- | --- |
| `--suite` | —（必填） | 任务集，如 `libero_object_task`、`libero_spatial_swap` |
| `--task` | —（必填） | 任务集内的任务编号 |
| `--seed` | `0` | 随机种子 |
| `--planner` | `api` | 推理大脑：`api` \| `claude_code` \| `codex` |
| `--model` | — | 模型 id；`api` 需带 provider 前缀（`anthropic:…`、`openai:…`、`openai-chat:…`） |
| `--max-turns` | `100` | 智能体最大轮数 |
| `--max-tokens` | `8192` | 单次 LLM 回复最大 token |
| `--max-episode-steps` | `10000` | 环境最大步数 |
| `--libero-type` | `LIBERO_TYPE` 或 `pro` | LIBERO 类型：`standard` \| `pro` \| `plus` |
| `--cuda-device` | 继承当前环境 | env / vla server 可见的 GPU 设备 |
| `--dashboard` | 关 | 为本次运行启动本地 dashboard |
| `--dashboard-language` | `en` | Dashboard 界面语言：`en` \| `zh-cn` |
| `--vla-endpoint` | — | 复用已在运行的 vla_server，而非新起一个 |
| `--no-driver` | 关 | 连接已存在的 env_server / vla_server |

## 文档

- [接入新环境](https://rpent.readthedocs.io/zh-cn/latest/rst_source/extending/new_env.html) — 把新的仿真器 / 机器人接入 runner（[English](https://rpent.readthedocs.io/en/latest/rst_source/extending/new_env.html)）。
- [RoboCasa 安装](docs/SETUP_ROBOCASA.zh.md) — RoboCasa365 + RLDX-1 安装与运行指南。
- [`docs/`](docs/README.md) — 本地 Sphinx 构建与预览说明。

## 引用与致谢

如果 **RPent** 或 **Harness VLA** 对你的工作有帮助，请引用：

```bibtex
@article{zhang2026harnessvla,
  title={Harness VLA: Steering Frozen VLAs into Reliable Manipulation Primitives via Memory-Guided Agents},
  author={Zhang, Yixian and Zhang, Huanming and Gao, Feng and Li, Xiao and Liu, Zhihao and Zhu, Chunyang and Qiu, Jiaxing and Yan, Yuchen and Liu, Jiyuan and Tang, Wenhao and Fang, Zhengru and Nie, Yi and Wei, Changxu and Wang, Yu and Ding, Wenbo and Yu, Chao},
  journal={arXiv preprint arXiv:2607.08448},
  year={2026},
  url={https://arxiv.org/abs/2607.08448}
}
```

RPent 构建于 [RLinf](https://github.com/RLinf/RLinf) 的仿真器、VLA 模型与训练基础设施之上，也得益于更广泛开源社区的 agent SDK — [pydantic-ai](https://ai.pydantic.dev/)、[Claude Agent SDK](https://docs.claude.com/en/api/agent-sdk/overview) 与 OpenAI Codex SDK。感谢 LIBERO、RoboCasa、robosuite、MuJoCo、openpi 背后的团队。
