<div align="center">
  <img src="https://github.com/RLinf/misc/raw/main/pic/rpent_logo.png" alt="RPent-logo" width="520"/>
</div>

<div align="center">
<a href="https://arxiv.org/abs/2607.08448"><img src="https://img.shields.io/badge/arXiv-Paper-red?logo=arxiv"></a>
<a href="https://huggingface.co/RLinf"><img src="https://img.shields.io/badge/HuggingFace-yellow?logo=huggingface&logoColor=white" alt="Hugging Face"></a>
<a href="https://rpent.readthedocs.io/en/latest/"><img src="https://img.shields.io/badge/Documentation-Purple?color=8A2BE2&logo=readthedocs"></a>
<a href="https://rpent.readthedocs.io/zh-cn/latest/"><img src="https://img.shields.io/badge/中文文档-red?logo=readthedocs"></a>
<a href="https://github.com/RLinf/misc/blob/main/pic/wechat.jpg?raw=true"><img src="https://img.shields.io/badge/微信-green?logo=wechat&amp"></a>
</div>

<div align="center">

[![English](https://img.shields.io/badge/lang-English-blue.svg)](README.md)
[![简体中文](https://img.shields.io/badge/语言-简体中文-red.svg)](README.zh-CN.md)

</div>

<h1 align="center">
  <sub>RPent: 面向物理世界的智能体基础设施</sub>
</h1>

**RPent (Recursive Physical Agent)** 是一个用于构建具身智能体的开放框架，使智能体能够通过与物理世界的递归交互持续演化。RPent 并不预设单一基础模型，而是提供一个递归智能体框架，将感知、推理、记忆、执行与自我演化等异构智能统一到一个物理智能体中。通过持续交互、反思与适应，RPent 使物理智能体能够获得新的能力，并超越其初始设计不断演进。我们将 RPent 构建于**服务化**、**标准化**和**可组合**的设计原则之上，确保框架具备高度的可扩展性。

<div align="center">
  <img src="https://github.com/RLinf/misc/raw/main/pic/rpent_framework.png" alt="RPent framework"/>
</div>

## 适用用户

RPent 面向以下四类用户：

- **具身智能研究者**：希望在具身任务与基准上取得更高成功率，尤其是长程操作任务。RPent 基于记忆引导的智能体组合方案，能够显著提升任务成功率，超越单纯依赖冻结 VLA 的效果。
- **在线学习 / 强化学习研究者**：致力于研究可自我演化的具身智能体。RPent 提供的递归交互、反思与记忆蒸馏闭环，可作为在物理世界中开展持续学习与强化学习研究的现成基础。
- **机器人应用开发者**：希望将具身方案部署到真实机器人硬件上。RPent 的服务化、标准化架构以及可定制的智能体控制逻辑，能够提升真实场景下的任务成功率，并缩短从原型到生产部署的周期。
- **具身智能体的最终用户**：即上一类开发者所服务的用户。只需安装 RPent 及相应的真实机器人扩展，即可开箱即用地运行预定义任务，无需具备机器学习专业知识。

## 最新动态

- [2026/08] 🔥 支持 RoboCasa，使用 RLDX-1 作为操作模型。参见 [RoboCasa 安装与 Target50 指南](robots/robocasa/README.md)和 [完整中文文档](https://rpent.readthedocs.io/zh-cn/latest/rst_source/usage/robocasa.html)。
- [2026/08] 🔥 新增非推理（non-reasoning）模式，平均执行时间降低约 40%。
- [2026/08] 🔥 支持 LIBERO 探索模式。文档：[LIBERO 探索模式](https://rpent.readthedocs.io/zh-cn/latest/rst_source/usage/libero.html#memory)。
- [2026/08] 🔥 支持 RoboTwin，使用 LingBot-VLA 处理双臂操作任务。文档：[RoboTwin](https://rpent.readthedocs.io/zh-cn/latest/rst_source/usage/robotwin.html)。
- [2026/07] 🔥 RPent 首篇论文 [Harness VLA: Steering Frozen VLAs into Reliable Manipulation Primitives via Memory-Guided Agents](https://arxiv.org/abs/2607.08448) 发布。

## 功能矩阵

<table width="100%" style="width: 100%; table-layout: auto; border-collapse: collapse;">
  <thead align="center" valign="bottom">
    <tr>
      <th style="min-width: 300px;">智能体规划器</th>
      <th style="min-width: 340px;">动作原语</th>
      <th style="min-width: 300px; text-align: left;">仿真环境</th>
      <th style="min-width: 260px;">真实世界</th>
    </tr>
  </thead>
  <tbody valign="top">
    <tr>
      <td>
        <ul style="margin-left: 0; padding-left: 16px;">
          <li><a href="https://rpent.readthedocs.io/zh-cn/latest/rst_source/usage/configure_planner.html#claude-code-planner">Claude Code</a> ✅</li>
          <li><a href="https://rpent.readthedocs.io/zh-cn/latest/rst_source/usage/configure_planner.html#codex-planner">Codex</a> ✅</li>
          <li><a href="https://rpent.readthedocs.io/zh-cn/latest/rst_source/usage/configure_planner.html#planner">Custom Planner</a> ✅</li>
        </ul>
      </td>
      <td>
        <ul style="margin-left: 0; padding-left: 16px;">
          <li><b>VLA</b></li>
          <ul>
            <li><a href="https://rpent.readthedocs.io/zh-cn/latest/rst_source/usage/libero.html">Pi0.5</a> ✅</li>
            <li><a href="https://rpent.readthedocs.io/zh-cn/latest/rst_source/usage/robocasa.html">RLDX-1</a> ✅</li>
            <li><a href="https://rpent.readthedocs.io/zh-cn/latest/rst_source/usage/robotwin.html">LingBot-VLA</a> ✅</li>
          </ul>
          <li><b>WAM</b></li>
          <ul>
            <li>DreamZero</li>
          </ul>
        </ul>
      </td>
      <td style="text-align: left; padding-left: 8px;">
        <ul style="margin-left: 0; padding-left: 16px;">
          <li><a href="https://rpent.readthedocs.io/zh-cn/latest/rst_source/usage/libero.html">LIBERO-PRO</a> ✅</li>
          <li><a href="https://rpent.readthedocs.io/zh-cn/latest/rst_source/usage/robocasa.html">RoboCasa</a> ✅</li>
          <li><a href="https://rpent.readthedocs.io/zh-cn/latest/rst_source/usage/robotwin.html">RoboTwin</a> ✅</li>
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

**1. 选择一个环境并安装 RPent。**

```bash
git clone https://github.com/RLinf/RPent rpent && cd rpent
# 默认推荐（LIBERO-PRO）：
pip install -e ".[libero-pro]"

# 其他环境配置：
pip install -e ".[robocasa]"    # RoboCasa
pip install -e ".[robotwin]"    # RoboTwin
```

`.[libero-pro]` 是默认推荐配置。其他环境见
[安装文档](https://rpent.readthedocs.io/zh-cn/latest/rst_source/installation.html)。

RoboCasa 安装、任务 memory 与 Target50 协议参见
[RoboCasa 指南](robots/robocasa/README.md)。

下面的示例继续使用 LIBERO-PRO。

**2. 下载 LIBERO-PRO 仿真资产。**

```bash
liberopro-download-assets --skip-existing
```

> 💡 访问 Hugging Face 较慢时，可走镜像加速：`HF_ENDPOINT=https://hf-mirror.com liberopro-download-assets --skip-existing`。

其他仿真器见[安装文档](https://rpent.readthedocs.io/zh-cn/latest/rst_source/installation.html)。

**3. 配置密钥与 checkpoint，然后运行。**

```bash
# Anthropic 密钥；使用官方端点时无需 export base url。
export ANTHROPIC_BASE_URL=https://xxx
export ANTHROPIC_API_KEY=sk-xxx

# VLA checkpoint —— 从以下地址下载：
# https://huggingface.co/RLinf/RLinf-Pi05-LIBERO-130-fullshot-SFT
hf download RLinf/RLinf-Pi05-LIBERO-130-fullshot-SFT \
  --exclude optimizer.pt \
  --local-dir ./checkpoints/RLinf-Pi05-LIBERO-130-fullshot-SFT

export PI05_CHECKPOINT_PATH=$PWD/checkpoints/RLinf-Pi05-LIBERO-130-fullshot-SFT

# SAM 3.0 checkpoint —— 从以下地址下载：
# https://modelscope.cn/models/facebook/sam3
pip install -U modelscope

modelscope download facebook/sam3 \
  --local-dir ./checkpoints/sam3

export SAM3_CHECKPOINT_PATH=$PWD/checkpoints/sam3/sam3.pt
export LIBERO_TYPE=pro

# 运行一个任务：libero_object_swap，task 2，seed 0，使用 Claude Code
# 和 Claude Opus 4.8。
rpent --robot libero --suite libero_object_swap --task 2 --seed 0 \
  --cuda-device 0 --planner claude_code --model claude-opus-4-8
```

其他规划器（`api`、`codex`）与模型提供商的配置见[规划器文档](https://rpent.readthedocs.io/zh-cn/latest/rst_source/usage/configure_planner.html)。
探索模式与本地 memory 评测详见 [LIBERO 文档](https://rpent.readthedocs.io/zh-cn/latest/rst_source/usage/libero.html)。

### 交互模式

加上 `--interactive`（`-i`）即可在终端里实时引导智能体。在 `you>` 提示符处，内置任务已预填——按 Enter 直接使用，或替换为你自己的任务；智能体运行时，随时输入消息即可在下一轮引导它（`/help` 查看命令，`/quit` 或 Ctrl-D 结束）。需要交互式终端（TTY）。

```bash
rpent --robot libero --suite libero_object_swap --task 2 --seed 0 \
  --planner claude_code --model claude-opus-4-8 --interactive
```

### 实时 Dashboard

加上 `--dashboard` 后，会启动本地 Dashboard，并在终端输出访问地址。打开该地址并确认配置；服务就绪后，通过 `/rpent-task <suite> <task> <seed>` 启动任务。页面会实时显示智能体的推理过程、相机画面和动作时间线，任务结束后可以继续提交下一任务。使用 `--dashboard-language zh-cn` 可切换到中文界面。

```bash
rpent --robot libero --dashboard --dashboard-language zh-cn \
  --planner claude_code --model claude-opus-4-8
```

完整的命令行参数列表见 [快速开始](https://rpent.readthedocs.io/zh-cn/latest/rst_source/quickstart.html#cli) 文档中的「关键 CLI 选项」表格。RoboCasa 与 RoboTwin 使用独立的入口和命令行参数，参见 [RoboCasa](https://rpent.readthedocs.io/zh-cn/latest/rst_source/usage/robocasa.html) 与 [RoboTwin](https://rpent.readthedocs.io/zh-cn/latest/rst_source/usage/robotwin.html) 文档。

更详细的文档请参见 [RPent 中文文档](https://rpent.readthedocs.io/zh-cn/latest/)。

## 参与贡献

欢迎社区贡献。开发环境、必需检查、测试规范和集成检查清单请参见
[CONTRIBUTING_zh.md](CONTRIBUTING_zh.md)。

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
