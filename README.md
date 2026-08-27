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
  <sub>RPent: Agentic Infrastructure for the Physical World</sub>
</h1>

**RPent (Recursive Physical Agent)** is an open framework for building embodied agents that continuously evolve through recursive interaction with the physical world. Rather than prescribing a single foundation model, RPent provides a recursive agent framework that harnesses heterogeneous intelligence, including perception, reasoning, memory, execution, and self-evolution, into a unified physical agent. Through continuous interaction, reflection, and adaptation, RPent enables physical agents to acquire new capabilities and evolve beyond their initial design. We build RPent upon a foundation of **service-oriented**, **standardized**, and **composable** design principles, ensuring the framework remains highly extensible.

<div align="center">
  <img src="https://github.com/RLinf/misc/raw/main/pic/rpent_framework.png" alt="RPent framework"/>
</div>

## Who Should Consider Using RPent?

RPent is built for four kinds of users:

- **Embodied intelligence researchers** targeting high success rates on embodied tasks and benchmarks — especially long-horizon manipulation. RPent's memory-guided, agentic composition consistently lifts task success beyond what a frozen VLA delivers alone.
- **Online-learning and reinforcement-learning researchers** studying self-evolving embodied agents. RPent's recursive interaction, reflection, and memory-distillation loops provide a ready substrate for continual and reinforcement learning in the physical world.
- **Robotics application developers** deploying embodied solutions on real robot hardware. RPent's service-oriented, standardized architecture and customizable agentic control logic improve real-world success rates and shorten the path from prototype to production.
- **End users of deployed embodied agents** — the customers of the developers above. Install RPent together with the relevant real-robot extensions and run the predefined tasks out of the box, with no ML expertise required.

## What's NEW!

- [2026/08] 🔥 RPent supports the non-reasoning mode, which reduces average execution time by ~40%.
- [2026/08] 🔥 RPent supports exploration mode for LIBERO. Doc: [LIBERO exploration mode](https://rpent.readthedocs.io/en/latest/rst_source/usage/libero.html#exploration-and-local-memory-evaluation).
- [2026/08] 🔥 RPent supports RoboTwin with LingBot-VLA for dual-arm manipulation tasks. Doc: [RoboTwin](https://rpent.readthedocs.io/en/latest/rst_source/usage/robotwin.html).
- [2026/08] 🔥 RPent supports RoboCasa with RLDX-1 as manipulation model. Doc: [RoboCasa](https://rpent.readthedocs.io/en/latest/rst_source/usage/robocasa.html).
- [2026/07] 🔥 Our first RPent publication, [Harness VLA: Steering Frozen VLAs into Reliable Manipulation Primitives via Memory-Guided Agents](https://arxiv.org/abs/2607.08448), is released.

## Feature Matrix

<table width="100%" style="width: 100%; table-layout: auto; border-collapse: collapse;">
  <thead align="center" valign="bottom">
    <tr>
      <th style="min-width: 300px;">Agentic Planner</th>
      <th style="min-width: 340px;">Action Primitive</th>
      <th style="min-width: 300px; text-align: left;">Simulator</th>
      <th style="min-width: 260px;">Real World</th>
    </tr>
  </thead>
  <tbody valign="top">
    <tr>
      <td>
        <ul style="margin-left: 0; padding-left: 16px;">
          <li><a href="https://rpent.readthedocs.io/en/latest/rst_source/usage/configure_planner.html#the-claude-code-planner">Claude Code</a> ✅</li>
          <li><a href="https://rpent.readthedocs.io/en/latest/rst_source/usage/configure_planner.html#the-codex-planner">Codex</a> ✅</li>
          <li><a href="https://rpent.readthedocs.io/en/latest/rst_source/usage/configure_planner.html#add-a-custom-planner">Custom Planner</a> ✅</li>
        </ul>
      </td>
      <td>
        <ul style="margin-left: 0; padding-left: 16px;">
          <li><b>VLA</b></li>
          <ul>
            <li><a href="https://rpent.readthedocs.io/en/latest/rst_source/usage/libero.html">Pi0.5</a> ✅</li>
            <li><a href="https://rpent.readthedocs.io/en/latest/rst_source/usage/robocasa.html">RLDX-1</a> ✅</li>
            <li><a href="https://rpent.readthedocs.io/en/latest/rst_source/usage/robotwin.html">LingBot-VLA</a> ✅</li>
          </ul>
          <li><b>WAM</b></li>
          <ul>
            <li>DreamZero</li>
          </ul>
        </ul>
      </td>
      <td style="text-align: left; padding-left: 8px;">
        <ul style="margin-left: 0; padding-left: 16px;">
          <li><a href="https://rpent.readthedocs.io/en/latest/rst_source/usage/libero.html">LIBERO-PRO</a> ✅</li>
          <li><a href="https://rpent.readthedocs.io/en/latest/rst_source/usage/robocasa.html">RoboCasa</a> ✅</li>
          <li><a href="https://rpent.readthedocs.io/en/latest/rst_source/usage/robotwin.html">RoboTwin</a> ✅</li>
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

## Quick Start

**1. Install RPent with a single `pip install`.**

```bash
git clone https://github.com/RLinf/RPent rpent && cd rpent
pip install -e ".[full]"
```

`.[full]` is the default end-to-end stack (openpi Pi0.5 VLA + LIBERO-PRO and RoboCasa365 simulators + SAM 3.0 on the RLinf runtime).
If you don't need the whole stack, see the [installation docs](https://rpent.readthedocs.io/en/latest/rst_source/installation.html) for narrower extras.

**2. Download the LIBERO-PRO simulator assets.**

```bash
liberopro-download-assets --skip-existing
```

> 💡 Slow connection to Hugging Face? Download through the mirror: `HF_ENDPOINT=https://hf-mirror.com liberopro-download-assets --skip-existing`.

See the [installation docs](https://rpent.readthedocs.io/en/latest/rst_source/installation.html) for other simulators.

**3. Configure keys and checkpoints, then run.**

```bash
# Anthropic key; no need to export the base url if you use the official endpoint.
export ANTHROPIC_BASE_URL=https://xxx
export ANTHROPIC_API_KEY=sk-xxx

# VLA checkpoint — download from
# https://huggingface.co/RLinf/RLinf-Pi05-LIBERO-130-fullshot-SFT
hf download RLinf/RLinf-Pi05-LIBERO-130-fullshot-SFT \
  --exclude optimizer.pt \
  --local-dir ./checkpoints/RLinf-Pi05-LIBERO-130-fullshot-SFT

export PI05_CHECKPOINT_PATH=$PWD/checkpoints/RLinf-Pi05-LIBERO-130-fullshot-SFT

# SAM 3.0 checkpoint — download from
# https://modelscope.cn/models/facebook/sam3
pip install -U modelscope

modelscope download facebook/sam3 \
  --local-dir ./checkpoints/sam3

export SAM3_CHECKPOINT_PATH=$PWD/checkpoints/sam3/sam3.pt
export LIBERO_TYPE=pro

# Run one task: libero_object_swap, task 2, seed 0, using Claude Code
# with Claude Opus 4.8.
rpent --robot libero --suite libero_object_swap --task 2 --seed 0 \
  --cuda-device 0 --planner claude_code --model claude-opus-4-8
```

See the [planner docs](https://rpent.readthedocs.io/en/latest/rst_source/usage/configure_planner.html) to configure other planners (`api`, `codex`) and model providers.
For the exploration workflow and local-memory evaluation, see [LIBERO exploration mode](https://rpent.readthedocs.io/en/latest/rst_source/usage/libero.html#exploration-and-local-memory-evaluation).

### Interactive CLI mode

Add `--interactive` (`-i`) to steer the agent live from your terminal. At the `you>` prompt, the built-in task is pre-filled — press Enter to use it or replace it with your own — then type any message while it runs to steer the agent at the next turn (`/help` lists commands; `/quit` or Ctrl-D ends). Requires an interactive terminal (TTY).

```bash
rpent --robot libero --suite libero_object_swap --task 2 --seed 0 \
  --planner claude_code --model claude-opus-4-8 --interactive
```

### Live Dashboard

Add `--dashboard` to start a local Dashboard and print its URL in the terminal. Open the URL and confirm the configuration; once the services are ready, start a task with `/rpent-task <suite> <task> <seed>`. The page streams agent reasoning, camera views, and the action timeline, and you can submit another task after the current one finishes. Use `--dashboard-language zh-cn` for the Chinese UI.

```bash
rpent --robot libero --dashboard --dashboard-language zh-cn \
  --planner claude_code --model claude-opus-4-8
```

For a complete list of CLI options, see the [Key CLI options](https://rpent.readthedocs.io/en/latest/rst_source/quickstart.html#key-cli-options) table in the Quick Start docs. RoboCasa and RoboTwin use their own entrypoints and CLI — see the [RoboCasa](https://rpent.readthedocs.io/en/latest/rst_source/usage/robocasa.html) and [RoboTwin](https://rpent.readthedocs.io/en/latest/rst_source/usage/robotwin.html) docs.

For more detailed documentation, see the [RPent documentation](https://rpent.readthedocs.io/en/latest/).

## Citation and Acknowledgement

If you find **RPent** or **Harness VLA** helpful, please cite the paper:

```bibtex
@article{zhang2026harnessvla,
  title={Harness VLA: Steering Frozen VLAs into Reliable Manipulation Primitives via Memory-Guided Agents},
  author={Zhang, Yixian and Zhang, Huanming and Gao, Feng and Li, Xiao and Liu, Zhihao and Zhu, Chunyang and Qiu, Jiaxing and Yan, Yuchen and Liu, Jiyuan and Tang, Wenhao and Fang, Zhengru and Nie, Yi and Wei, Changxu and Wang, Yu and Ding, Wenbo and Yu, Chao},
  journal={arXiv preprint arXiv:2607.08448},
  year={2026},
  url={https://arxiv.org/abs/2607.08448}
}
```

RPent builds on the simulators, VLA models, and training infrastructure of [RLinf](https://github.com/RLinf/RLinf), and on the agent SDKs of the broader open-source community — [pydantic-ai](https://ai.pydantic.dev/), the [Claude Agent SDK](https://docs.claude.com/en/api/agent-sdk/overview), and the OpenAI Codex SDK. Thanks to the teams behind LIBERO, RoboCasa, robosuite, MuJoCo, and openpi.
