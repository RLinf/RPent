<div align="center">
  <img src="https://github.com/RLinf/misc/raw/main/pic/rpent_logo.png" alt="RPent-logo" width="520"/>
</div>

<div align="center">
<a href="https://arxiv.org/abs/2607.08448"><img src="https://img.shields.io/badge/arXiv-Paper-red?logo=arxiv" alt="arXiv Paper"></a>
<a href="https://huggingface.co/RLinf"><img src="https://img.shields.io/badge/HuggingFace-yellow?logo=huggingface&logoColor=white" alt="Hugging Face"></a>
<a href="https://rpent.readthedocs.io/en/latest/"><img src="https://img.shields.io/badge/Documentation-Purple?color=8A2BE2&logo=readthedocs" alt="Documentation"></a>
<a href="https://rpent.readthedocs.io/zh-cn/latest/"><img src="https://img.shields.io/badge/中文文档-red?logo=readthedocs" alt="Chinese Documentation"></a>
<a href="https://github.com/RLinf/misc/blob/main/pic/wechat.jpg?raw=true"><img src="https://img.shields.io/badge/微信-green?logo=wechat" alt="WeChat"></a>
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

## Leaderboard

Compare RPent with reference methods on LIBERO, LIBERO-PRO, RoboCasa365 Target50, and RoboTwin C2R. Rankings apply to the methods and evaluation coverage shown; see the [Leaderboard](https://rpent.readthedocs.io/en/latest/rst_source/leaderboard/index.html) for suite results and model configurations.

Codex / GPT-6 Astra / low / reasoning: **92.63% Overall (741/800)** across all eight LIBERO-PRO suites. See the [suite results and memory-batch explanation](https://rpent.readthedocs.io/en/latest/rst_source/leaderboard/index.html#libero-pro-astra-memory), including the separately frozen Long and Spatial/Object/Goal memory batches.

[![RPent Leaderboard](https://rpent.readthedocs.io/en/latest/_static/leaderboard/leaderboard-en-light.svg)](https://rpent.readthedocs.io/en/latest/rst_source/leaderboard/index.html)


## Who Should Consider Using RPent?

RPent is built for four kinds of users:

- **Embodied intelligence researchers** targeting high success rates on embodied tasks and benchmarks — especially long-horizon manipulation. RPent's memory-guided, agentic composition consistently lifts task success beyond what a frozen VLA delivers alone.
- **Online-learning and reinforcement-learning researchers** studying self-evolving embodied agents. RPent's recursive interaction, reflection, and memory-distillation loops provide a ready substrate for continual and reinforcement learning in the physical world.
- **Robotics application developers** deploying embodied solutions on real robot hardware. RPent's service-oriented, standardized architecture and customizable agentic control logic improve real-world success rates and shorten the path from prototype to production.
- **End users of deployed embodied agents** — the customers of the developers above. Install RPent together with the relevant real-robot extensions and run the predefined tasks out of the box, with no ML expertise required.

## What's NEW!

- [2026/11] 🔥 Replaced per-robot `pyproject.toml` extras with per-component environment installation and diagnostics via `./rpent-install.sh` (`rpent-install`). See the [installation & migration guide](https://rpent.readthedocs.io/en/latest/rst_source/get_started/installation.html).
- [2026/11] 🔥 Fixed an issue that caused the API planner to underperform. With DeepSeek V4.1 Flash, the API planner achieved a success rate of **65.6% (164/250)** on RoboTwin C2R. See the [Leaderboard](https://rpent.readthedocs.io/en/latest/rst_source/leaderboard/index.html).
- [2026/11] 🔥 RPent supports Embodied First Exam (EFE) tasks through Harbor, with Skills or the task's native actions. See the [EFE guide](https://rpent.readthedocs.io/en/latest/rst_source/simulators/efe.html).
- [2026/11] 🔥 Added an [embodied RGB-D toolkit](https://rpent.readthedocs.io/en/latest/rst_source/toolkits/rgbd.html) for reading 3D coordinates, segmenting objects, and measuring object parts from recorded camera views.
- [2026/11] 🔥 RPent supports external MCP tools through `--mcp-config`, exposing third-party Streamable HTTP servers across all online planners. See the [external MCP guide](https://rpent.readthedocs.io/en/latest/rst_source/toolkits/mcp.html).
- [2026/11] 🔥 Added the data flywheel. Supervised fine-tuning of the VLA on successful trajectories collected with the flywheel raised RPent's task success rate on RoboTwin from 48% to 72%, a gain of **24 percentage points**. See the [user guide](https://rpent.readthedocs.io/en/latest/rst_source/guides/flywheel.html).
- [2026/10] 🔥 Added an optional APXInf VLA endpoint. See the [deployment guide](https://rpent.readthedocs.io/en/latest/rst_source/guides/advanced_deployment.html#apxinf-vla-service-optional).
- [2026/09] 🔥 Added the interactive RPent Leaderboard for LIBERO, LIBERO-PRO, RoboCasa365, and RoboTwin, with model comparisons and suite-level results. Explore the [Leaderboard](https://rpent.readthedocs.io/en/latest/rst_source/leaderboard/index.html).
- [2026/09] 🔥 RPent supports Franka single-arm and dual-arm real-robot extensions. Doc: [Franka](https://rpent.readthedocs.io/en/latest/rst_source/real_world_robots/franka.html) · [Dual Franka](https://rpent.readthedocs.io/en/latest/rst_source/real_world_robots/dual_franka.html).
- [2026/08] 🔥 RPent supports RoboCasa with RLDX-1 as manipulation model. See the [RoboCasa setup and Target50 guide](https://rpent.readthedocs.io/en/latest/rst_source/simulators/robocasa.html).
- [2026/08] 🔥 RPent supports the non-reasoning mode, which reduces average execution time by ~40%.
- [2026/08] 🔥 RPent supports exploration mode for LIBERO. Doc: [LIBERO exploration mode](https://rpent.readthedocs.io/en/latest/rst_source/simulators/libero.html#libero-exploration).
- [2026/08] 🔥 RPent supports RoboTwin with LingBot-VLA for dual-arm manipulation tasks. Doc: [RoboTwin](https://rpent.readthedocs.io/en/latest/rst_source/simulators/robotwin.html).
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
          <li><a href="https://rpent.readthedocs.io/en/latest/rst_source/guides/configure_planner.html#planner-claude-code">Claude Code</a> ✅</li>
          <li><a href="https://rpent.readthedocs.io/en/latest/rst_source/guides/configure_planner.html#planner-codex">Codex</a> ✅</li>
          <li><a href="https://rpent.readthedocs.io/en/latest/rst_source/guides/configure_planner.html#planner-gemini-cli">Gemini CLI</a> ✅</li>
          <li><a href="https://rpent.readthedocs.io/en/latest/rst_source/guides/configure_planner.html#planner-opencode">OpenCode</a> ✅</li>
          <li><a href="https://rpent.readthedocs.io/en/latest/rst_source/development/add_planner.html">Custom Planner</a> ✅</li>
        </ul>
      </td>
      <td>
        <ul style="margin-left: 0; padding-left: 16px;">
          <li><b>VLA</b></li>
          <ul>
            <li><a href="https://rpent.readthedocs.io/en/latest/rst_source/simulators/libero.html">Pi0.5</a> ✅</li>
            <li><a href="https://rpent.readthedocs.io/en/latest/rst_source/simulators/robocasa.html">RLDX-1</a> ✅</li>
            <li><a href="https://rpent.readthedocs.io/en/latest/rst_source/simulators/robotwin.html">LingBot-VLA</a> ✅</li>
          </ul>
          <li><b>WAM</b></li>
          <ul>
            <li>DreamZero</li>
            <li><a href="https://rpent.readthedocs.io/en/latest/rst_source/simulators/robodojo.html#vpp2-setup">VPP2</a> (RoboDojo) ✅</li>
            <li><a href="https://rpent.readthedocs.io/en/latest/rst_source/simulators/robodojo.html#awomo05-setup">Awomo05</a> (RoboDojo) ✅</li>
            <li><a href="https://rpent.readthedocs.io/en/latest/rst_source/development/add_action_model.html">Cosmos Policy</a> (LIBERO / LIBERO-Pro)</li>
            <li><a href="https://rpent.readthedocs.io/en/latest/rst_source/development/add_action_model.html">Fast-WAM</a> (LIBERO / LIBERO-Pro / RoboTwin)</li>
            <li><a href="https://rpent.readthedocs.io/en/latest/rst_source/development/add_action_model.html">UniWAM</a> (RoboTwin)</li>
          </ul>
        </ul>
      </td>
      <td style="text-align: left; padding-left: 8px;">
        <ul style="margin-left: 0; padding-left: 16px;">
          <li><a href="https://rpent.readthedocs.io/en/latest/rst_source/simulators/libero.html">LIBERO-PRO</a> ✅</li>
          <li><a href="https://rpent.readthedocs.io/en/latest/rst_source/simulators/robocasa.html">RoboCasa</a> ✅</li>
          <li><a href="https://rpent.readthedocs.io/en/latest/rst_source/simulators/robotwin.html">RoboTwin</a> ✅</li>
          <li><a href="https://rpent.readthedocs.io/en/latest/rst_source/simulators/robodojo.html">RoboDojo</a> ✅</li>
          <li><a href="https://rpent.readthedocs.io/en/latest/rst_source/simulators/robolab.html">RoboLab</a> ✅</li>
          <li><a href="https://rpent.readthedocs.io/en/latest/rst_source/simulators/efe.html">EFE</a> ✅</li>
        </ul>
      </td>
      <td>
        <ul style="margin-left: 0; padding-left: 16px;">
          <li><a href="https://rpent.readthedocs.io/en/latest/rst_source/real_world_robots/franka.html">Franka</a> ✅</li>
          <li><a href="https://rpent.readthedocs.io/en/latest/rst_source/real_world_robots/dual_franka.html">Dual Franka</a> ✅</li>
          <li>SO-101 (planned)</li>
          <li><a href="https://rpent.readthedocs.io/en/latest/rst_source/real_world_robots/yam.html">YAM</a> ✅</li>
        </ul>
      </td>
    </tr>
  </tbody>
</table>

## Quick Start

**1. Choose an environment and install RPent.**

```bash
git clone https://github.com/RLinf/RPent rpent && cd rpent
# Recommended default (LIBERO-PRO):
./rpent-install.sh install robot-libero --venv .venv-libero
source .venv-libero/bin/activate

# Other robots install the same way, each into its own environment (uncomment one):
# ./rpent-install.sh install robot-robocasa --venv .venv-robocasa
# ./rpent-install.sh install robot-robotwin --venv .venv-robotwin
# ./rpent-install.sh install robot-robolab --venv .venv-robolab-policy
# ./rpent-install.sh install robot-franka --venv .venv-franka-real
```

`robot-libero` is the recommended default. See the [installation docs](https://rpent.readthedocs.io/en/latest/rst_source/get_started/installation.html) for all components and diagnostics, or jump directly to the [RoboLab guide](https://rpent.readthedocs.io/en/latest/rst_source/simulators/robolab.html) and [RoboCasa guide](https://rpent.readthedocs.io/en/latest/rst_source/simulators/robocasa.html) (including [RPent-memory](https://huggingface.co/datasets/RLinf/RPent-memory/tree/main/robocasa) and Target50 setup).

The example below continues with LIBERO-PRO.

**2. Download the LIBERO-PRO simulator assets.**

```bash
liberopro-download-assets --skip-existing
```

> 💡 Slow connection to Hugging Face? Download through the mirror: `HF_ENDPOINT=https://hf-mirror.com liberopro-download-assets --skip-existing`.

See the [installation docs](https://rpent.readthedocs.io/en/latest/rst_source/get_started/installation.html) for other simulators.

**3. Configure keys and checkpoints, then run.**

```bash
# Anthropic key; no need to export the base url if you use the official endpoint.
export ANTHROPIC_API_KEY=sk-xxx

# VLA checkpoint — download from
# https://huggingface.co/RLinf/RLinf-Pi05-LIBERO-130-fullshot-SFT
hf download RLinf/RLinf-Pi05-LIBERO-130-fullshot-SFT \
  --exclude optimizer.pt \
  --local-dir ./checkpoints/RLinf-Pi05-LIBERO-130-fullshot-SFT

export PI05_CHECKPOINT_PATH=$PWD/checkpoints/RLinf-Pi05-LIBERO-130-fullshot-SFT

# SAM 3.0 checkpoint — download from
# https://modelscope.cn/models/facebook/sam3
uv pip install -U modelscope

modelscope download --model facebook/sam3 sam3.pt \
  --local_dir ./checkpoints/sam3

export SAM3_CHECKPOINT_PATH=$PWD/checkpoints/sam3/sam3.pt

# Run one task: libero_object_swap, task 2, seed 0, using Claude Code
# with Claude Opus 4.8.
rpent --robot libero --libero-type pro --suite libero_object_swap --task 2 --seed 0 \
  --env-cuda-device 0 --am-cuda-device 0 --sam3-cuda-device 0 \
  --planner claude_code --model claude-opus-4-8
```

See the [planner docs](https://rpent.readthedocs.io/en/latest/rst_source/guides/configure_planner.html) to configure other planners (`api`, `codex`, `gemini_cli`, `opencode`) and model providers.
For the exploration workflow and local-memory evaluation, see [LIBERO exploration mode](https://rpent.readthedocs.io/en/latest/rst_source/simulators/libero.html#libero-exploration).

### Interactive CLI mode

With `claude_code`, `codex`, `gemini_cli`, or `opencode`, add `--tui` (`-t`) for terminal interaction. The built-in task is pre-filled at the `you>` prompt; edit it or press Enter to submit. Follow-up messages queue until the current turn completes. Use `/interrupt [message]` to interrupt; an attached message goes ahead of the queue. `/help` lists commands; `/quit` or Ctrl-D ends the session. Requires a TTY. The `api` planner uses a native CLI that runs the preset task first and accepts follow-ups between runs; use `/exit` to close it.

```bash
rpent --robot libero --libero-type pro --suite libero_object_swap --task 2 --seed 0 \
  --planner claude_code --model claude-opus-4-8 --tui
```

### Live Dashboard

Add `--dashboard` to start a local Dashboard and print its URL in the terminal. Session settings come from the CLI, and the page opens directly in the live monitor. Once the services are ready, start a task with `/rpent-task <suite> <task> <seed>`. The page streams agent reasoning, camera views, and the action timeline, and you can submit another task after the current one finishes. Use `--dashboard-language zh-cn` for the Chinese UI.

```bash
rpent --robot libero --dashboard --dashboard-language zh-cn \
  --planner claude_code --model claude-opus-4-8
```

For a complete list of CLI options, see the [Command-Line Reference](https://rpent.readthedocs.io/en/latest/rst_source/guides/cli.html). RoboCasa and RoboTwin provide their own task parameters — see the [RoboCasa](https://rpent.readthedocs.io/en/latest/rst_source/simulators/robocasa.html) and [RoboTwin](https://rpent.readthedocs.io/en/latest/rst_source/simulators/robotwin.html) docs.

For more detailed documentation, see the [RPent documentation](https://rpent.readthedocs.io/en/latest/).

## Adoption

We are happy to see RPent has been adopted by leading companies and research institutions, with users and collaborators including NVIDIA, Huawei, Tencent, ROBAM, Z-Trans AI (Zetta project), Tsinghua University, Peking University, and the National University of Singapore.

✨ If you find RPent useful for embodied AI research or robotics applications, feel free to reach out or submit a PR to have your organization listed here.

<div align="center">
  <img src="https://cdn.jsdelivr.net/gh/RLinf/misc@beb75d9289a0dee012004c26436f8fb1c688f310/rpent/adoption/adoption.png" alt="RPent community: 10 companies, including Z-Trans AI (Zetta project) and ROBAM, followed by 22 universities and 8 research institutions"/>
</div>

## Contributing

Contributions are welcome. See [CONTRIBUTING.md](CONTRIBUTING.md) for the
development setup, required checks, testing policy, and integration checklists.

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
