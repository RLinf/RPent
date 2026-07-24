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
  <sub>RPent: Agentic Infrastructure for the Physical World</sub>
</h1>

**RPent (Recursive Physical Agent)** is an open framework for building embodied agents that continuously evolve through recursive interaction with the physical world. Rather than prescribing a single foundation model, RPent provides a recursive agent framework that harnesses heterogeneous intelligence, including perception, reasoning, memory, execution, and self-evolution, into a unified physical agent. Through continuous interaction, reflection, and adaptation, RPent enables physical agents to acquire new capabilities and evolve beyond their initial design.

RPent is built upon three core design principles: **service-oriented, standardized, and composable**. RPent enables capabilities to be deployed as reusable services, connected through unified interfaces, and flexibly composed into diverse physical agents. Together, these principles allow RPent to move beyond traditional robot control frameworks and establish an agentic infrastructure for the physical world, where intelligence is not only deployed, but continuously built, expanded, and evolved.

<div align="center">
  <img src="https://github.com/RLinf/misc/raw/main/pic/rpent_framework.png" alt="RPent framework"/>
</div>

## What's NEW!

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

## Quick Start

**1. Install RPent with a single `pip install`.**

```bash
git clone https://github.com/RLinf/RPent rpent && cd rpent
pip install -e ".[full]"
```

`.[full]` is the default end-to-end stack (openpi Pi0.5 VLA + LIBERO-PRO simulator + SAM 3.0 on the RLinf runtime).
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
export PI05_CHECKPOINT_PATH=/path/to/rlinf-pi05-libero-130-fullshot-sft
# SAM 3.0 checkpoint — download from
# https://huggingface.co/facebook/sam3
# https://modelscope.cn/models/facebook/sam3
export SAM3_CHECKPOINT_PATH=/path/to/sam3/sam3.pt
export LIBERO_TYPE=pro
export CUDA_VISIBLE_DEVICES=0

# Run one task: libero_object_swap, task 2, seed 0, using Claude Code
# with Claude Opus 4.8.
rpent --env libero --suite libero_object_swap --task 2 --seed 0 \
  --planner claude_code --model claude-opus-4-8
```

See the [planner docs](https://rpent.readthedocs.io/en/latest/rst_source/usage/configure_planner.html) to configure other planners (`api`, `codex`) and model providers.

### Interactive CLI mode

Add `--interactive` (`-i`) to steer the agent live from your terminal. At the `you>` prompt, the built-in task is pre-filled — press Enter to use it or replace it with your own — then type any message while it runs to steer the agent at the next turn (`/help` lists commands; `/quit` or Ctrl-D ends). Requires an interactive terminal (TTY).

```bash
rpent --env libero --suite libero_object_swap --task 2 --seed 0 \
  --planner claude_code --model claude-opus-4-8 --interactive
```

### Live Dashboard

Add `--dashboard` to start a local dashboard server. The command prints the URL in the terminal; open it to confirm the configuration on the launcher screen. Once the run starts, the page streams agent reasoning, camera and Pi0 views, the action timeline, and clip replays. Use `--dashboard-language zh-cn` for the Chinese UI.

```bash
rpent --env libero --dashboard --dashboard-language zh-cn \
  --suite libero_goal_task --task 1 --seed 0 \
  --planner claude_code --model claude-opus-4-8
```

For more detailed documentation, see the [RPent documentation](https://rpent.readthedocs.io/en/latest/).

## Key CLI Options

<table width="100%" style="width: 100%; table-layout: auto; border-collapse: collapse;">
  <thead align="center" valign="bottom">
    <tr>
      <th style="min-width: 160px; text-align: left;">Flag</th>
      <th style="min-width: 120px;">Default</th>
      <th style="min-width: 360px;">Description</th>
    </tr>
  </thead>
  <tbody valign="top">
    <tr><td><code>--env</code></td><td>— (required)</td><td>Environment backend. Currently <code>libero</code>.</td></tr>
    <tr><td><code>--suite</code></td><td>— (required)</td><td>Task suite, e.g. <code>libero_object_task</code>, <code>libero_spatial_swap</code></td></tr>
    <tr><td><code>--task</code></td><td>— (required)</td><td>Task id within the suite</td></tr>
    <tr><td><code>--seed</code></td><td><code>0</code></td><td>Random seed</td></tr>
    <tr><td><code>--planner</code></td><td><code>api</code></td><td><code>api</code> | <code>claude_code</code> | <code>codex</code></td></tr>
    <tr><td><code>--model</code></td><td>—</td><td>Model id; for <code>api</code>, prefix the provider (<code>anthropic:…</code>, <code>openai:…</code>, <code>openai-chat:…</code>)</td></tr>
    <tr><td><code>--max-turns</code></td><td><code>100</code></td><td>Max agent turns</td></tr>
    <tr><td><code>--max-tokens</code></td><td><code>8192</code></td><td>Max tokens per LLM reply</td></tr>
    <tr><td><code>--no-images</code></td><td>off</td><td>Text-only mode: never send image bytes (for models that reject image input)</td></tr>
    <tr><td><code>--max-episode-steps</code></td><td><code>10000</code></td><td>Max env steps</td></tr>
    <tr><td><code>--libero-type</code></td><td><code>LIBERO_TYPE</code> or <code>pro</code></td><td>LIBERO variant: <code>standard</code> | <code>pro</code> | <code>plus</code></td></tr>
    <tr><td><code>--cuda-device</code></td><td>inherited</td><td>GPU device(s) exposed to the env / VLA / SAM3 servers</td></tr>
    <tr><td><code>--dashboard</code></td><td>off</td><td>Start the local dashboard for this run</td></tr>
    <tr><td><code>--dashboard-language</code></td><td><code>en</code></td><td>Dashboard UI language: <code>en</code> | <code>zh-cn</code></td></tr>
    <tr><td><code>--env-endpoint</code></td><td>— (spawn)</td><td><code>[protocol://]host:port</code> of an existing env_server (<code>protocol=http|socket</code>, default <code>http</code>). If unset, one is spawned locally.</td></tr>
    <tr><td><code>--vla-endpoint</code></td><td>— (spawn)</td><td><code>[protocol://]host:port</code> of an existing vla_server (same rules). If unset, one is spawned locally.</td></tr>
    <tr><td><code>--sam3-endpoint</code></td><td>— (spawn)</td><td><code>[protocol://]host:port</code> of an existing RPent SAM3 service (same rules). If unset, one is spawned locally.</td></tr>
  </tbody>
</table>

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
