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

`.[full]` is the default end-to-end stack (openpi Pi0.5 VLA + LIBERO-PRO simulator on the RLinf runtime). 
Pick a narrower extra if you don't need the whole stack:

<table width="100%" style="width: 100%; table-layout: auto; border-collapse: collapse;">
  <thead align="center" valign="bottom">
    <tr>
      <th style="min-width: 120px; text-align: left;">Extra</th>
      <th style="min-width: 240px;">Installs</th>
    </tr>
  </thead>
  <tbody valign="top">
    <tr>
      <td><code>.[full]</code></td>
      <td><code>rlinf</code> + <code>openpi</code> + <code>libero-pro</code> — the default run stack</td>
    </tr>
    <tr>
      <td><code>.[libero-pro]</code></td>
      <td>Base LIBERO + LIBERO-PRO simulator</td>
    </tr>
    <tr>
      <td><code>.[libero-plus]</code></td>
      <td>Base LIBERO + LIBERO-plus simulator</td>
    </tr>
    <tr>
      <td><code>.[libero]</code></td>
      <td>Base LIBERO only</td>
    </tr>
    <tr>
      <td><code>.[openpi]</code></td>
      <td>openpi VLA only</td>
    </tr>
    <tr>
      <td><code>.[rlinf]</code></td>
      <td>RLinf runtime only</td>
    </tr>
  </tbody>
</table>

**2. Configure keys and checkpoints, then run.**

```bash
# LLM API keys (the `api` planner)
export ANTHROPIC_BASE_URL=https://xxx
export ANTHROPIC_API_KEY=sk-xxx
export OPENAI_BASE_URL=https://xxx
export OPENAI_API_KEY=sk-xxx

# VLA checkpoint — download from
# https://huggingface.co/RLinf/RLinf-Pi05-LIBERO-130-fullshot-SFT
export PI05_CHECKPOINT_PATH=/path/to/rlinf-pi05-libero-130-fullshot-sft
export LIBERO_TYPE=pro
export CUDA_VISIBLE_DEVICES=0

# Run one task: libero_object_swap, task 2, seed 0, using the `api` planner
# with an Anthropic model and an 8192-token cap.
#   • OpenAI-compatible chat endpoints:  --model openai-chat:glm-5.2
#   • OpenAI responses endpoints:        --model openai:gpt-5.5
#   • claude_code / codex planners:     no provider prefix, e.g. --model claude-opus-4-8
rpent --env libero --suite libero_object_swap --task 2 --seed 0 \
  --planner api --model anthropic:claude-opus-4-8 --max-tokens 8192
```

### Live Dashboard

Add `--dashboard` to open a browser monitor for the run. It boots a launcher screen where you pick the config, then streams reasoning, live views, and the action timeline. Use `--dashboard-language zh-cn` for the Chinese UI.

```bash
rpent --env libero --dashboard --dashboard-language zh-cn \
  --suite libero_goal_task --task 1 --seed 0 --planner claude_code
```

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
    <tr><td><code>--planner</code></td><td><code>api</code></td><td>Reasoning brain: <code>api</code> | <code>claude_code</code> | <code>codex</code></td></tr>
    <tr><td><code>--model</code></td><td>—</td><td>Model id; for <code>api</code>, prefix the provider (<code>anthropic:…</code>, <code>openai:…</code>, <code>openai-chat:…</code>)</td></tr>
    <tr><td><code>--max-turns</code></td><td><code>100</code></td><td>Max agent turns</td></tr>
    <tr><td><code>--max-tokens</code></td><td><code>8192</code></td><td>Max tokens per LLM reply</td></tr>
    <tr><td><code>--max-episode-steps</code></td><td><code>10000</code></td><td>Max env steps</td></tr>
    <tr><td><code>--libero-type</code></td><td><code>LIBERO_TYPE</code> or <code>pro</code></td><td>LIBERO variant: <code>standard</code> | <code>pro</code> | <code>plus</code></td></tr>
    <tr><td><code>--cuda-device</code></td><td>inherited</td><td>GPU device(s) exposed to the env / vla servers</td></tr>
    <tr><td><code>--dashboard</code></td><td>off</td><td>Start the local dashboard for this run</td></tr>
    <tr><td><code>--dashboard-language</code></td><td><code>en</code></td><td>Dashboard UI language: <code>en</code> | <code>zh-cn</code></td></tr>
    <tr><td><code>--env-endpoint</code></td><td>— (spawn)</td><td><code>[protocol://]host:port</code> of an existing env_server (<code>protocol=http|socket</code>, default <code>http</code>). If unset, one is spawned locally.</td></tr>
    <tr><td><code>--vla-endpoint</code></td><td>— (spawn)</td><td><code>[protocol://]host:port</code> of an existing vla_server (same rules). If unset, one is spawned locally.</td></tr>
  </tbody>
</table>

## Documentation

- [Adding a new environment](https://rpent.readthedocs.io/en/latest/rst_source/extending/new_env.html) — plug a new simulator / robot into the runner ([中文](https://rpent.readthedocs.io/zh-cn/latest/rst_source/extending/new_env.html)).
- [`docs/`](docs/README.md) — local Sphinx build and preview instructions.

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
