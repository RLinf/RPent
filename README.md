<div align="center">
  <img src="https://github.com/RLinf/misc/raw/main/pic/rpent_logo.png" alt="RPent-logo" width="520"/>
</div>

<div align="center">
<a href="https://arxiv.org/abs/2607.08448"><img src="https://img.shields.io/badge/arXiv-Paper-red?logo=arxiv"></a>
<a href="https://github.com/RLinf/RPent"><img src="https://img.shields.io/badge/GitHub-RPent-181717?logo=github"></a>
<a href="https://github.com/RLinf/RPent"><img src="https://img.shields.io/badge/Code-RPent-blue?logo=github"></a>
<a href="https://huggingface.co/RLinf"><img src="https://img.shields.io/badge/HuggingFace-yellow?logo=huggingface&logoColor=white" alt="Hugging Face"></a>
</div>

<div align="center">

[![English](https://img.shields.io/badge/lang-English-blue.svg)](README.md)
[![简体中文](https://img.shields.io/badge/语言-简体中文-red.svg)](README.zh-CN.md)

</div>

<h1 align="center">
  <sub>RPent: Agentic Infrastructure for the Physical World</sub>
</h1>

**RPent (Recursive Physical Agent)** is an open framework for building embodied agents that continuously evolve through recursive interaction with the physical world. Rather than prescribing a single foundation model, RPent provides a recursive agent framework that harnesses heterogeneous intelligence, including perception, reasoning, memory, execution, and self-evolution, into a unified physical agent. Through continuous interaction, reflection, and adaptation, RPent enables physical agents to acquire new capabilities and evolve beyond their initial design.

The name Pent is inspired by the ***Pentagram***, whose five points symbolize the integration of multimodal intelligence into a unified embodied agent. At its center, the infinity symbol (∞) represents the endless recursive cycle of perception, reasoning, execution, and self-evolution, through which intelligence continuously expands into the physical world.

RPent is built upon three core design principles: **service-oriented, standardized, and composable**. RPent enables capabilities to be deployed as reusable services, connected through unified interfaces, and flexibly composed into diverse physical agents. Together, these principles allow RPent to move beyond traditional robot control frameworks and establish an agentic infrastructure for the physical world, where intelligence is not only deployed, but continuously built, expanded, and evolved.

<div align="center">
  <img src="https://github.com/RLinf/misc/raw/main/pic/rpent_framework.png" alt="RPent framework"/>
</div>

## What's NEW!

- [2026/07] 🔥 Our first RPent publication, [Harness VLA: Steering Frozen VLAs into Reliable Manipulation Primitives via Memory-Guided Agents](https://arxiv.org/abs/2607.08448), is released.

## Supported Environments

<table width="100%">
  <thead align="center" valign="bottom">
    <tr>
      <th width="26%">Agentic Planner</th>
      <th width="28%">Action Primitives</th>
      <th width="26%" align="left">Simulator</th>
      <th width="20%">Real World</th>
    </tr>
  </thead>
  <tbody valign="top">
    <tr>
      <td>
        <ul style="margin-left: 0; padding-left: 16px;">
          <li><b>api</b> — pydantic-ai ✅</li>
          <ul>
            <li>Anthropic (Claude) ✅</li>
            <li>OpenAI (responses) ✅</li>
            <li>OpenAI-compatible (chat) ✅</li>
          </ul>
          <li><b>claude_code</b> — Claude Agent SDK ✅</li>
          <li><b>codex</b> — OpenAI Codex SDK ✅</li>
        </ul>
      </td>
      <td>
        <ul style="margin-left: 0; padding-left: 16px;">
          <li><b>VLA manipulation</b> ✅</li>
          <ul>
            <li>Pi0.5 (LIBERO, HTTP) ✅</li>
            <li>RLDX-1 (RoboCasa, socket-RPC) ✅</li>
          </ul>
        </ul>
      </td>
      <td style="text-align: left; padding-left: 8px;">
        <ul style="margin-left: 0; padding-left: 16px;">
          <li><b>LIBERO</b> (standard / pro / plus) ✅</li>
          <ul>
            <li>libero_object · _task / _swap / _lan</li>
            <li>libero_goal · _task / _swap / _lan</li>
            <li>libero_spatial · _task / _lan</li>
            <li>libero_10 · _task / _swap / _lan</li>
          </ul>
          <li><b>RoboCasa</b> (kitchen, long-horizon) ✅</li>
          <ul>
            <li>PickPlace* · Open/Close* · TurnOn/Off* …</li>
          </ul>
        </ul>
      </td>
      <td>
        <ul style="margin-left: 0; padding-left: 16px;">
          <li><b>Franka</b></li>
          <li><b>SO-101</b></li>
        </ul>
      </td>
    </tr>
  </tbody>
</table>

## Quick Start

RPent runs on top of a forked branch of [RLinf](https://github.com/RLinf/RLinf) for the simulators and VLA models. Clone them side by side.

**1. Clone RLinf and RPent side by side.**

```bash
mkdir workspace && cd workspace
# RPent depends on a forked branch of RLinf; it will be merged back to main after more iterations.
git clone https://github.com/jx-qiu/RLinf -b feature/physicalagent rlinf
git clone https://github.com/RLinf/RPent rpent
```

**2. In RLinf, create an openpi + LIBERO virtualenv.**

```bash
cd rlinf
bash requirements/install.sh embodied --env libero --model openpi --use-mirror --venv ../.venv-opi-libero
cd ..
source .venv-opi-libero/bin/activate
```

**3. Install RPent's extra dependencies on top of that venv.**

```bash
cd rpent
uv sync --active --inexact
bash scripts/install_libero_pro_plus.sh
```

**4. Configure keys and checkpoints, then run.**

```bash
# LLM API keys (the `api` cerebrum)
export ANTHROPIC_BASE_URL=https://xxx
export ANTHROPIC_API_KEY=sk-xxx
export OPENAI_BASE_URL=https://xxx
export OPENAI_API_KEY=sk-xxx

# VLA checkpoint — download from
# https://huggingface.co/datasets/RLinf/rlinf-pi05-libero-130-fullshot-sft
export PI05_CHECKPOINT_PATH=/path/to/rlinf-pi05-libero-130-fullshot-sft
export LIBERO_TYPE=pro
export CUDA_VISIBLE_DEVICES=0

# Run one task: libero_object_swap, task 2, seed 0, using the `api` cerebrum
# with an Anthropic model and an 8192-token cap.
#   • OpenAI-compatible chat endpoints:  --model openai-chat:glm-5.2
#   • OpenAI responses endpoints:        --model openai:gpt-5.5
#   • claude_code / codex cerebrums:     no provider prefix, e.g. --model claude-opus-4-8
python cli/main.py --suite libero_object_swap --task 2 --seed 0 \
  --cerebrum api --model anthropic:claude-opus-4-8 --max-tokens 8192
```

### Live Dashboard

Add `--dashboard` to open a browser monitor for the run. It boots a launcher screen where you pick the config, then streams reasoning, live views, and the action timeline. Use `--dashboard-language zh-cn` for the Chinese UI.

```bash
python cli/main.py --dashboard --dashboard-language zh-cn \
  --suite libero_goal_task --task 1 --seed 0 --cerebrum claude_code
```

### RoboCasa

RoboCasa uses a separate entrypoint and setup guide.

```bash
bash scripts/setup_robocasa.sh                                # one-time setup
bash scripts/run_robocasa.sh PickPlaceCounterToCabinet 0 0    # <task> <gpu> <seed>
```

See [SETUP_ROBOCASA.zh.md](docs/SETUP_ROBOCASA.zh.md) for the full RoboCasa365 + RLDX-1 walkthrough.

## Key CLI Options

| Flag | Default | Description |
| --- | --- | --- |
| `--suite` | — (required) | Task suite, e.g. `libero_object_task`, `libero_spatial_swap` |
| `--task` | — (required) | Task id within the suite |
| `--seed` | `0` | Random seed |
| `--cerebrum` | `api` | Reasoning brain: `api` \| `claude_code` \| `codex` |
| `--model` | — | Model id; for `api`, prefix the provider (`anthropic:…`, `openai:…`, `openai-chat:…`) |
| `--max-turns` | `100` | Max agent turns |
| `--max-tokens` | `8192` | Max tokens per LLM reply |
| `--max-episode-steps` | `10000` | Max env steps |
| `--libero-type` | `LIBERO_TYPE` or `pro` | LIBERO variant: `standard` \| `pro` \| `plus` |
| `--cuda-device` | inherited | GPU device(s) exposed to the env / vla servers |
| `--dashboard` | off | Start the local dashboard for this run |
| `--dashboard-language` | `en` | Dashboard UI language: `en` \| `zh-cn` |
| `--vla-endpoint` | — | Reuse an already-running vla_server instead of spawning one |
| `--no-driver` | off | Attach to an existing env_server / vla_server |

## Documentation

- [Adding a new environment](https://rpent.readthedocs.io/en/latest/rst_source/extending/new_env.html) — plug a new simulator / robot into the runner ([中文](https://rpent.readthedocs.io/zh-cn/latest/rst_source/extending/new_env.html)).
- [RoboCasa setup](docs/SETUP_ROBOCASA.zh.md) — RoboCasa365 + RLDX-1 install and run guide.
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
