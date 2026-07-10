<div align="center">
  <h1>RPent</h1>
  <p><i>A physical-agent framework where LLMs reason and VLAs act, in a closed loop.</i></p>
</div>

<div align="center">

[![English](https://img.shields.io/badge/lang-English-blue.svg)](README.md)
[![简体中文](https://img.shields.io/badge/语言-简体中文-red.svg)](README.zh-CN.md)
[![GitHub](https://img.shields.io/badge/GitHub-RPent-181717?logo=github)](https://github.com/RLinf/RPent)

</div>

RPent is an **physical-agent framework** that puts a large language model *in the loop* as the decision-making brain. The LLM does high-level reasoning and calls tools; a Vision-Language-Action (VLA) policy such as **Pi0.5** or **RLDX-1** executes the low-level motor actions; a simulator (**LIBERO** or **RoboCasa**) closes the loop by returning observations and rendered frames. Reasoning, action, and simulation each run in their own process, so heavyweight GPU models and the physics engine never fight over one Python interpreter.

<div align="center">
  <img src="docs/architecture.svg" alt="RPent architecture" width="960"/>
</div>

## Key Features

- **LLM-in-the-loop control.** The LLM is not fine-tuned — it drives the robot purely by calling tools (`pi0_pick`, `move_to`, `rotate_wrist`, `back_project`, `finish`, …). Each tool result is fed back as multimodal context (text + rendered images), so the model reasons over what it actually sees.
- **Three-process architecture.** The **agent process** (LLM cerebrum + toolkit, no `torch`), the **env_server** (simulator + EGL rendering), and the **vla_server** (GPU policy weights) are separate processes wired by lightweight RPC. Either heavyweight process can be restarted, moved to another GPU, or pointed at a remote host independently.
- **Pluggable reasoning brains (cerebrums).** Swap the decision brain with one flag — `--cerebrum {api, claude_code, codex}` — without touching the tools or prompts:
  - `api` — a provider-agnostic tool-calling loop built on [pydantic-ai](https://ai.pydantic.dev/) (Anthropic / OpenAI / OpenAI-compatible), with prompt caching and history-image pruning.
  - `claude_code` — the [Claude Agent SDK](https://docs.claude.com/en/api/agent-sdk/overview), exposing the toolkit as an in-process MCP server.
  - `codex` — the OpenAI Codex SDK, bridged to the toolkit over an HTTP MCP server.
- **Two environments, two VLAs, one contract.** LIBERO (Pi0.5 over HTTP) and RoboCasa (RLDX-1 over socket-RPC) share the exact same env/vla process split; only the wire codec differs, chosen to fit each env's observation shape.
- **Live dashboard.** An optional `--dashboard` starts a local FastAPI monitor that streams the agent's reasoning, real-time camera / Pi0 views, an action timeline, and clip replays — with a **bilingual UI** (`--dashboard-language {en, zh-cn}`).
- **Add an environment by dropping a package on disk.** No central registry to edit — see [Adding a new environment](docs/ADD_A_NEW_ENV.md).

## How It Works

A single run is an **LLM-in-the-loop** cycle:

1. The LLM reasons about the task and calls a tool (e.g. `pi0_pick`).
2. The tool's **primitive driver** asks the `vla_server` for an action chunk (`predict` / `vla_infer`).
3. The `env_server` executes that chunk (`chunk_step` for LIBERO, stepwise `step` for RoboCasa).
4. The env renders the resulting observation and camera frames.
5. Results are turned into text + image content blocks and fed back to the LLM for the next turn.

The loop ends when the LLM calls the `finish` tool (`success` / `failure` / `stuck`) or hits `--max-turns` / `--max-episode-steps`.

## Supported Environments

<table style="width: 100%; table-layout: auto; border-collapse: collapse;">
  <thead align="center" valign="bottom">
    <tr>
      <th style="text-align: left;">Simulator</th>
      <th>VLA Policy</th>
      <th>Reasoning Brain</th>
    </tr>
  </thead>
  <tbody valign="top">
    <tr>
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
          <li><b>Pi0.5</b> (LIBERO, HTTP) ✅</li>
          <li><b>RLDX-1</b> (RoboCasa, socket-RPC) ✅</li>
        </ul>
      </td>
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
| `--cuda-device` | inherited | GPU device(s) exposed to the env / vla servers via `CUDA_VISIBLE_DEVICES` |
| `--dashboard` | off | Start the local dashboard for this run |
| `--dashboard-language` | `en` | Dashboard UI language: `en` \| `zh-cn` |
| `--vla-endpoint` | — | Reuse an already-running vla_server instead of spawning one |
| `--no-driver` | off | Attach to an existing env_server / vla_server |

## Documentation

- [Adding a new environment](docs/ADD_A_NEW_ENV.md) — plug a new simulator / robot into the runner ([中文](docs/ADD_A_NEW_ENV.zh.md)).
- [RoboCasa setup](docs/SETUP_ROBOCASA.zh.md) — RoboCasa365 + RLDX-1 install and run guide.
- [`docs/`](docs/README.md) — the full documentation index.

## Acknowledgements

RPent builds on the simulators, VLA models, and training infrastructure of [RLinf](https://github.com/RLinf/RLinf), and on the agent SDKs of the broader open-source community — [pydantic-ai](https://ai.pydantic.dev/), the [Claude Agent SDK](https://docs.claude.com/en/api/agent-sdk/overview), and the OpenAI Codex SDK. Thanks to the teams behind LIBERO, RoboCasa, robosuite, MuJoCo, and openpi.
