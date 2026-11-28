# AGENTS.md

This file tells coding agents how to build, test, and change RPent so that the result passes CI and review. `CLAUDE.md` is a symlink to it, and [`.gemini/settings.json`](.gemini/settings.json) loads it for Gemini CLI. It holds only rules that apply across the repository: the contribution process is in [CONTRIBUTING.md](CONTRIBUTING.md), writing and documentation rules are in the docs skill's [rules](.agents/skills/docs/rules/base.md) (indexed by [docs/STYLE_GUIDE.md](docs/STYLE_GUIDE.md)), and step-by-step workflows are skills under `.agents/skills/`. When this file disagrees with the code, the code is right; fix this file in the same change.

RPent is a recursive physical agent framework that coordinates high-level planners (`api`, `claude_code`, `codex`, `gemini_cli`, `opencode`, `flash`) with robot-owned toolkits, RPC-backed Env/VLA/perception services, shared prompts, and task memory. A run loads a robot's `RobotSpec` and `Toolkit` through `rpent/robots/base.py`, spawns or connects to its runtime services via `rpent/robots/runtime.py`, runs a planner session or Dashboard loop, and saves step artifacts and final evaluation status. User and developer documentation lives under `docs/source-en/rst_source/` and `docs/source-zh/rst_source/`.

## Working with a human submitter

- A human submits and answers for every change. Push, open or edit PRs and issues, and post review comments only when the user asks for that action and has seen the content.
- Before starting a fix or feature, search open PRs and issues for the same work (`gh pr list --repo RLinf/RPent --search "<keywords>"`) and tell the user about any overlap.
- When a requirement is ambiguous or a constraint blocks the change, stop and ask. Do not leave `TODO(agent)` or similar placeholders in code; record known limitations in the PR description.
- Report results as they are: which tests ran, on what hardware, and which failed or were skipped. Keep bounded integration evidence distinct from native task success, and state in the PR description that an AI assistant was used.

## Setup and commands

Create an isolated Python 3.10–3.12 environment (3.12 default when unconstrained; per-component newest supported minor) with `./rpent-install.sh` and install only the component your change needs:

```bash
./rpent-install.sh install infra-dev --venv .venv-dev && source .venv-dev/bin/activate
./rpent-install.sh install infra-unit_cpu --venv .venv-unit-cpu  # full CPU unit suite
./rpent-install.sh install robot-libero --venv .venv-libero  # or robot-robocasa / tool-flywheel
```

Lint runs through pre-commit: Ruff lint (`--preview --fix`) and format, plus Conventional Commit checks on commit messages:

```bash
uv pip install pre-commit==4.6.2
pre-commit install
pre-commit run --all-files
```

Unit tests run offline on CPU:

```bash
pytest tests/unit_tests -v
pytest tests/unit_tests/rpent/cli/test_main_contracts.py -k xxx -x
```

Supported GPU end-to-end suites run through `tests/e2e_tests/run_gpu_suite.sh` on a host with a dedicated NVIDIA GPU, EGL, and the target's assets/checkpoints (see [Testing and CI](https://rpent.readthedocs.io/en/latest/rst_source/development/testing.html)):

```bash
bash tests/e2e_tests/run_gpu_suite.sh <target> /path/to/new-output-dir /path/to/new-venv-root
```

Docs must build in both languages with no warnings. The docs check builds both trees the way Read the Docs does and checks the changed pages:

```bash
python3 .agents/skills/docs/check.py
```

## Repository map

- `rpent/planner/`, `rpent/session/`, `rpent/prompt/` – planner adapters (`api`, `claude_code`, `codex`, `gemini_cli`, `opencode`, `flash`), execution sessions, observation/step capture, and shared prompt construction.
- `rpent/tools/` and `rpent/memory/` – `Toolkit` container, scheduling/cancellation, common/HIL tools, and task memory sync and retrieval.
- `rpent/robots/` and `rpent/components/` – robot discovery (`base.py`), `RobotSpec`/`RunConfig`/`PromptBundle`, shared daemon lifecycle (`runtime.py`), and shared Env/VLA/perception contracts and facades (`rpent/components/env/`, `rpent/components/models/`).
- `rpent/cli/`, `rpent/dashboard/`, `rpent/evaluation/`, `rpent/install/`, `rpent/utils/` – CLI entry points, interactive Dashboard, run finalization, component installer (`component.toml` manifests, resolution, fingerprints), and RPC/daemon/logging helpers.
- `robots/<name>/` – concrete robot integrations (`libero`, `robocasa`, `robotwin`, `robodojo`, `robolab`, `efe`, `franka`, `dual_franka`, `yam`), each owning its `component.toml` and exposing `get_robot_spec` and `get_toolkit` (plus manifest-only dependency variants such as `robots/libero_plus/`).
- `components/{sim,models,tools,infra}/<name>/` – installable simulator/model services (including manifest-only variants such as `components/sim/libero_plus/`), tools (e.g. `components/tools/flywheel`, RGB-D `components/tools/rgbd`), and manifest-only infrastructure environments (`infra-dev`, `infra-unit_cpu`).
- `rpent-install.sh`, `tests/` (`unit_tests/`, `e2e_tests/`, `tooling/`), and `docs/` (`source-en/`, `source-zh/`, `STYLE_GUIDE.md`).

The dependency direction is `rpent` ← `components/sim` ← `components/models` ← `robots` (enforced by `tests/tooling/layout.py`): core `rpent/` discovers components dynamically and never imports concrete slices statically, tools depend only on `rpent/` and model clients, and sibling simulators/models/tools do not import each other.

## Rules CI and reviewers enforce

- **Optional dependencies and lazy imports.** Simulators, VLA models, and hardware SDKs are not installed in every environment. Import them inside the function or branch that uses them so `import rpent`, `enumerate_robots()`, and `rpent --help` work in a core-only environment. Put component-only dependencies in `component.toml` manifests rather than `pyproject.toml`.
- **Package layout and headers.** Every directory containing Python files under `rpent/`, `robots/`, and `components/` has an `__init__.py`, and every Python file starts with the Apache-2.0 header `# Copyright <year> The RPent Authors.` (Ruff `CPY001`).
- **Ruff and style.** Pre-commit runs Ruff (`--preview`) and Ruff format. Public Python APIs need type hints and Google-style docstrings.
- **Comments and docstrings.** Keep them short, direct, and written for someone reading the current code. State the contract, invariants, ownership, and non-obvious reasons; do not narrate the implementation, repeat the signature, hedge, or record review/fix history.
- **Logging and errors.** Use `rpent.utils.logging.get_logger()`, never `print`, in runtime code. Error and assertion messages say what was expected and what was found, preserving original exceptions (`raise ... from exc`).
- **Runtime ownership and success semantics.** Use `try_spawn_server`, `try_wait_server`, and `stop_owned_daemons` in `rpent/robots/runtime.py`; tear down earlier owned daemons on startup failure and never stop borrowed external endpoints. Keep planner `finish` calls distinct from the environment's native success signal (`terminated`).
- **Prompts and tool descriptions.** Keep them focused on the model's task, keep large observations out of tool-result text, and verify tool exposure across supported planners.

## Extending RPent

Plug into an existing contract instead of adding robot- or planner-name branches to shared code.

| Component | Extension point | Guide |
|---|---|---|
| Robot / simulator | `robots/<robot>/` exposing `get_robot_spec()` (`RobotSpec`) and `get_toolkit()` (`Toolkit`) | `docs/source-en/rst_source/development/add_robot.rst` |
| Planner | `rpent/planner/` subclassing `Planner` | `docs/source-en/rst_source/development/add_planner.rst` |
| Action model (VLA/WAM) | `BaseVLAClient`/`BaseVLAFacade` in `rpent/components/models/` | `docs/source-en/rst_source/development/add_action_model.rst` |
| Primitive / tool | `Toolkit.add_tool` in `rpent/tools/toolkit.py` | `docs/source-en/rst_source/development/add_primitive.rst` |
| RGB-D perception tool | `components/tools/rgbd/` adapters over robot observations | `docs/source-en/rst_source/toolkits/rgbd.rst` |

## Tests

- Mirror production layout under `tests/unit_tests/rpent/`, `tests/unit_tests/robots/<robot>/`, `tests/unit_tests/components/`, `tests/unit_tests/installs/`, `tests/unit_tests/tooling/`, and `tests/unit_tests/skills/` (see [Testing and CI](https://rpent.readthedocs.io/en/latest/rst_source/development/testing.html)).
- Test observable contracts and concrete regressions offline on CPU; use real local loopback transports or fakes at external edges (LLM APIs, simulators, hardware SDKs).
- Never weaken product code (for example replacing direct member access with `getattr`/`hasattr` or adding guard branches) so an incomplete mock or stub test passes; construct the real object or satisfy its contract in the test.
- Never weaken or delete a failing test to make CI pass. Delete a test when you cannot name the change that would make it fail.

## Design and review principles

- Preserve behavior before simplifying. Trace the full call path across planners, CLI, Dashboard, and robots; a difference is intentional only when named, documented, and tested.
- Prefer small, explicit abstractions with one responsibility and one owner. Keep robot-specific policy inside `robots/<robot>/` and reuse RPent and RLinf components with thin adapters.
- Validate external inputs once at the owning boundary (`RobotSpec.validate_args`, CLI parsing, RPC edges). Inside RPent, trust established contracts and keep control flow direct.
- Review the whole contract, not only the newest diff: callers, subclasses, schemas, prompts, tests, and paired EN/ZH docs.

## Commits, PRs, and writing

- Commit subjects follow Conventional Commits (`<type>(<scope>): <description>`, imperative).
- PR titles keep the description after `: ` at most 72 characters, starting with a lowercase letter, imperative mood, plain ASCII, no trailing period. Use `create-pr` to fill `.github/PULL_REQUEST_TEMPLATE.md` and run `python3 .agents/skills/create-pr/lint_pr.py lint --title "<title>" --body-file <body.md>`.
- All project writing—docs, PR descriptions, review comments, and commit messages—follows [english.md](.agents/skills/docs/rules/english.md) and [chinese.md](.agents/skills/docs/rules/chinese.md). Change English and Chinese docs together, writing Chinese natively instead of translating sentence by sentence.

## Skills

Skills live in `.agents/skills/<name>/SKILL.md` ([.agents/skills/README.md](.agents/skills/README.md)). An area skill owns part of the repository: read it before editing a file it owns, even when that edit is part of a larger change.

| Files | Skill |
|---|---|
| Code, tests, tools, and CI: `rpent/`, `robots/`, `components/`, `tests/`, `.github/`, `pyproject.toml`, `setup.py`, any `.py` file | `code` |
| Docs: `docs/`, `README.md`, `README.zh-CN.md`, `CONTRIBUTING.md`, `CONTRIBUTING_zh.md` | `docs` |
| Robots, component manifests, and GPU e2e: `robots/`, `components/`, `rpent/robots/`, `rpent/components/`, `pyproject.toml`, `tests/e2e_tests/`, `tests/tooling/ci_targets.py`, `.github/workflows/embodied-gpu-e2e*.yml` | `robots` |

| Task | Skill |
|---|---|
| Open a PR, or fix a PR title or description | `create-pr` |
| Review a PR (or local diff), or post review comments on it | `review-pr` |

Before reporting a change as done, run `python3 .agents/skills/route.py check`. It finds the areas the change touches and runs each one's check; fix every critical or major finding, or tell the user why it stays.

## Maintaining agent instructions

- This file is loaded on every agent request. Keep it under 200 lines, and add a rule only when it applies across the repository and an agent would get it wrong without being told.
- Put area-specific guidance in a skill under `.agents/skills/`. Edit canonical files under `.agents/skills/` so Codex, Cursor, Claude Code (`.claude/skills`), and Gemini CLI (`.gemini/settings.json`) share the same instructions.
- Refer to code by symbol or search pattern instead of copying lists that drift. Check every path, command, and API you add against the code.
- An area skill declares `paths:` and ships `check.py` ([.agents/skills/README.md](.agents/skills/README.md)).
