# Guide, Toolkit, Quick Start, and Demo Pages

Pages under `get_started/`, `guides/`, `toolkits/`, and `real_world_demos/` help a user accomplish a concrete task: install and run their first LIBERO-PRO task, configure planners, manage memory, use Dashboard, Flash Mode, RGB-D perception, or external MCP tools, collect flywheel trajectories, or run remote/parallel services.

## Quick Start (`get_started/quickstart.rst`)

Follow this fixed sequence:

1. Requirements (OS, Python, GPU, credentials).
2. Installation (`uv venv`, editable install).
3. Assets and checkpoints.
4. Planner connection check (`rpent-check-llm`).
5. One runnable LIBERO-PRO task.
6. Expected artifacts and native success criterion.
7. Common failures.

Keep complete flag tables in `guides/cli.rst` and per-robot install commands in each robot's own guide rather than maintaining a duplicated install-extras matrix.

## User guides (`guides/`, `toolkits/`)

- **Task-oriented title and headings.** Name the task in the reader's words (`Configure Planners`, `Interactive Usage`, `External MCP Tools`, `Data Flywheel`), not internal module names.
- **When you need it.** Open with 1–2 sentences on what the reader will accomplish and what changes once configured.
- **Prerequisites** before the first command, with links to Quick Start or simulator guides.
- **One clean main path.** Introduce each command in one sentence, highlight only the 3–5 flags/env vars the reader sets, and show the expected output or saved artifacts. Link to `guides/cli.rst` for full option tables.
- **Mode-specific notes.** In `guides/dashboard.rst` (Interactive Usage), cover terminal and Dashboard operation together, including planner-specific differences and environment restrictions. Put operational footguns in `.. warning::` right beside the affected step.

## What a guide leaves out

Keep user guides focused on running and configuring RPent; do not mix developer internals into user pages:

| Belongs in `development/` or `guides/cli.rst` | Keep in the user guide instead |
|---|---|
| `rpent/`, `robots/`, and `components/` module tours, class hierarchies, and internal RPC/scheduler flow | Observable CLI behavior, config keys, tool names the planner calls, and saved run artifacts |
| How to implement a new `RobotSpec`, `Toolkit`, `Planner`, `BaseEnvFacade`, or `BaseVLAClient` | Link to `development/add_robot.rst`, `development/add_primitive.rst`, `development/add_action_model.rst`, or `development/add_planner.rst` |
| Exhaustive CLI flag matrices and internal storage schemas | The 3–5 flags for the main path, plus a link to `guides/cli.rst` |
| Per-robot install extras and hardware calibration steps | Link to the matching page under `simulators/` or `real_world_robots/` |
| Non-runnable placeholder snippets (`example.com`, fictional flags or paths) | Copyable commands and configs grounded in shipped defaults and examples |

## CLI Reference (`guides/cli.rst`)

`guides/cli.rst` is the shared reference for CLI options, defaults, environment variables, and saved output files:

- Group flags consistently, stating type, default, and which robot or mode owns each default.
- Exact flag and command identifiers may serve as headings here.
- Keep internal storage formats separate from public CLI inspection tools.

## Real-world demo pages (`real_world_demos/`)

Describe the task, embed the video with controls and a direct HTTP 200 URL, link to the matching installation and usage guide under `real_world_robots/`, and mark any pending instructions clearly.

## Checklist

- [ ] Title and headings name tasks (or exact flags on `guides/cli.rst`), not internal classes or modules.
- [ ] Prerequisites precede the first command, and one copyable main path runs end to end (no placeholder domains like `example.com`).
- [ ] No internal `rpent/`, `robots/`, or `components/` source paths, class inventories, or duplicated `development/` content.
- [ ] Every flag, default, and output artifact matches the current implementation.
