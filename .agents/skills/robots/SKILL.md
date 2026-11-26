---
name: robots
description: "RPent robot integrations, packaging extras, and GPU e2e (robots/, rpent/robots/, pyproject.toml, requirements/, tests/e2e_tests/, .github/workflows/embodied-gpu-e2e*.yml): add or extend a robot integration's RobotSpec, Env/VLA, toolkit, prompt, runtime, pyproject extra, and GPU e2e tests, then check and verify them. Use before editing any robot package, shared robot contract, packaging extra, or e2e suite."
paths:
  - robots/**
  - rpent/robots/**
  - pyproject.toml
  - requirements/**
  - tests/e2e_tests/**
  - ".github/workflows/embodied-gpu-e2e*.yml"
---

# Robots, Packaging, and E2E

Each concrete robot integration lives under `robots/<robot>/` and plugs into RPent through `RobotSpec`, `Toolkit`, shared RPC facades (`rpent/robots/components/`), and shared daemon helpers (`rpent/robots/runtime.py`). Dependencies live in `pyproject.toml` optional extras (plus `requirements/*-override.txt` when overrides must stay isolated), and GPU suites run through `tests/e2e_tests/run_gpu_suite.sh` and `.github/workflows/embodied-gpu-e2e*.yml`. Concrete code and packaging shapes are in [reference.md](reference.md).

## Conventions

1. **Reuse RPent and RLinf components first.** Start with [add_robot.rst](../../../docs/source-en/rst_source/development/add_robot.rst), [add_primitive.rst](../../../docs/source-en/rst_source/development/add_primitive.rst), and [add_action_model.rst](../../../docs/source-en/rst_source/development/add_action_model.rst), then read `rpent/robots/robot_spec.py`, `rpent/robots/base.py`, `rpent/robots/runtime.py`, and the closest current sibling under `robots/`. Prefer RLinf's supported Env/VLA implementations with thin RPent adapters; explain any necessary divergence in the PR.
2. **Contain robot-specific policy in `robots/<robot>/`.** Never add robot-name branches to `rpent/cli/`, `rpent/dashboard/`, `rpent/planner/`, or `rpent/session/`. Use `RobotSpec` hooks and capability flags (`validate_args`, `prepare_memory`, `prepare_prompt_vars`, `resolve_dashboard`, `finalize_run`, `run_flash`, `flywheel`, `is_real_robot`, `supports_exploration`, `supports_human_interactive_exploration`).
3. **Keep package imports lazy.** `robots/<robot>/__init__.py` and `robot_spec.py` must not import simulator, VLA, perception, or hardware SDK packages (`torch`, `mujoco`, `robosuite`, `libero`, `openpi`, `rlinf`, `sam3`, `isaacsim`, `curobo`, `pyrealsense2`, `franky_control`, `cv2`, `transformers`) at module top level. `import rpent`, `enumerate_robots()`, and `rpent --help` must work with only base dependencies.
4. **Make daemon ownership explicit.** Spawn and wait for owned subprocesses through `try_spawn_server` and `try_wait_server`, roll back earlier daemons on startup failure via `stop_owned_daemons`, handle Dashboard component subsets when supported, and never terminate borrowed external endpoints.
5. **Pin optional dependencies cleanly.** Put integration-only dependencies in a `[project.optional-dependencies]` extra in `pyproject.toml`. Pin git dependencies to an explicit ref (`@<commit|tag|branch>`). Declare mutually incompatible extras in `[tool.uv] conflicts`, and keep per-robot resolver overrides in `requirements/<robot>-override.txt` so other extras stay unaffected.
6. **Keep E2E targets aligned.** Adding a directory under `tests/e2e_tests/<robot>/` does not wire it into GPU CI. When adding a new CI target, update `tests/e2e_tests/run_gpu_suite.sh`, `.github/workflows/embodied-gpu-e2e.yml` (`options` and `/ci-*`), and `.github/workflows/embodied-gpu-e2e-run.yml` together, and document opt-in prerequisites in [tests/README.md](../../../tests/README.md).

## Write

### Add or extend a robot integration

1. **Package exports (`robots/<robot>/__init__.py`).** Expose `get_robot_spec` and `get_toolkit` (prefer lazy `__getattr__` as in `robots/libero/__init__.py` and `robots/robotwin/__init__.py`).
2. **Spec and config (`robots/<robot>/robot_spec.py`).** Populate `RobotSpec` and `RunConfig` for identity, prompts, CLI args (`add_cli_args`, `validate_args`), config parsing (`parse_config`), and runner hooks. Inspect current fields in `rpent/robots/robot_spec.py` instead of copying a stale sibling.
3. **Env/VLA RPC adapters.** Reuse `BaseEnvClient`/`BaseEnvFacade` and, when needed, `BaseVLAClient`/`BaseVLAFacade`. Match both sides of RPC routes, observation and action formats, reset/session semantics, and server constraints.
4. **Toolkit and primitives.** Connect tool schemas, handlers, primitives, and state capture. Reuse `toolkits/` for opt-in RGB-D perception when applicable. Keep large observations out of tool-result text, and check tool exposure across supported planners.
5. **Prompts.** Build `PromptBundle` from robot-owned templates. Verify rendered variables, tool names, and concise instructions, following [docs](../docs/SKILL.md) for prose quality.
6. **Runtime lifecycle.** Implement `init_runtime` with `rpent/robots/runtime.py` helpers for both complete CLI startup and selected Dashboard component subsets. Keep robot-specific result schemas in `finalize_run`.
7. **Packaging, tests, and docs.** Add or update the `pyproject.toml` extra and `conflicts`, add offline unit tests under `tests/unit_tests/robots/<robot>/`, add real-component/policy-chain e2e tests under `tests/e2e_tests/<robot>/` reusing `tests/e2e_tests/common.py`, and update paired EN/ZH docs and README highlights ([docs](../docs/SKILL.md)).

## Check

```bash
python3 .agents/skills/robots/check.py            # files changed against main
python3 .agents/skills/robots/check.py FILE ...
```

Findings use the shared format of [README.md](../README.md#findings):

| Rule | Severity | Reports |
|---|---|---|
| `robots/exports` | major | a package under `robots/<name>/` whose `__init__.py` does not expose both `get_robot_spec` and `get_toolkit` |
| `robots/lazy-import` | major | a top-level import of an optional simulator, VLA, perception, or hardware SDK package in `robots/<name>/__init__.py` or `robot_spec.py` |
| `robots/packaging` | major | `pyproject.toml` issues: `[tool.uv] conflicts` referencing an unknown extra, unpinned git dependencies in extras, or missing `robots`/`robots.*` package discovery |
| `robots/e2e-syntax` | critical | a changed shell script under `tests/e2e_tests/` that fails `bash -n` |
| `robots/e2e-targets` | major | drift between targets in `.github/workflows/embodied-gpu-e2e.yml` (`options` / `/ci-*`) and `case "$target"` in `tests/e2e_tests/run_gpu_suite.sh` |

Then judge what the script cannot:

- [ ] No robot-specific logic leaked into shared `rpent/` modules; existing RLinf/RPent Env/VLA/runtime/artifact helpers are reused.
- [ ] Startup failure tears down every earlier owned daemon and leaves borrowed endpoints running.
- [ ] Tool schemas, handlers, and rendered prompts agree on every tool name and placeholder across supported planners.
- [ ] Offline unit tests cover config, CLI args, dispatch, prompts, and lifecycle without requiring the robot extra; real-component and policy-chain checks exist for simulator integrations.
- [ ] Paired EN/ZH simulator or real-world robot guides and `tests/README.md` match actual install extras, env vars, and commands.

## Verify

1. **Offline unit tests** (always required before PR):

   ```bash
   pytest tests/unit_tests/rpent/robots/ tests/unit_tests/robots/<robot>/ -v
   ```

2. **Supported GPU e2e suites** (on a host with one dedicated NVIDIA GPU, EGL, and the target's checkpoints/assets; check `nvidia-smi` first and never interrupt another user's job):

   ```bash
   export CUDA_VISIBLE_DEVICES=<gpu-ordinal>
   export MUJOCO_GL=egl
   export PYOPENGL_PLATFORM=egl
   bash tests/e2e_tests/run_gpu_suite.sh \
     <libero-pro|robocasa|robotwin> \
     /path/to/new-output-dir \
     /path/to/new-venv-root
   ```

   Inspect JUnit XML, server logs, and `*-daemon-cleanup.txt` in the output directory. For opt-in simulator suites not wired into `run_gpu_suite.sh` (such as RoboDojo, EFE, or external WAM endpoints) or operator-controlled real-robot diagnostics (`dual_franka`), follow the exact prerequisites and commands in [tests/README.md](../../../tests/README.md).

3. **Distinguish integration from task success.** A bounded policy-chain test proves action execution, recorded `finish`, saved state/observation artifacts, and owned-daemon cleanup; report native environment success (`terminated` or benchmark verifier reward) separately.

## When to stop and ask

Stop and ask the user when an integration dependency conflicts with a core RPent pin (`pydantic`, `fastapi`, `numpy`, `pyarrow`, `huggingface_hub`) and neither an isolated extra conflict nor a `requirements/<robot>-override.txt` is clearly right; when required simulator assets or checkpoints are unavailable for requested GPU verification; or when physical robot hardware requires operator supervision.
