# RPent code and design patterns

This page is the standard for code that goes into RPent: authors follow it while writing a change, and reviews measure PRs against it. It distills [AGENTS.md](../../../AGENTS.md) and [CONTRIBUTING.md](../../../CONTRIBUTING.md) into patterns you can spot in a diff. A finding that cites a pattern names the pattern, the code that breaks it, and the version that follows it. Documentation follows the [docs](../docs/SKILL.md) skill, and robot integration/packaging specifics follow the [robots](../robots/SKILL.md) skill.

## Design philosophy

RPent coordinates heterogeneous planners, robot toolkits, RPC-backed services, and task memory through a small set of explicit contracts. `RobotSpec` describes a robot's identity, prompts, CLI arguments, config parsing, runtime lifecycle, and optional hooks (`prepare_memory`, `prepare_prompt_vars`, `resolve_dashboard`, `finalize_run`, `run_flash`, `flywheel`). `Toolkit` owns tool schemas, admission/cancellation scheduling, and execution. `Planner` adapts each reasoning backend (`api`, `claude_code`, `codex`, `gemini_cli`, `flash`) to the same tool and prompt interfaces. A new robot, planner, action model, or perception tool plugs into these contracts instead of opening a parallel path through the CLI or session runner.

1. **One general mechanism, with each robot or planner as an instance of it.** Extend `RobotSpec`, `Toolkit`, `Planner`, or shared RPC clients/facades until the new case falls out naturally, rather than branching on robot or planner names in shared code. See [Generalize; don't special-case](#generalize-dont-special-case).
2. **Each layer knows only its own concern.** The CLI and Dashboard orchestrate runs without knowing robot internals; planners drive tool calls without knowing simulator APIs; `robots/<robot>/` owns robot-specific policy, prompts, and adapters. Shared code meets concrete robots only through `RobotSpec`, `Toolkit`, and shared RPC contracts. See [Contain component code](#contain-component-code) and [Fix at the owning layer](#fix-at-the-owning-layer).
3. **Every resource has one owner.** A subprocess daemon (`ProcessDaemon`), RPC client, lock, temp directory, or borrowed endpoint has one explicit owner and lifecycle. Owned daemons are registered with `try_spawn_server`/`try_wait_server` and stopped in reverse order on startup failure or exit; borrowed external endpoints are never stopped by RPent. See [Small abstractions with one owner](#small-abstractions-with-one-owner).
4. **Keep the common path direct.** Avoid extra indirection, forwarding wrappers, or dynamic dispatch when a direct method or `RobotSpec` hook expresses the behavior clearly. Keep optional simulator/model imports lazy so basic imports and CLI help remain fast and dependency-free. See [Keep the common path direct](#keep-the-common-path-direct) and [Optional dependencies and lazy imports](#optional-dependencies-and-lazy-imports).
5. **Validate at the edge, trust inside.** Validate CLI arguments (`RobotSpec.validate_args`), config files, RPC payloads, and external tool/model responses once at the boundary that owns them. Inside RPent, trust typed contracts and keep control flow direct. See [No unnecessary protection code](#no-unnecessary-protection-code).
6. **Changes are proven, not asserted.** Preserve existing behavior across planners, CLI, Dashboard, and robots unless a difference is named, documented, and tested. Write offline CPU unit tests through public contracts with fakes only at external edges. See [Preserve behavior](#preserve-behavior) and [Tests](#tests).
7. **Clear code over clever code.** One stable name per concept across code, CLI flags, prompts, and docs; explicit types and concise contract comments. A good change reads as if it had always belonged in RPent. See [Code](#code).

## Design

### Generalize; don't special-case

Expect a change to plug into RPent's existing extension points:

- `RobotSpec` hooks and capability flags in `rpent/robots/robot_spec.py` (`add_cli_args`, `validate_args`, `parse_config`, `init_runtime`, `prepare_memory`, `prepare_prompt_vars`, `resolve_dashboard`, `finalize_run`, `run_flash`, `flywheel`, `is_real_robot`, `supports_exploration`, `supports_human_interactive_exploration`).
- `Toolkit` (`rpent/tools/toolkit.py`) and shared RGB-D adapters (`components/tools/rgbd/`).
- `Planner` (`rpent/planner/base.py`).
- Shared Env/VLA/SAM3 RPC bases in `rpent/components/` and lifecycle helpers in `rpent/robots/runtime.py`.

Flag:

- `if args.robot == ...`, `if spec.name in (...)`, or `if planner == ...` in `rpent/cli/`, `rpent/dashboard/`, `rpent/session/`, or `rpent/tools/`.
- A copy of an existing client, server, toolkit, or runner with a few lines changed for a new robot or action model instead of parameterizing or subclassing the shared base.
- A new CLI flag or config key whose only purpose is toggling a one-off branch in shared code.

Ask for a hook or capability field on `RobotSpec`, a method on `Toolkit` or `Planner`, or a parameter on the shared RPC component.

### Contain component code

Expect each component to live in its own module:

| Component | Home |
|---|---|
| Robot integration (spec, toolkit, env/VLA adapters, prompts) | `robots/<robot>/` |
| Shared robot discovery, spec, runtime, RPC components | `rpent/robots/`, `rpent/components/` |
| Simulator and model services | `components/sim/<name>/`, `components/models/<name>/` |
| Planner adapter | `rpent/planner/<name>.py` |
| Tool container, common/HIL/world-coordinate tools | `rpent/tools/` |
| Opt-in RGB-D perception and data flywheel tools | `components/tools/rgbd/`, `components/tools/flywheel/` |
| Session, observation capture, prompt building, memory | `rpent/session/`, `rpent/prompt/`, `rpent/memory/` |
| CLI, Dashboard, evaluation, installer | `rpent/cli/`, `rpent/dashboard/`, `rpent/evaluation/`, `rpent/install/` |
| Component dependencies, overrides, and conflict rules | `component.toml` manifests (`robots/`, `components/`) |

Flag robot-specific prompt text, simulator assumptions, or coordinate transforms placed in `rpent/cli/`, `rpent/planner/`, or `rpent/utils/`.

### Small abstractions with one owner

Expect:

- One responsibility and one stable name per class, function, CLI option, and prompt variable.
- Explicit resource ownership: register every spawned subprocess in `daemons` via `try_spawn_server` / `try_wait_server`, roll back earlier daemons on startup failure via `stop_owned_daemons`, make `close()` / `stop()` idempotent, and never kill externally supplied endpoints.
- Invalid states ruled out by types and boundary checks rather than silent fallbacks.

Flag:

- Wrapper functions or manager classes that only forward one call.
- Dynamic `getattr`/`setattr`/`importlib` tricks or monkey-patches where a direct method or `RobotSpec` field works.
- Reaching into private attributes (`obj._client`, `obj._server`) across module boundaries.
- Parallel vocabularies: two names, schemas, or flags for one concept.

### Fix at the owning layer

Expect a bug fix in the layer that owns the invariant so every robot and planner benefits. Flag call-site band-aids, `# noqa` comments that hide real lint issues, and robot-level workarounds for bugs in shared `rpent/` helpers.

### No unnecessary protection code

Expect validation once at the boundary: CLI argument validation (`RobotSpec.validate_args`), config/YAML loading, RPC request/response decoding, and external model/simulator results. Error messages state what was expected and what was found, preserving root causes (`raise ... from exc`).

Flag:

- `hasattr`, `getattr(x, "y", None)`, or `isinstance` ladders for attributes and types that internal contracts guarantee.
- Replacing direct member access (`obj.attr`) with `getattr`/`hasattr` or adding fallback guards in product code so incomplete mocks or stubs in tests pass; fix the test to construct the real object or satisfy its contract instead.
- `if x is None` checks for non-optional values, or silent defaults that mask missing configuration.
- Broad `try: ... except Exception:` blocks that log and keep running in a broken state, or bare `except: pass`.
- Unjustified retries, `time.sleep` polling loops, or compatibility shims with no real consumer.

### Keep the common path direct

Keep local in-process or direct RPC calls straightforward. Introduce extra threads, background processes, locks, or abstraction layers only when concurrency or isolation genuinely requires them, and ensure cancellation and shutdown drain cleanly (`Toolkit._scheduler`, `ProcessDaemon.stop`).

### Preserve behavior

A refactor preserves behavior unless the PR names, documents, and tests the difference. Trace affected planners (`api`, `claude_code`, `codex`, `gemini_cli`, `flash`), CLI and Dashboard runs, memory sync, and robots against `main`. Before deleting a field, CLI flag, or prompt variable, grep all callers, tests, and docs.

### Optional dependencies and lazy imports

Simulators (`mujoco`, `robosuite`, `robocasa`, `libero`, `isaacsim`, `curobo`), VLA/perception stacks (`torch`, `openpi`, `rlinf`, `sam3`, `transformers`), and hardware SDKs (`pyrealsense2`, `franky_control`, `rospy`) are not installed in every environment. Import them inside the function, method, or server entry point that uses them. `import rpent`, `enumerate_robots()`, and `rpent --help` must succeed with only base dependencies installed.

### Prompts and tool contracts

Prompts and tool descriptions are model inputs, not developer prose:

- Keep instructions focused on the model's task and remove repetition without dropping operational constraints.
- Verify rendered prompt variables and tool names match what `Toolkit.list_tools()` exposes across all supported planners.
- Keep large binary payloads (images, point clouds, raw depth arrays) out of tool-result text; save artifacts to disk and return concise metadata/paths.
- Never conflate planner `finish` calls with native environment success (`terminated`).

## Code

- Logging uses `rpent.utils.logging.get_logger(...)`. Never use `print` in runtime code (CLI entry points that intentionally write structured stdout are the explicit exception).
- Public Python APIs have type hints and concise Google-style docstrings.
- Comments state contracts, invariants, ownership, and non-obvious reasons in one or two sentences. They do not narrate code, repeat signatures, hedge, or record PR review exchanges or fix history.
- Exception and assertion messages state what was expected and what was found.
- New Python files start with `# Copyright <current year> The RPent Authors.` and the Apache-2.0 header; moved files keep their original year; vendored files keep upstream headers.
- Every new function, method, field, CLI flag, and prompt variable has a real caller or consumer. Leave no commented-out code, debug statements, or hardcoded machine paths/credentials.
- Keep diffs focused on the task; avoid drive-by formatting or unrelated refactors.

## Tests

- Mirror production layout: `tests/unit_tests/rpent/`, `tests/unit_tests/robots/<robot>/`, `tests/unit_tests/components/`, `tests/unit_tests/installs/`, `tests/unit_tests/tooling/`, `tests/unit_tests/skills/`.
- Unit tests run offline on CPU. Use real objects, real local loopback transports (HTTP/socket/MCP), or fakes at external edges (LLM APIs, simulators, hardware SDKs).
- Flag tests that are mostly mock setup, tests that only assert a mock was called, tests that mirror internal implementation details rather than observable contracts, and tests that still pass without the change.
- Never weaken or delete a failing test to make CI pass.
