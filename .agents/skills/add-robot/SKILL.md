---
name: add-robot
description: Add an RPent robot integration or extend its Env/VLA, tool, prompt, configuration, or runtime wiring using existing robot contracts and validation tools.
---

# Add or extend a robot integration

Start with the [robot guide](../../../docs/source-en/rst_source/development/add_robot.rst)
and [primitive guide](../../../docs/source-en/rst_source/development/add_primitive.rst).
Read the current [RobotSpec](../../../rpent/robots/robot_spec.py),
[registry](../../../rpent/robots/base.py), and
[runtime helpers](../../../rpent/robots/runtime.py), then the closest existing
robot package. The guide uses LIBERO as a worked example; select a sibling that
fits the target's actual control, observation, and lifecycle contracts.

## Establish ownership and scope

Identify the required environment, action model, tools, observations, and
supported entry points. Check whether RPent or RLinf already owns each
capability. Prefer shared Env/VLA components and thin adapters over duplicated
simulation, action conversion, success logic, or lifecycle machinery. Explain
any necessary divergence in the PR.

For an existing robot, follow only the steps affected by the change. Keep
robot-specific choices in `robots/<robot>/` and optional dependencies in the
appropriate extra with lazy imports.

## Wire the integration

1. Expose the package factories through the existing registry. Use `RobotSpec`
   and `RunConfig` for identity, prompts, CLI parsing, and runner hooks; inspect
   current capability fields and consumers instead of copying a stale spec.
2. Reuse `BaseEnvClient`/`BaseEnvFacade` and, when needed,
   `BaseVLAClient`/`BaseVLAFacade`. Match both sides of RPC routes, observation
   and action formats, reset/session semantics, and server-side constraints.
3. Connect the toolkit's schemas, handlers, primitives, and state capture.
   Reuse existing artifact and memory ownership; keep large observations out
   of tool-result text. Check actual tool exposure across supported planners.
4. Build the shared `PromptBundle` from robot-owned content. Verify rendered
   variables, tool names, concise instructions, and consistency with the
   toolkit. Follow [docs-check](../docs-check/SKILL.md) for model-facing prose.
5. Use runtime helpers for owned daemons and borrowed endpoints. Cover startup
   failure and cleanup; handle complete CLI startup and selected Dashboard
   components when supported. Keep robot-specific result schemas in the
   robot's finalization hook and verify which entry points call it.
6. Update the relevant installation extra, usage/development documentation,
   and tests. Follow the actual runner and CI configuration when adding a new
   supported test target; adding a directory alone does not wire it into CI.

## Validate and hand off

Follow [verify-change](../verify-change/SKILL.md) and
[tests/README.md](../../../tests/README.md). Reuse offline tests for config,
dispatch, and lifecycle, and existing coverage for unchanged shared components.
For simulator integrations, exercise each new real component/adaptation and
the bounded policy chain with the helpers in
[tests/e2e_tests/common.py](../../../tests/e2e_tests/common.py). Use the documented
operator-controlled diagnostic path for physical hardware.

Report the reused components, necessary adapter differences, supported entry
points, and validation evidence. Identify unavailable hardware/model checks
and distinguish runtime integration from native task success.
