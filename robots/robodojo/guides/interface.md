# RoboDojo Backend Interface

Setup and CLI configuration are documented in the paired installation guides:
[English](../../../docs/source-en/rst_source/usage/robodojo/installation.rst)
and [Chinese](../../../docs/source-zh/rst_source/usage/robodojo/installation.rst).

## Runtime Ownership

`robot_spec.py` registers the backend and its CLI options. Runtime hooks use
the shared spawn/wait helpers and return only owned daemons for cleanup.
Explicit endpoints attach to borrowed services without launching a process.
Source imports and Python executable paths come from CLI arguments; no
workspace environment file is loaded.

Isaac Sim must initialize before simulator imports. The environment server
serializes environment requests on its main thread because camera rendering
is not safe on RPC worker threads. Each simulator process owns one application.

## RPC and Observations

`env_client.py` and `env_server.py` adapt the shared BaseEnvClient/BaseEnvFacade
contracts. Observations contain camera RGB-D/calibration data and arm/gripper
state. The backend supports joint and end-effector actions; motion requests
must remain on the environment action path so its bounds and counters apply.

`vla_client.py` uses the shared model RPC contract. The shared
`rpent.robots.components.pi05_vla_server` selects the XPolicyLab adapter with
`--policy-backend xpolicylab`; the default `rlinf` backend retains its own loader
and observation contract. Both implement `BaseVLAFacade`. The policy process
has its own Python environment. The CLI passes `--policy-root`, derived from
the explicit XPolicyLab checkout, to locate the policy launcher.
The adapter passes observations and actions through unchanged and serializes
`update_obs`/`get_action` with `reset`. It does not provide session isolation.

## Tool State Capture

The toolkit uses shared tool registration and EnvState artifact ownership.
`view_env_state`, `back_project`, `segment`, `get_reward_details`, and
`get_safety_status` are read-only: they do not advance the environment.
Observation cache population and segmentation inference do not constitute
robot actions. Motion, gripper, policy, and stabilization tools are not
read-only and retain post-action state capture.

Offline contracts live under `tests/unit_tests/robots/`. Unit-test success
does not establish simulator compatibility or benchmark task success.

## Shared Perception and Task Context

`rpent.robots.components.perception_tools` owns recorded-state reading (also
used by LIBERO) and calibrated depth projection. RoboDojo supplies image names
and the negative optical Z convention. SAM3 inference already uses the shared
client. LIBERO/RoboCasa world-map lookup, segmentation artifact handling,
OSC controllers, and gripper units differ from RoboDojo's CuRobo dual-arm
actions, so these backend adapters remain separate.

The planner injects tools; call the names in its tool list. The generic system
prompt contains no dustbin instructions. Only the `put_bottles_into_dustbin`
user context supplies placement, bottle alarm, and scoring guidance.

## Information Groups and Registration Hook

`tools.TOOL_GROUPS` classifies direct robot-tool outputs:

- `general`: `back_project`, `segment`, `move_to`, `pi0_pick`, `stabilize`.
  World coordinates from calibrated depth and robot proprioception are not
  ground-truth object poses; policy grasp success is a proprioceptive heuristic.
- `privileged`: `get_reward_details` exposes per-object predicates, score and
  success; `get_safety_status` exposes ground-truth object positions and motion.
- `mixed`: `view_env_state` includes environment termination and historical
  results; `set_gripper` returns status containing success and safety alarms;
  `place_in_bin` can include that status in its release result.

`get_toolkit(..., allowed_tool_groups=frozenset({"general"}))` filters robot
schemas and handlers together. None keeps the existing task-dependent tool set;
an empty set disables robot tools; unknown groups raise an error. Common file
tools remain available. This is not an eval mode: automatic post-action state,
stored logs, files/memory, and environment observation state still require
output filtering and an explicit policy. Observation `state` is currently
passed through without a field allowlist, so any upstream object poses would
also be exposed. Do not claim perception isolation from registration alone.
