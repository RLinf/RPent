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

`vla_client.py` and `vla_server.py` adapt the shared model RPC contract to
XPolicyLab Pi_05. The policy process has its own Python environment.
The wrapper uses `ROBODOJO_PI05_POLICY_ROOT`, derived from the explicit
XPolicyLab checkout by the CLI, to locate the policy launcher.

## Tool State Capture

The toolkit uses shared tool registration and EnvState artifact ownership.
`view_env_state`, `back_project`, `segment`, `get_reward_details`, and
`get_safety_status` are read-only: they do not advance the environment.
Observation cache population and segmentation inference do not constitute
robot actions. Motion, gripper, policy, and stabilization tools are not
read-only and retain post-action state capture.

Offline contracts live under `tests/unit_tests/robots/`. Unit-test success
does not establish simulator compatibility or benchmark task success.
