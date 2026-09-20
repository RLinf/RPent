# Lynsense Read-Only Backend

RPent's native `lynsense` extension reads the right arm's ROS state. It does
not depend on `lynsense_pytrees`. There are no motion primitives, wave skill,
gripper actions, enable/reset commands, shell tools, or file/image tools.

## Current Status

Implemented locally with fake ROS modules and offline model responses.
The regression gate currently passes 175 tests on Python 3.10.20. No robot
connection, deployment, real message subscription, or model API request was
performed during implementation. This is not real-robot acceptance.

The local `.venv` contains RPent's approved base/test dependencies, not a
validated ROS runtime. `pyproject.toml` is unchanged.

## Call Path

```text
RPent CLI -> RobotSpec.init_runtime -> inert adapter
         -> get_toolkit -> register read_robot_state -> adapter.connect
         -> API Planner -> read_robot_state({}) -> copied state snapshot
         -> ordinary text completion -> Toolkit.close -> ROS cleanup
```

`init_runtime` allocates no ROS resources. The factory consumes its adapter
once via an atomic ownership claim, even when its argument dictionary was
copied. It connects after toolkit construction and closes on failure.
The model sees exactly one tool, `read_robot_state`, with no parameters.
The API Planner does not inject `read_image` for this backend.

| Default Topic | Type |
| --- | --- |
| `/right_xarm/joint_states` | `sensor_msgs/msg/JointState` |
| `/right_xarm/robot_states` | `xarm_msgs/msg/RobotMsg` |

Both streams must deliver a fresh sample before connection succeeds.
Joint names/positions are validated; malformed data is never reported as `ok`.
`RobotMsg` establishes receipt only: its diagnostics are not interpreted.

Snapshots contain `status`, `reason`, `observed_at`, `joint_names`,
`positions`, `robot_state_received`, `age_s` for `joints` and `driver`,
and `warnings`. Valid optional `velocities` and `efforts` are included;
invalid optional arrays are omitted with warnings. Missing data is null,
never invented zeros. Status is `ok`, `unavailable`, `invalid`, `stale`,
`failed`, or `closed`.

Each age uses its own local monotonic receipt time. `observed_at` is the UTC
joint receipt time, not sensor sampling time. A recent receipt does not prove
synchronized samples or a healthy, idle, or safe-to-move robot. Retained old
values must always be interpreted together with their non-ok status.

## Offline Verification

From RPent, using the approved local environment:

```bash
.venv/bin/python -m pytest tests/unit_tests/robots/lynsense \
  tests/unit_tests/rpent/robots/test_registry_contracts.py \
  tests/unit_tests/rpent/cli/test_main_contracts.py \
  tests/unit_tests/rpent/planner/test_api_contracts.py \
  tests/unit_tests/rpent/tools/test_toolkit_contracts.py -q
```

Coverage includes data validation, independent freshness, missing packages,
Domain mismatch, foreign-context rejection, partial initialization, callback
failure, cancellation, cleanup retry, the final model tool table, and the
actual CLI/API Planner with `FunctionModel`. Lynsense tests prohibit socket
network access; fake ROS nodes reject control factories.

Independent review identified and prompted regressions for automatic cleanup
after executor faults, copied/concurrent ownership claims, and retrying an
owned context shutdown without touching a replacement context. Ruff and
pre-commit were not available in the approved environment and were not run.
All three findings were closed in follow-up independent review. That verdict
applies only to the reviewed code and offline scope, not robot deployment.

## Future Authorized Run

Do not execute this example until robot deployment, ROS subscription, local
log writes, and the selected model/provider API use have been authorized.
It is a future entry point, not an offline dry-run command.

Prerequisites:

- A dedicated RPent process in an ordinary operator TTY. Only `--planner api`
  and `--memory-profile local` are supported. Dashboard, interactive,
  exploration, task-card, and shell-based planners are rejected.
- A compatible Python/ROS 2 Humble environment that can import `rclpy`,
  `sensor_msgs`, and `xarm_msgs`, including native Python ABI/type support.
  A workspace directory alone is insufficient. RPent does not run `source`,
  install packages, or change environment variables for you.
- Explicit `ROS_DOMAIN_ID` matching `--ros-domain-id`, default 3. No foreign
  initialized default ROS context; the adapter owns only the one it creates.
- Access to both topics, confirmed by actual incoming messages. Earlier graph
  inspection established names/types only. The observed `/home/rpp/rpp_ws`
  workspace was not readable by the tested account. Package imports and
  subscription behavior remain unverified. Do not bypass permissions.
- An approved nonempty `provider:model` identifier in `RPENT_MODEL`, with
  provider credentials/configuration supplied securely. Never put secrets in
  source files, command history, logs, or memory documents.

```bash
ROS_DOMAIN_ID=3 rpent --robot lynsense --planner api \
  --model "${RPENT_MODEL:?Set an approved provider:model identifier first}" \
  --memory-profile local --ros-domain-id 3 --max-turns 3
```

This contacts the selected model and writes RPent logs/transcripts under a
dated local `logs/lynsense_*` directory, or `--output-dir`. It does not sync
Hugging Face memory. `--memory-dir` selects a local directory; the model has
no tools to read or write that directory.

Options: `--joint-state-topic`, `--robot-state-topic`, `--state-timeout`
(default 5 seconds for the first valid pair), and `--state-max-age` (default
2 seconds per stream). Topics must be distinct absolute ROS names; durations
must be positive finite numbers.

Normal completion, planner errors, and an operator interrupt use RPent's
existing cleanup path. Cleanup wakes the executor, waits for the callback
thread, shuts down the executor, destroys the node, then shuts down its owned
default context. Thread/executor waits share a 2-second budget. Cleanup
failure raises instead of reporting `closed`; retained resources allow
`close()` to be retried. This stops subscriptions, not robot motion, and is
not an emergency stop.

An executor fault automatically runs cleanup after `spin_once` unwinds; an
external closer never holds the lifecycle lock while joining that worker.
Its snapshot remains `failed` with the original cause even after resources
are released. Shutdown uses the explicitly owned context so failures can be
retried; the default context slot is released only after successful shutdown
and only if it still refers to the owned context.

Real-robot acceptance still requires separately authorized message-package
validation, first messages on both topics, independent stale-state checks,
and shutdown verification. Arm waving requires a separate motion design
and supervised validation; this backend cannot wave the arm.
