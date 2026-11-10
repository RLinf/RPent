# BEHAVIOR

BEHAVIOR connects the standard RPent Planner and Toolkit to the official
BEHAVIOR-1K evaluator. The Planner selects tools; the evaluator executes native
robot actions, advances the simulation, and owns reset, termination and scoring.

[中文说明](README.zh-CN.md)

## Supported runtime

- One R1Pro environment and one official evaluator connection per Env service.
- The 100 challenge task IDs, `0..99`; supply the matching full main-task
  instruction. Recognition of a task ID is not evidence of task success.
- Three RGB views in `head`, `left_wrist`, `right_wrist` order, with public robot
  state and optional depth/calibration for geometric tools.
- Native actions have 23 channels. The RLinf Pi0.5 preset uses `abs_joint`, BF16,
  five denoising steps and a 32-action chunk. Main-task instructions stay fixed.
- Services are started separately and borrowed by the RPent CLI. Closing the
  Planner closes its client connections; it does not stop borrowed services.
  Dashboard startup and multiple evaluator sessions are not supported here.

## Install and configure

Use the official [BEHAVIOR-1K installation](https://github.com/StanfordVL/BEHAVIOR-1K)
for the simulator, challenge assets and evaluator. Keep its environment separate
from the Planner and model environments.

In a source checkout, install the lightweight Planner/Env dependencies:

```bash
python -m pip install -e '.[behavior]'
```

The VLA service additionally needs a BEHAVIOR-capable RLinf installation and
its model dependencies, checkpoint, normalization statistics and tokenizer
assets. Install `.[behavior-vla]` in that model environment. It must provide
`rlinf.models.embodiment.openpi_pytorch.get_model`, the `pi05_behavior`
configuration and `predict_action_batch(..., mode="eval")`. RPent does not
download weights or install a particular RLinf fork implicitly. The policy
adapter delegates image transforms and action normalization to that RLinf build.
If its tokenizer assets are stored separately, set `OPENPI_DATA_HOME` as required
by RLinf before starting the model service.

The complete Agent also requires the shared RPent SAM3 service (`.[sam3]`, its
checkpoint in a separate environment) and CuRobo v0.7.8 with a compatible
PyTorch/Warp/CUDA environment for `move_to`. These dependencies are optional for
the VLA-only tool configuration. No simulator, model or GPU library is imported
by robot discovery or `--help`.

Choose a fresh run directory outside the source checkout. Set the following
variables in each terminal, replacing the installation and weight paths:

```bash
export RPENT_REPO=/path/to/RPent
export RPENT_RUN=/path/to/behavior-runs/run-name
export VLA_PYTHON=/path/to/vla-env/bin/python
export ENV_PYTHON=/path/to/env-service-env/bin/python
export SIM_PYTHON=/path/to/simulator-env/bin/python
export PLANNER_PYTHON=/path/to/planner-env/bin/python
export VLA_CHECKPOINT=/path/to/checkpoint
export PLANNER_MODEL=provider:vision-model-name
export TASK_ID=40
export TASK_NAME=make_microwave_popcorn
export TASK_LANGUAGE='In the kitchen, take the popcorn bag from the countertop, put it into the microwave, and heat it until the popcorn is cooked so the cooked popcorn ends up inside the bag.'
cd "$RPENT_REPO"
```

The task above is an example. Replace all three task variables together for a
different task; [`tasks.py`](../tasks.py) lists the official ID/name correspondence. Do not put
hostnames, credentials, model weights or run outputs into the checked-in YAML.

## Run one episode

Start each long-running command in a separate terminal. The commands below use
the VLA-only tool configuration first, so SAM3 and CuRobo are not needed.

**1. Start the VLA service.** It predicts actions and never advances simulation.

```bash
"$VLA_PYTHON" -m robots.behavior.vla_server \
  --checkpoint "$VLA_CHECKPOINT" --task-prompt "$TASK_LANGUAGE" \
  --host 127.0.0.1 --port 28011 --transport socket \
  --output-dir "$RPENT_RUN/vla"
```

**2. Start the Env bridge after the model is ready.** Its internal RPC and
official policy WebSocket use different ports.

```bash
"$ENV_PYTHON" -m robots.behavior.env_server \
  --task-id "$TASK_ID" --task-language "$TASK_LANGUAGE" \
  --vla-endpoint socket://127.0.0.1:28011 \
  --host 127.0.0.1 --port 28010 --transport socket \
  --policy-host 127.0.0.1 --policy-port 28012
```

**3. Start the official evaluator.** Public-test index `0` is instance `301`.
The evaluator owns the simulation loop. Leave its default step budget intact
for task evaluation; the bridge requires one observation/action exchange per
step, not client-side action-chunk replay.

```bash
"$SIM_PYTHON" -m omnigibson.eval.eval \
  --task-name "$TASK_NAME" --mode public_test --instance-indices 0 \
  --num-rollouts 1 --num-envs 1 --host 127.0.0.1 --port 28012 \
  --write-video --video-fps 120 --output-dir "$RPENT_RUN/official"
```

**4. Run the standard RPent CLI after the first observation is available.**
Supply a configured image-capable Planner model and credentials through the
normal RPent provider configuration; a provider URL does not belong in the robot
template. `--model` below is the VLM Planner, not the VLA checkpoint.

```bash
"$PLANNER_PYTHON" -m rpent.cli.main --robot behavior --planner api \
  --model "$PLANNER_MODEL" \
  --behavior-config robots/behavior/configs/vla_only.yaml \
  --task-id "$TASK_ID" --task-language "$TASK_LANGUAGE" \
  --env-endpoint socket://127.0.0.1:28010 \
  --vla-endpoint socket://127.0.0.1:28011 \
  --output-dir "$RPENT_RUN/agent"
```

`vla_only.yaml` exposes observation, VLA and finish to the Planner. A VLM still
selects these tools, so this is not a pure VLA benchmark baseline.

For the complete `agent.yaml`, start the shared SAM3 server with its checkpoint,
start the Env bridge with `--motion-backend curobo`, and add
`--sam3-endpoint socket://127.0.0.1:28013` to the Planner command. See
`python -m rpent.robots.components.sam3_server --help` for the service options.
Reuse the same task text and model checkpoint. Stop the separately owned
services explicitly after the episode; restarting an episode requires a fresh
Env service and output directory.

## Tools and contracts

`agent.yaml` exposes twelve robot tools. RPent also supplies its normal
file/memory tools; the robot allowlist does not replace those tools.

| Tool | Purpose | Advances the environment |
| --- | --- | --- |
| `view_env_state` | Read saved RGB views, measured state and execution feedback | No |
| `read_robot_state` | Read detailed measured joints and velocities of a saved record | No |
| `segment` | SAM3 text or original-image `[row, column]` segmentation | No |
| `pixel_to_robot_base` | Project a saved-frame pixel/depth point into that frame's robot base | No |
| `vla_act` | Infer and execute a bounded main-task action chunk | Yes |
| `set_gripper` | Command either gripper, `val=0` closed / `val=1` open | Yes |
| `move_to` | Move one arm to a nearby pose solved by CuRobo IK | Yes |
| `move_eef_delta` | Move one arm by a small relative xyz offset through the `move_to` path | Yes |
| `rotate_wrist` | Rotate one EEF about robot-base Z through the `move_to` path | Yes |
| `rotate_wrist_joint` | Turn one wrist joint (roll, pitch or yaw) by a relative angle, without IK | Yes |
| `move_base` | Execute a bounded local base-velocity segment and zero command | Yes |
| `finish` | End the Planner session and save its assessment | No |

`release` opens the selected gripper; it is available by name in
`tool_allowlist` and is not part of `agent.yaml`.

Tool parameters and limits are defined by the typed `@tool` functions. All
tools return the existing RPent `ToolResult(data, images)` contract. The Toolkit
records new observations after action tools and returns three views with
`state` and `log.result`. A saved observation `step` is not an environment action
count. Issued, confirmed and unconfirmed actions remain separate; unknown
execution is not treated as zero and a timed-out chunk is not automatically
replayed.

Geometric points are valid in the saved observation's frame. Re-localize after
the robot moves. `move_to`, `move_eef_delta` and `rotate_wrist` require the Env
service's CuRobo motion backend. They hold the trunk and the other arm, solve IK
for the requested pose and interpolate the arm joints; no path-level collision
check is made against the scene, and they do not provide general grasp
planning. When the requested orientation cannot be kept, the planner retries for
position only and reports `orientation_relaxed` with the achieved orientation
error. When IK fails, nothing moves and the result reports the nearest reachable
position. `move_base` has no reliable odometry or arrival guarantee. Gripper
command completion does not prove a grasp. Tool success and Planner `finish` do
not substitute for official scoring.

`rotate_wrist(arm=..., delta_yaw_deg=..., settle_steps=0)` rotates the EEF about
robot-base Z by a relative angle between -180 and 180 degrees; reachability is
decided by IK. Its target retains the current EEF xyz, but the path and
fingertips can sweep through surrounding space. It requires the same stationary
planning start as `move_to`; it does not turn held objects or perform contact
operations.

`rotate_wrist_joint(arm=..., joint=..., delta_deg=...)` turns a single wrist
joint by a relative angle between -90 and 90 degrees with a linear joint ramp,
then holds the final target. It does not use IK, so it always executes; the
hand position moves along a short arc about the wrist. The other joints, the
trunk, the base and both gripper commands are held. The result reports the
measured change and whether the request was clipped to the joint limit.

Arm tools change left-arm channels 7–13 or right-arm channels 15–21; base
channels 0–2 are zero and the trunk, other arm and both gripper commands hold.
Completion uses confirmed actions and measured pose error, not zero terminal
velocity.

## Memory

Memory is prior experience read through RPent's shared `read_text_file` and
`list_dir`, not another robot action tool. Select a local corpus with
`--memory-dir`; without it the default `memory/behavior/` directory is used.
The corpus follows the shared `MEMORY.md`, `global/` and `task-family/` layout.
The Planner reads the index after the initial observation, then up to three
relevant entries before acting. Targets still come from the current
observations; prior notes do not establish current coordinates, a successful
grasp or an official task outcome. BEHAVIOR evaluation opens the corpus
read-only and does not synchronize a remote corpus.

## Code map and development

| Area | Files under `robots/behavior/` | Responsibility |
| --- | --- | --- |
| Usage/task catalog | `guides/README.md`, `guides/README.zh-CN.md`, `tasks.py` | Installation, development navigation and official task identifiers |
| Registration/configuration | `__init__.py`, `robot_spec.py`, `configs/` | Existing RobotSpec/RunConfig, task identity and borrowed services |
| Planner guidance | `prompts/evaluate.py`, `prompt_bundle.py` | General strategy and tool-dependent prompt assembly |
| Robot actions | `primitives.py` | VLA, gripper, arm and base tools; action accounting |
| Agentic tools | `tools.py` | Observation queries, SAM3 segmentation with saved mask/overlay artifacts, pixel projection and `finish`; public observation decoding |
| Tool registration | `toolkit.py` | Register both tool groups, bind runtime resources and capture post-action observations |
| Motion backend | `motion.py`, `assets/r1pro/` | Environment-side CuRobo IK, robot mapping and the bundled robot model |
| VLA service | `vla_client.py`, `vla_server.py`, `vla_policy.py` | Model-only RPC and RLinf input/output adaptation |
| Environment service | `env_client.py`, `env_server.py` | Env contract, numeric request compatibility and service startup |
| Official evaluator adapter | `eval/evaluator.py` | Official connection, policy lifecycle and action/observation handshake |

Define tools that command the robot on `BehaviorPrimitives` in `primitives.py`.
Define observation, perception, geometry and session tools in `tools.py`. Register
both groups in `toolkit.py`, using RPent's existing `iter_tools` and resource
binding. Small helpers stay beside the capability that owns them. The Env service
uses `motion.py` for CuRobo IK; it does not define Planner tools.

`move_to` in `primitives.py` requests a joint path through the Env client and
executes that path through the common action interface. The Env service owns
`R1ProMotionPlanner` in `motion.py`: it loads the robot model, solves IK for
the requested pose with CuRobo and interpolates the seven arm joints under
their velocity limits. No path-level collision check is made against the
scene; the simulator's physics is the contact authority. The returned joint
waypoints are inputs to the action tool. `motion.py` is not registered in the
Toolkit.

`motion.py` exists because of what the official evaluation interface exposes. The
official evaluator owns the simulator and exchanges only observations, actions
and scores over the policy connection; it does not offer the Agent side a
callable arm planner or inverse-kinematics service. A tool that moves an arm to a
requested pose therefore has to solve IK inside the Env service: it needs the
robot model, the joint limits and the robot's self-collision spheres to turn a
target pose into joint waypoints. `R1ProMotionModel` reads those from the
bundled URDF and CuRobo robot configuration, and `R1ProMotionPlanner` drives
CuRobo with them. Keeping the controller-side planning in `motion.py` is what
lets `primitives.py` stay a collection of tools that only produce the
23-dimensional action.

`eval/evaluator.py` contains the complete official evaluator adapter. Its three
classes follow the message path from the official connection to RPent tools:

| Class | Responsibility |
| --- | --- |
| `WebsocketPolicyServer` | Encode/decode the official messages and manage the WebSocket connection |
| `OfficialEvaluatorPolicy` | Implement `reset()` and `act(observation)` for one official evaluation session |
| `EvaluatorStepBridge` | Exchange observations and queued actions between the evaluator and tools; confirm each issued action from the following observation |

The official evaluator advances the simulator and computes scores. The Env
service shares one bridge between its internal RPC interface and the official
policy interface. `env_server.py` constructs the bridge and policy and calls
`create_official_policy_server()`; the adapter does not create a second
simulation loop. Transport dependencies are loaded only when serving messages.

Change common decision guidance in `prompts/evaluate.py`; use `prompt_bundle.py`
for assembly. Put parameter units, limits and result semantics in tool
docstrings, and actual capability changes in the owning implementation. Keep
task goals dynamic.

Offline regression tests live in `tests/unit_tests/robots/behavior/`. Explicit
real-service checks live in `tests/e2e_tests/behavior/`; see
[the test guide](../../../tests/README.md). Logs, videos and checkpoints belong
outside this package.

## Results and troubleshooting

- `agent/`: RPent transcript, saved observations and tool results.
- `vla/`: model settings, inference timings and actual predicted chunks.
- `official/`: evaluator JSON scores and videos, the authority for task success.

Video encoding defaults to 120 FPS in the commands above; control remains
independent of video playback. Each run requires fresh output directories.
For a connection error, check the service addresses and logs. Do not pass a
`ws://` policy URL as `--env-endpoint`. On a timeout or connection loss,
preserve the logs and inspect execution feedback before starting a new episode.
