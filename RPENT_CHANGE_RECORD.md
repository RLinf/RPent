# RPent change record

This file records RPent modifications made during live dual-Franka debugging so
they can be reviewed later. New RPent code/config changes should append an entry
here with the intent, touched files, validation, and any runtime caveats.

## 2026-09-10 — dual-Franka live deployment fixes and SAM3 integration

Status: local working tree changes, not yet committed/pushed.

Context:

- Target setup: real dual-Franka clean-desk task under `robots/dual_franka`.
- User requested that runner decisions should not be overridden by helper code;
  perception tools may return annotated images for the runner/agent to inspect.
- Debug run analyzed: `logs/20260910-140639-clean-desk-sam3-live`.

Main changes currently present in the RPent working tree:

1. Hardware/runtime configuration alignment

   - Updated dual-Franka config and runtime plumbing for the current physical
     setup, including camera identities, D455 perception camera support, safety
     bounds, reset joint posture, and calibration loading.
   - Main files:
     - `robots/dual_franka/config/example.yaml`
     - `robots/dual_franka/runtime_config.py`
     - `robots/dual_franka/robot_spec.py`
     - `robots/dual_franka/env_client.py`

2. Agent-visible D455 perception path

   - Added D455 RGB/depth capture as a fixed external perception view.
   - D455 is exposed to the agent/runner for localization and verification; it
     is not part of the 3-image VLA policy input.
   - Main files:
     - `robots/dual_franka/env_server.py`
     - `robots/dual_franka/tools.py`
     - `robots/dual_franka/toolkit.py`
     - `robots/dual_franka/perception.py`

3. SAM3 D455 segmentation support

   - Added `segment` so the agent can call SAM3 with either a text prompt
     or a positive `[row, col]` point on the D455 image.
   - The tool returns a mask overlay image to the agent for visual checking.
   - Localization uses D455 depth and calibration to return a median mask point
     in `right_base`.
   - Main files:
     - `robots/dual_franka/perception.py`
     - `robots/dual_franka/tools.py`
     - `robots/dual_franka/toolkit.py`
     - `robots/dual_franka/robot_spec.py`

4. Prompt/task alignment for clean-desk VLA execution

   - Registered named clean-desk VLA tools:
     `vla_right_grasp`, `vla_right_grasp`, `vla_right_grasp`,
     `vla_right_grasp`, `vla_right_grasp`, `vla_handoff`,
     `vla_left_place`.
   - Prompts/tasks emphasize category order, D455 verification, cardboard-box
     versus metal-basket disambiguation, and VLA segment boundary checks.
   - Main files:
     - `robots/dual_franka/tools.py`
     - `robots/dual_franka/tasks.py`
     - `robots/dual_franka/prompts/system.py`
     - `robots/dual_franka/prompts/user.py`
     - `robots/dual_franka/prompt_bundle.py`

5. Joint-health monitoring and recovery

   - Added joint-health reporting in robot state.
   - Added `recover_joint_posture` / `reset_joint_posture` style recovery that
     preserves gripper open/closed state, re-commanding closed grippers around
     joint reset so held objects are not released by recovery.
   - Main files:
     - `robots/dual_franka/env_server.py`
     - `robots/dual_franka/tools.py`
     - `robots/franka/env_server.py`

6. 2026-09-10 latest fixes requested as items 1/4/5

   - Disabled RLinf auto-reset for RPent-controlled live tasks:
     - `robots/dual_franka/runtime_config.py`
       - `env.eval.auto_reset = False`
       - `env.eval.max_episode_steps = None`
     - `robots/dual_franka/env_server.py`
       - all RPent internal `self.env.step(...)` calls now pass
         `auto_reset=False`
   - Reason: the latest log showed `move_delta` crossing the RLinf episode
     horizon; auto-reset opened both grippers and moved arms home, which caused
     an apparent `move_delta` gripper release.
   - Improved SAM3 prompt guidance:
     - Prefer short phrases such as
       `white interior of the black cardboard box` or `cardboard box`.
     - Avoid blindly lowering `min_score` after a very low-score text prompt.
   - Made D455 back-projection diagnostics non-overwriting:
     - New artifact names use per-step call indices, e.g.
       `d455_back_project_00_annotated.png` and
       `d455_back_project_00.json`.

Validation performed:

- Targeted unit tests:
  - Command:
    `.venv/bin/python -m pytest tests/unit_tests/robots/dual_franka/test_dual_franka_tools.py tests/unit_tests/robots/dual_franka/test_dual_franka_env_server.py tests/unit_tests/robots/dual_franka/test_dual_franka_extension.py tests/unit_tests/robots/dual_franka/test_rlinf_contract.py`
  - Result: `16 passed, 1 warning`.
- SAM3 prompt sweep on the logged step-2 D455 image:
  - `white interior floor of black cardboard box`: score `0.0085`
  - `white interior of the black cardboard box`: score `0.629`
  - `cardboard box`: score `0.605`
  - Conclusion: the observed failure was prompt phrasing sensitivity, not a
    simple case of the default `min_score=0.2` being too high.

Runtime caveats:

- The running env server must be restarted before the auto-reset fix takes
  effect on the robot.
- `.codex-rpent-live/` is a local Codex state/log isolation directory for RPent
  live runs. It is untracked and should not be committed; add a gitignore rule
  later if desired.

## 2026-09-10 — agent-facing right_base coordinate isolation

Status: local working tree changes, not yet committed/pushed.

Context:

- User clarified the core goal: the agent should only reason in one unified
  coordinate frame and should not need to maintain separate left/right robot
  base frames.
- Log confirmation from
  `logs/20260910-140639-clean-desk-sam3-live` showed that `view_env_state` and
  `move_delta` results exposed `left_arm.tcp_pose` in the left robot base while
  D455 projection points were in `right_base`.
- The same log also showed `back_project` returning `point_xyz` but
  missing both `delta_left_tcp_to_point_xyz` and
  `delta_right_tcp_to_point_xyz`.

Changes:

1. Agent-visible robot state now uses `right_base`

   - File: `robots/dual_franka/env_server.py`
   - `get_robot_state()` now returns `coordinate_frame: right_base`.
   - `left_arm.tcp_pose` is transformed from `left_base` into `right_base`
     before it is exposed to the agent.
   - Raw controller poses are retained for debugging as `raw_tcp_pose` with
     `raw_tcp_pose_frame`.

2. Rule-based motion contract is now explicit world-frame

   - File: `robots/dual_franka/env_server.py`
   - `move_delta(left, ...)` and `rotate_delta(left, ...)` compute their command
     and error in `right_base`, then transform the target back to the left
     controller's local base before dispatch.
   - Returned `start_tcp_pose`, `target_tcp_pose`, and `final_tcp_pose` are in
     `right_base`; raw local start/final poses are kept separately.
   - `recover_joint_posture(return_to_start=True)` now computes return deltas in
     `right_base`, so the recovery path follows the same public coordinate
     contract.
   - File: `robots/franka/env_server.py`
     - Added `--calibration-path` to the env server CLI and calls
       `set_calibration_path()` before runtime/worker startup, because
       dual-Franka env server now needs calibration for agent-facing pose
       transforms.
   - File: `robots/dual_franka/robot_spec.py`
     - The auto-spawned dual-Franka env server now receives the same
       `--calibration-path` selected by the runner/toolkit.

3. Projection/SAM3 localization now always tries to return both arm deltas

   - File: `robots/dual_franka/perception.py`
   - Added TCP pose transformation helper for full xyz+quat poses.
   - Fixed `_tcp_xyz()` to accept `np.ndarray` as well as list/tuple, matching
     the live in-memory state type before JSON serialization.
   - `back_project`, `back_project`, and `segment`
     now include, when state is available:
     - `left_tcp_xyz`
     - `right_tcp_xyz`
     - `delta_left_tcp_to_point_xyz`
     - `delta_right_tcp_to_point_xyz`
     - `tcp_delta_coordinate_frame: right_base`
   - Both deltas are computed in the shared `right_base` world frame.

4. Tests

   - File: `tests/unit_tests/robots/dual_franka/test_dual_franka_tools.py`
   - Added coverage for D455 back-projection returning annotated images and both
     left/right TCP deltas.
   - Regression coverage uses an `np.ndarray` left TCP pose to reproduce the
     live-path type that previously caused delta fields to be omitted.

Validation performed:

- Syntax check:
  `.venv/bin/python -m py_compile robots/dual_franka/perception.py robots/dual_franka/env_server.py robots/franka/env_server.py robots/dual_franka/robot_spec.py tests/unit_tests/robots/dual_franka/test_dual_franka_tools.py`
- Targeted unit tests:
  `.venv/bin/python -m pytest tests/unit_tests/robots/dual_franka/test_dual_franka_tools.py tests/unit_tests/robots/dual_franka/test_dual_franka_env_server.py tests/unit_tests/robots/dual_franka/test_dual_franka_extension.py tests/unit_tests/robots/dual_franka/test_rlinf_contract.py`
- Result: `16 passed, 1 warning`.
- Offline sanity check using
  `logs/20260910-140639-clean-desk-sam3-live` step 2 and the physicalagent
  calibration bundle:
  - projected point `[0.79174, 1.06515, 0.80258]` now produces
    `left_tcp_xyz [0.57548, 0.36391, 0.57657]` and
    `delta_left_tcp_to_point_xyz [0.21626, 0.70124, 0.22601]`;
  - also produces `right_tcp_xyz [0.59652, 0.19663, 0.59939]` and
    `delta_right_tcp_to_point_xyz [0.19522, 0.86852, 0.20319]`.

Runtime caveat:

- The currently running env server must be restarted before this coordinate
  isolation takes effect in live robot runs.

Follow-up fix:

- File: `robots/dual_franka/runtime_config.py`
  - Added `controller["calibration_path"]` so the explicit runner/env-server
    calibration path is serialized into the Ray worker configuration.
- File: `robots/dual_franka/env_server.py`
  - `DualFrankaEnvWorker._calibration_bundle()` now loads from
    `self.controller["calibration_path"]` instead of falling back to the
    process-local default path.
- File: `tests/unit_tests/robots/dual_franka/test_rlinf_contract.py`
  - Added regression coverage that `load_runtime_config()` carries the
    configured calibration path into the controller config.

## 2026-09-11 — clean-desk prompt and D455-primary observation alignment

Status: local working tree changes, not yet committed/pushed.

Context:

- User accepted that RPent structured tool names may differ from the old
  PhysicalAgent runner, but requested that the clean-desk task description
  prompt stay aligned with the old deployment and that the agent primarily use
  D455, as before.

Changes:

1. Restored old clean-desk task description fields

   - File: `robots/dual_franka/tasks.py`
   - Task 1 now uses the old PhysicalAgent text verbatim for:
     - `instruction`
     - `setup`
     - `success_criteria`
   - File: `robots/franka/tasks.py`
     - Added optional `setup` field to `FrankaTask`.
   - File: `robots/dual_franka/robot_spec.py`
     - Passes `setup` into prompt rendering.
   - File: `robots/dual_franka/prompts/user.py`
     - Renders `initial_setup` in the task prompt.

2. Restored D455-primary planner observation behavior

   - File: `robots/dual_franka/tools.py`
   - `view_env_state` now returns D455 as the only inline image block.
   - `left_wrist`, `base`, and `right_wrist` image paths remain in the returned
     JSON as artifact views for targeted `read_image` inspection.
   - `describe_dual_franka_setup` now reports D455 as the inline
     agent-visible image and the other three cameras as auxiliary artifacts.
   - File: `tests/unit_tests/robots/dual_franka/test_dual_franka_tools.py`
     - Updated regression expectations to assert D455-only inline image blocks
       while preserving four camera artifact paths.

3. Prompt wording updated for D455 priority

   - File: `robots/dual_franka/prompts/system.py`
   - File: `robots/dual_franka/prompts/user.py`
   - The prompt now states that routine snapshots inline only D455 and that
     other cameras should be inspected only when specifically needed.

Validation performed:

- Syntax check:
  `.venv/bin/python -m py_compile robots/franka/tasks.py robots/dual_franka/tasks.py robots/dual_franka/prompts/user.py robots/dual_franka/prompts/system.py robots/dual_franka/robot_spec.py robots/dual_franka/tools.py`
- Prompt parity check:
  task 1 `instruction`, `setup`, and `success_criteria` exactly match
  `/home/raojiaji/nieyi/physicalagent/physical_agent/envs/dual_franka/tasks.py`.
- Targeted unit tests:
  `.venv/bin/python -m pytest tests/unit_tests/robots/dual_franka/test_dual_franka_tools.py tests/unit_tests/robots/dual_franka/test_dual_franka_extension.py`
- Result: `13 passed, 1 warning`.

Runtime caveat:

- Existing long-running runner processes will not pick up these prompt/tool
  changes. Start a new runner process for the next live attempt. The env/VLA/SAM3
  services do not need to restart for this prompt-only alignment.

## 2026-09-11 — registered dirty/clean sorting task for dual-Franka

Status: local working tree change, not yet committed/pushed.

Context:

- User pointed out that the intended new RPent task should be the dirty/clean
  sorting variant, but the current local `robots/dual_franka/tasks.py` only
  registered task IDs 0 and 1.
- The dirty/clean task definition was recovered from the old PhysicalAgent
  implementation at
  `/home/raojiaji/nieyi/physicalagent/physical_agent/envs/dual_franka/tasks.py`.

Changes:

- File: `robots/dual_franka/tasks.py`
  - Added `task_id=3` named
    `clean_desk_dirty_clean_sorting_agent_vla`.
  - The task keeps the old category/color order:
    bowls -> plates -> cup -> chopsticks -> spoon, green before blue.
  - Added explicit dirty/clean classification semantics:
    - dirty bowl/plate: visible egg tart or egg-tart foil cup/tray inside or on
      the concave eating surface;
    - clean bowl/plate: no egg-tart/foil marker;
    - dirty bowls/plates go to the metal wire basket/frame;
    - clean bowls/plates and cup go to the black-outside, white-inside
      cardboard box;
    - dirty/clean does not affect grasp order, only placement destination.
  - Adapted the old task constraints to the current RPent tool names and live
    infra: D455-primary observation, optional SAM3 `segment`, unified
    `right_base` coordinates, named VLA skill tools, and gripper-preserving
    `recover_joint_posture`.
- File: `robots/dual_franka/prompts/system.py`
  - Added a system-level container semantic rule for dirty/clean sorting tasks,
    so the metal basket is treated as valid only for dirty bowls/plates rather
    than being globally forbidden by the default clean-desk guidance.

Validation performed:

- Syntax check:
  `.venv/bin/python -m py_compile robots/dual_franka/tasks.py robots/dual_franka/prompts/system.py`
- Task registry check:
  `get_dual_franka_task(3)` returns
  `clean_desk_dirty_clean_sorting_agent_vla`, with dirty/egg-tart/metal-wire
  rules present.
- CLI check:
  `.venv/bin/python -m rpent.cli.main --robot dual_franka --help` now lists
  `--task-id {0,1,3}`.

Runtime caveat:

- Existing runner processes will not pick up the new task registry. Start a new
  runner with `--task-id 3` for dirty/clean sorting. The env/VLA/SAM3 services
  do not need to restart for this prompt/task-registry-only change unless their
  process embeds a stale copy of task prompts.

## 2026-09-11 — live-debug organization and manual single-skill test entry

Status: local working tree change, not yet committed/pushed.

Context:

- User requested the recent reproducibility changes be organized for later
  review.
- Old PhysicalAgent had manual single-skill script entry points such as
  `scripts/run_dual_franka_vla_10_chunks.sh` and `cli.dual_franka_manual`.
- User also asked whether the old operator feedback/evaluation control scripts
  could become a mid-task human intervention surface.

Changes:

- File: `DUAL_FRANKA_LIVE_DEBUG_REVIEW.rst`
  - Added a compact live-debug guide covering:
    - current live modifications;
    - single-skill/manual testing commands;
    - native RPent human intervention paths;
    - dirty/clean task 3 runner command.
- File: `scripts/dual_franka_manual_call.py`
  - Added a planner-free manual primitive/tool caller for live dual-Franka
    debugging.
  - Supports direct `--primitive/--params` calls.
  - Exposes `--list-primitives`, `--schema <primitive>`, and
    `--example <primitive>` from the active `TOOLS_SPEC`, so manual testing
    follows the same registered tool schema as the planner instead of relying
    on separate JSON preset files.
  - Supports read-only tools such as `view_env_state`,
    `back_project`, and `segment`, as well as mutating robot
    calls such as `reset`, `move_delta`, gripper open/close,
    `recover_joint_posture`, and named VLA skills.
  - Reuses current RPent `DualFrankaPrimitives`, `Pi05VLAClient`, `Sam3Client`,
    and `dump_state`, so manual skill execution follows the same named-VLA
    semantic stop rules and writes `states.json` plus image/depth artifacts.
- Removed the separate `resources/dual_franka/manual_calls/*.json` preset layer.
  It was redundant with `TOOLS_SPEC.input_schema` and made the manual testing
  surface look like it had a second skill registry.

Human intervention note:

- The old PhysicalAgent operator console used a file-queue command bus with
  commands such as `start`, `success`, `failure`, `done`, and `quit`.
- RPent already has native interactive paths:
  - terminal `--interactive` for injecting messages between planner turns;
  - `--dashboard` for queued messages, message withdraw, interrupt at tool
    boundaries, and task replacement.
- Recommendation: use RPent dashboard as the primary mid-task human intervention
  interface. If the old operator command vocabulary is still desired later, add
  a thin adapter over the dashboard HTTP APIs instead of porting the full old
  file-queue session system.

Validation performed:

- Syntax check:
  `.venv/bin/python -m py_compile scripts/dual_franka_manual_call.py`
- Manual CLI registry check:
  `.venv/bin/python scripts/dual_franka_manual_call.py --list-primitives`
- Read-only live smoke test:
  `.venv/bin/python scripts/dual_franka_manual_call.py --env-endpoint http://127.0.0.1:6001 --primitive view_env_state --output-dir logs/20260911-142520-manual-view-env-state-smoke --timeout-s 120`
  wrote `result.json`, `states.json`, D455 RGB/depth, left/right wrist images,
  base image, and camera metadata without moving the robot.

## 2026-09-11 — RLinf FrankyController live-control parameters aligned to old PhysicalAgent deployment

Status: local test changes applied to both local and remote RLinf working trees,
not yet committed/pushed.

Context:

- The dual-Franka RPent workspace/reset parameters were already aligned with
  the old PhysicalAgent deployment, but the newer RLinf FrankyController still
  used softer/larger-step Cartesian defaults.
- Old reference used for comparison:
  `/home/raojiaji/nieyi/rlinf/rlinf/envs/realworld/franka/franky_controller.py`
  on branch `dual-franka` (`375a6e19`).
- Current test repo:
  `/home/raojiaji/nieyi/franka_port_test/RLinf` on branch `port/franka`
  (`76c4ede1`).

Changes:

- File: `../RLinf/rlinf/envs/realworld/franka/franky_controller.py`
- Defaults changed to old-deployment-compatible values:
  - `RLINF_CART_K_T`: `500.0` -> `1000.0`
  - `RLINF_CART_K_R`: `40.0` -> `50.0`
  - `RLINF_CART_K_NS`: `5.0` -> `0.0`
  - `RLINF_CART_MAX_STEP_M`: `0.10` -> `0.03`
  - `RLINF_CART_MAX_STEP_RAD`: `0.30` -> `0.15`
- The same file was copied to the remote `master` host under the matching
  `/home/raojiaji/nieyi/franka_port_test/RLinf` path so Ray workers on both
  nodes load the same controller defaults.

Validation/runtime:

- Local and remote `py_compile` passed for `franky_controller.py`.
- Rebuilt the test Ray cluster on port `6389` with:
  - independent temp dir `/tmp/ray_rpent_6389`
  - Ray client port `11001` to avoid the existing old Ray on `10001`
  - `RLINF_NODE_RANK=0` on `192.168.121.167`
  - `RLINF_NODE_RANK=1` on `192.168.121.168`
- Restarted env server on `127.0.0.1:6001`; logs confirm RLinf sees 2 nodes,
  left controller on rank 0, and right controller on rank 1.

## 2026-09-11 — project-local Codex isolation wrappers for live robot runs

Status: local working tree change, not yet committed/pushed.

Context:

- User wanted robot-control Codex records, model/API configuration, and memory
  to stay isolated from the normal assistant-for-coding Codex environment.
- OpenAI Codex config supports `CODEX_HOME` for local state/config isolation,
  while project `.codex/config.toml` cannot override provider/auth/profile
  settings.  Therefore the live-run isolation is implemented with explicit
  wrapper scripts and a gitignored project-local `CODEX_HOME`.

Changes:

- File: `.gitignore`
  - Added `.codex-rpent-live/` so live Codex auth/history/cache/memory is never
    committed.
  - Replaced the broad `/resources/` ignore with `/resources/local/`. Manual
    single-skill calls are schema-driven, so no `resources/dual_franka`
    preset templates are tracked.
- File: `scripts/rpent_live_env.sh`
  - Added a source-only shell environment helper.
  - Sets project-local `CODEX_HOME`, local RPent memory dir, `RLINF_REPO_PATH`,
    `PYTHONPATH`, live env/VLA/SAM3 endpoints, robot config, calibration path,
    and default planner/model/task values.
- File: `scripts/run_dual_franka_task3_codex.sh`
  - Added reproducible full-task runner for the dirty/clean sorting task.
  - Uses project-local Codex state and local memory by default.
- File: `scripts/run_dual_franka_interactive.sh`
  - Added the same full-task runner with `--interactive` enabled.
- File: `scripts/run_manual_skill.sh`
  - Added a convenience wrapper around `scripts/dual_franka_manual_call.py` so
    single-skill tests use the same isolated environment and endpoints.
- File: `.codex-rpent-live.example/config.toml.example`
  - Added a key-free example Codex config for the project-local live directory.
- File: `DUAL_FRANKA_LIVE_DEBUG_REVIEW.rst`
  - Documented the isolation model and wrapper commands.

Validation performed:

- Shell syntax checks:
  - `bash -n scripts/rpent_live_env.sh scripts/run_dual_franka_task3_codex.sh scripts/run_dual_franka_interactive.sh scripts/run_manual_skill.sh`
- Source check:
  - `source scripts/rpent_live_env.sh` creates `.codex-rpent-live/` and sets
    `CODEX_HOME` to the project-local directory.
- Manual wrapper smoke check:
  - `scripts/run_manual_skill.sh --list-primitives` resolves the live RPent and
    RLinf paths and lists manual primitives successfully.

Runtime caveat:

- The wrappers intentionally do not store API keys.  If a live run needs
  `CODEX_API_KEY` or `CODEX_BASE_URL`, export it in the shell or keep it only in
  the private `.codex-rpent-live/` state directory.

## 2026-09-11 — PR-scope cleanup after live repro review

Status: local working tree change, not yet committed/pushed.

Context:

- User reviewed the live repro modifications and requested separating
  RPent-general mechanisms from lab/table/task-specific parameters.
- User specifically requested:
  - joint-health recovery should remain, but thresholds should live in the
    parameter file with tuning notes;
  - manual single-skill testing should remain, but should not hard-code skill
    names;
  - RLinf FrankyController live-control parameter changes should keep the
    original upstream defaults visible as comments.

Changes:

- File: `robots/dual_franka/config/example.yaml`
  - Moved joint-health thresholds into a top-level `joint_health.thresholds`
    section.
  - Added comments that reset poses, workspace limits, and joint-health
    thresholds are deployment-specific and must be tuned for other scenes.
- File: `robots/dual_franka/runtime_config.py`
  - Removed the hard-coded `JOINT_HEALTH_THRESHOLDS` constant.
  - `load_runtime_config()` now reads per-arm thresholds from YAML and passes
    them to the Ray worker through `runtime.controller`.
- File: `scripts/dual_franka_manual_call.py`
  - Removed the hard-coded named-VLA skill set.
  - Manual primitive discovery now uses the live `TOOLS_SPEC`; any registered
    tool with a matching `DualFrankaPrimitives` method can be called without
    editing the test script.

Validation performed:

- Syntax checks:
  - `.venv/bin/python -m py_compile robots/dual_franka/runtime_config.py scripts/dual_franka_manual_call.py`
- Manual registry check:
  - `.venv/bin/python scripts/dual_franka_manual_call.py --list-primitives`
    lists the currently registered dual-Franka tools, including named VLA
    skills.

Related RLinf cleanup:

- File: `../RLinf/rlinf/envs/realworld/franka/franky_controller.py`
  - Kept the old PhysicalAgent-compatible controller defaults, and added
    inline comments recording the previous upstream defaults:
    `500.0`, `40.0`, `5.0`, `0.10`, and `0.30`.
  - Syntax check passed with the RPent venv Python.

## 2026-09-11 — Generalized dual-Franka views, VLA segment tools, and native Franka health

Status: local working tree change, not yet committed/pushed.

Context:

- User approved the PR-oriented cleanup direction:
  - expose Franka native health signals where possible;
  - replace old camera-specific projection/SAM3 tool names;
  - make planner-visible cameras explicit in config;
  - keep VLA prompt as a tool argument, but override it for the current live
    clean-desk checkpoint;
  - clean up old object-specific VLA tool names;
  - postpone formal documentation until code settles.
- Main-branch reference checked before implementation:
  - fixed simulator cameras use schema enums;
  - dynamic-view style backends validate view names from runtime state/config.
  - The dual-Franka live robot follows the dynamic-view pattern.

Changes:

- File: `robots/dual_franka/config/example.yaml`
  - Added `cameras.agent_observation.inline_cameras` and
    `cameras.agent_observation.auxiliary_cameras`.
  - Added `perception.projection_views` so metric views are registered in YAML
    instead of being hard-coded in tool names.
- File: `robots/dual_franka/runtime_config.py`
  - Loads the agent-facing camera policy and projection-view registry from the
    robot config into `runtime.controller`.
- File: `robots/dual_franka/env_server.py`
  - Exposes `agent_observation` and `projection_views` through env metadata and
    `camera_meta.json`.
  - `joint_health` now prioritizes native Franka/libfranka diagnostics when
    present (`robot_mode`, `has_errors`, current/last motion errors, contact,
    collision), then falls back to the YAML heuristic thresholds.
- File: `robots/dual_franka/perception.py`
  - Replaced `back_project_base_pixel` and `back_project_d455_pixel` with
    generic `back_project(camera=..., row=..., col=...)`.
  - Replaced `segment_d455` with generic
    `segment(camera=..., prompt=... | point=...)`.
  - Projection/SAM3 camera validation now reads `projection_views` from the
    current state's `camera_meta.json`, falling back to `example.yaml` only for
    offline tests or old logs.
  - SAM3 still returns the overlay image block for planner verification.
  - Projection still returns both left and right TCP-to-point deltas in the
    shared `right_base` world frame.
- File: `robots/dual_franka/tools.py`
  - Tool registry now exposes `back_project`, `segment`,
    `vla_right_grasp`, `vla_handoff`, and `vla_left_place`.
  - Removed object-specific VLA tool names from the active registry.
  - VLA segment tools require a `prompt` argument; for the current live
    clean-desk checkpoint, policy inference still uses
    `CLEAN_DESK_VLA_PROMPT`. Tool results record `requested_prompt`,
    `effective_policy_prompt`, and `prompt_overridden`.
  - `describe_dual_franka_setup()` now reports actual camera policy and
    projection views from runtime metadata.
  - `dump_state()` no longer assumes only left_wrist/base/right_wrist plus
    d455. It saves configured observation views, raw camera snapshot views, and
    any `*_images`/`*_depths` view pairs emitted by the env.
  - `view_env_state()` now reports `available_camera_views` and uses configured
    inline cameras for multimodal image blocks; non-inline views remain artifact
    paths.
- File: `robots/dual_franka/toolkit.py`
  - Registers the generic perception tools instead of old d455/base-specific
    names.
- Files: `robots/dual_franka/prompts/system.py`,
  `robots/dual_franka/prompts/user.py`, `robots/dual_franka/prompt_bundle.py`
  - Removed the task/container/D455-specific block from the dual-Franka system
    prompt.
  - Kept robot-level instructions generic: configured inline views, registered
    localization cameras, and three VLA segment tools.
- File: `robots/dual_franka/tasks.py`
  - Updated task prompts to call the new tool names while keeping task-specific
    clean-desk/dirty-clean/D455 wording inside the task definitions.
- File: `scripts/dual_franka_manual_call.py`
  - Manual readonly calls now use `back_project` and `segment`.
  - Binary stripping covers all `_image_*` fields.
- File: `scripts/dual_franka_manual_call.py`
  - Generic perception/VLA manual calls are now expressed with
    `--primitive/--params`; `--schema` and `--example` are generated from the
    registered tool schema.
- File: `robots/dual_franka/robot_spec.py`
  - VLA auto-start detection now treats dual-Franka tasks whose names end with
    `_vla` as VLA-backed instead of checking the stale `vla_grasp` task name.
- Files: `tests/unit_tests/robots/dual_franka/*.py`
  - Updated unit tests for generic perception tools, configurable inline camera
    delivery, projection deltas for both arms, and VLA prompt override logging.

Related RLinf changes:

- File: `../RLinf/rlinf/envs/realworld/franka/franka_robot_state.py`
  - Added optional native Franka/libfranka diagnostic fields:
    `robot_mode`, `has_errors`, `current_errors`, `last_motion_errors`,
    `joint_contact`, `cartesian_contact`, `joint_collision`, and
    `cartesian_collision`.
- File: `../RLinf/rlinf/envs/realworld/franka/franky_controller.py`
  - Populates those native diagnostic fields from `franky.Robot.state` and
    `franky.Robot.has_errors`.

Validation performed:

- RPent lint:
  - `.venv/bin/python -m ruff check robots/dual_franka scripts/dual_franka_manual_call.py tests/unit_tests/robots/dual_franka`
- RPent syntax checks:
  - `.venv/bin/python -m py_compile robots/dual_franka/runtime_config.py robots/dual_franka/env_server.py robots/dual_franka/perception.py robots/dual_franka/tools.py robots/dual_franka/toolkit.py robots/dual_franka/prompts/system.py robots/dual_franka/prompts/user.py robots/dual_franka/prompt_bundle.py scripts/dual_franka_manual_call.py`
- RLinf lint/syntax:
  - `ruff check rlinf/envs/realworld/franka/franka_robot_state.py rlinf/envs/realworld/franka/franky_controller.py`
  - `py_compile rlinf/envs/realworld/franka/franka_robot_state.py rlinf/envs/realworld/franka/franky_controller.py`
- RPent unit tests:
  - `.venv/bin/python -m pytest tests/unit_tests/robots/dual_franka/test_dual_franka_extension.py tests/unit_tests/robots/dual_franka/test_rlinf_contract.py tests/unit_tests/robots/dual_franka/test_dual_franka_tools.py`
  - Result: `17 passed, 1 warning` (`pytest` unknown `timeout` config warning).

Known open questions / deferred cleanup:

- The current live VLA checkpoint still requires a fixed clean-desk training
  instruction.  The interface now accepts `prompt`, but the deployment override
  remains.  A later PR should decide whether this override belongs in robot
  config, task config, or a checkpoint profile.
- `load_calibration_bundle()` still obtains `localization_validity` and
  `base_frames` from the local robot YAML path used by the process.  Runtime
  `projection_views` are now captured in `camera_meta.json`, but validity/base
  frame provenance should be made equally explicit before supporting multiple
  live robot config files in one analysis process.
- The dual-Franka system prompt is now generic, but task 1 and task 3 still
  contain lab-scene details by design.  Formal docs should explain this split
  after the code is finalized.
- Full PR split is still undecided: one PR for generic dual-Franka mechanisms
  plus a separate live lab-profile/task PR may be cleaner than one large PR.

## 2026-09-11 — comments documenting live-deployment specialization

Status: local working tree change, not yet committed/pushed.

Context:

- User asked that the remaining PhysicalAgent/live-lab specializations be
  explicitly annotated before later review, so reviewers can tell which code is
  generic RPent mechanism and which code exists only to reproduce the deployed
  clean-desk setup.

Changes:

- File: `robots/dual_franka/tools.py`
  - Added comments around the D455-facing state description, SAM3 prompt hints,
    and clean-desk fixed VLA prompt override.
- File: `robots/dual_franka/runtime_config.py`
  - Added comments explaining closed-loop timing/gripper settle values, D455
    default observation policy, explicit projection view registry, and joint
    health thresholds as PhysicalAgent-alignment knobs.
- File: `robots/dual_franka/env_server.py`
  - Added comments documenting why RPent temporarily reaches into RLinf
    internals for fresh state, wrapped observations, direct joint reset, and
    gripper-preserving recovery.
- File: `robots/dual_franka/perception.py`
  - Added comments explaining why `d455` remains the default projection/SAM3
    camera even though registered camera names are generic.
- File: `robots/dual_franka/tasks.py`
  - Added a task-registry note explaining that tasks 1 and 3 intentionally
    mirror PhysicalAgent clean-desk prompts/logs and should remain optional demo
    tasks rather than robot-wide behavior.
- File: `scripts/rpent_live_env.sh`
  - Added a comment explaining that the default calibration path points to the
    old PhysicalAgent deployment for log alignment and must be overridden for
    other machines.
- File: `../RLinf/rlinf/envs/realworld/franka/franky_controller.py`
  - Added a comment explaining why the Franky impedance/slew defaults differ
    from upstream RLinf defaults for this reproduction.

Validation performed:

- RPent lint/syntax:
  - `.venv/bin/python -m ruff check robots/dual_franka scripts/dual_franka_manual_call.py tests/unit_tests/robots/dual_franka`
  - `.venv/bin/python -m py_compile robots/dual_franka/runtime_config.py robots/dual_franka/env_server.py robots/dual_franka/perception.py robots/dual_franka/tools.py robots/dual_franka/tasks.py scripts/dual_franka_manual_call.py`
- Shell syntax:
  - `bash -n scripts/rpent_live_env.sh scripts/run_dual_franka_task3_codex.sh scripts/run_dual_franka_interactive.sh scripts/run_manual_skill.sh`
- RLinf lint/syntax:
  - `ruff check rlinf/envs/realworld/franka/franky_controller.py rlinf/envs/realworld/franka/franka_robot_state.py`
  - `py_compile rlinf/envs/realworld/franka/franky_controller.py rlinf/envs/realworld/franka/franka_robot_state.py`

## 2026-09-11 — moved live debug guide into standalone review note

Status: local working tree change, not yet committed/pushed.

Context:

- User asked to move the live debug document to a more appropriate location,
  make its purpose/content clearer, and keep it outside the generated docs tree
  so reviewers can audit it directly.

Changes:

- File: `DUAL_FRANKA_LIVE_DEBUG_REVIEW.rst`
  - Moved the former `robots/dual_franka/LIVE_DEBUG.md` out of the robot source
    tree and rewrote it as a standalone Chinese review/debug note.
  - Expanded it into a live deployment guide covering scope, script layout,
    Codex state isolation, manual single-skill testing, full task runners,
    interactive/dashboard intervention, log inspection, known specialization
    debt, and a safety checklist.
- File: `docs/source-zh/index.rst`
  - Kept the generated docs toctree unchanged; the live debug note is not
    embedded in the published documentation.
- Removed file: `robots/dual_franka/LIVE_DEBUG.md`
  - The document was not source code and is now tracked at the repository root
    for direct review.

Validation performed:

- RPent lint/syntax:
  - `.venv/bin/python -m ruff check robots/dual_franka scripts/dual_franka_manual_call.py tests/unit_tests/robots/dual_franka`
  - `.venv/bin/python -m py_compile robots/dual_franka/runtime_config.py robots/dual_franka/env_server.py robots/dual_franka/perception.py robots/dual_franka/tools.py robots/dual_franka/tasks.py scripts/dual_franka_manual_call.py`
- Shell syntax:
  - `bash -n scripts/rpent_live_env.sh scripts/run_dual_franka_task3_codex.sh scripts/run_dual_franka_interactive.sh scripts/run_manual_skill.sh`
