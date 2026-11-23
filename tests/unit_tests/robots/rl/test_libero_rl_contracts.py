# Copyright 2026 The RPent Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Offline contracts for adaptive LIBERO RL tools and run options."""

from __future__ import annotations

import argparse
import json
import math
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from robots.rl import geometry, prompt_bundle, robot_spec, tools
from robots.rl.toolkit import RLToolkit
from rpent.dashboard.events import NullDashboardEventSink
from rpent.memory import MemoryManager
from rpent.robots.components.sam3_client import Sam3Result
from rpent.utils import templates

OPEN_GAP = 0.08
SIZE = 32
NEAR, FAR = 0.01, 10.0
GRIPPER_DOWN = np.array([[1.0, 0, 0], [0, -1, 0], [0, 0, -1]])
ALL_TOOLS = {"execute_actions", "view_env", "segment", "back_project"}


class FakeEnv:
    """Apply OSC deltas exactly; fingers follow an incremental command."""

    def __init__(
        self,
        *,
        block_gap: float = 0.0,
        success_below_z: float | None = None,
        depth_m: float = 1.0,
        jitter_m: float = 0.0,
    ) -> None:
        self.xyz = np.array([0.1, -0.2, 1.0])
        self.rotation = Rotation.from_matrix(GRIPPER_DOWN)
        self.finger_command = -1.0
        self.gap = OPEN_GAP
        self.block_gap = block_gap
        self.success_below_z = success_below_z
        self.depth_m = depth_m
        self.jitter_m = jitter_m
        self.terminated = self.truncated = False
        self.actions: list[np.ndarray] = []
        self._observe()

    def _observe(self) -> None:
        half = self.gap / 2
        self.last_obs = {
            "states": np.concatenate(
                [self.xyz, self.rotation.as_rotvec(), [half, -half]]
            ),
            "main_images": np.zeros((4, 4, 3), dtype=np.uint8),
            "control_state": {
                "xyz": self.xyz.copy(),
                "rotation": self.rotation.as_matrix(),
                "control_to_world_rotation": np.eye(3),
                "input_min": np.full(6, -1.0),
                "input_max": np.full(6, 1.0),
                "output_min": np.array([-0.05] * 3 + [-0.5] * 3),
                "output_max": np.array([0.05] * 3 + [0.5] * 3),
            },
        }

    def step(self, action):
        assert not (self.terminated or self.truncated)
        action = np.asarray(action, dtype=np.float64)
        self.actions.append(action)
        self.xyz = self.xyz + action[:3] * tools.POSITION_SCALE_M
        self.xyz[0] += self.jitter_m * (-1) ** len(self.actions)
        self.rotation = (
            Rotation.from_rotvec(action[3:6] * tools.ROTATION_SCALE_RAD) * self.rotation
        )
        self.finger_command = float(
            np.clip(self.finger_command + 0.5 * np.sign(action[6]), -1, 1)
        )
        self.gap = max(OPEN_GAP * (1 - self.finger_command) / 2, self.block_gap)
        self.terminated = bool(
            self.success_below_z is not None and self.xyz[2] < self.success_below_z
        )
        self._observe()
        return self.last_obs, 0.0, self.terminated, False, {}

    def get_task_language(self) -> str:
        return "put the cube in the bowl"

    def render_camera(self, camera_name, height, width, depth=False):
        buffer = np.full((height, width), (1 - NEAR / self.depth_m) / (1 - NEAR / FAR))
        return np.zeros((height, width, 3), dtype=np.uint8), buffer

    def get_camera_meta(self, camera_name, height, width):
        return {
            "intrinsic_K": [[width, 0, width / 2], [0, height, height / 2], [0, 0, 1]],
            "extrinsic_cam2world": np.eye(4).tolist(),
            "depth_near": NEAR,
            "depth_far": FAR,
        }


class FakeSam3:
    def __init__(self, mask: np.ndarray, box: list[float] | None = None) -> None:
        self.mask = mask
        self.box = box

    def segment(self, image, *, text_prompt=None, point=None, min_score=0.2):
        return Sam3Result(
            found=True,
            score=0.91,
            box=self.box,
            mask=self.mask.copy(),
            mask_shape=self.mask.shape,
        )


def bar_mask() -> np.ndarray:
    mask = np.zeros((SIZE, SIZE), dtype=bool)
    mask[8:12, 4:28] = True
    return mask


@pytest.fixture(autouse=True)
def offline_run(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(tools, "IMAGE_SIZE", SIZE)
    monkeypatch.setattr(
        templates, "default_variables", lambda: {"output_dir": "/offline/output"}
    )


def make_toolkit(
    tmp_path: Path,
    mode: str,
    *,
    env: FakeEnv | None = None,
    sam3: FakeSam3 | None = None,
    **options,
) -> RLToolkit:
    return RLToolkit(
        runtime_kwargs={
            "env": env or FakeEnv(),
            "sam3_client": sam3 or FakeSam3(bar_mask()),
        },
        dashboard_events=NullDashboardEventSink(),
        memory=MemoryManager(tmp_path / "memory"),
        observation_mode=mode,
        output_dir=tmp_path / "run",
        **options,
    )


def execute_item(toolkit, name, arguments):
    from rpent.tools import ToolResult

    result = toolkit.execute_tool(
        "execute_actions", {"actions": [{"action": name, **arguments}]}
    )
    if result.is_error or not result.data["results"]:
        return result
    return ToolResult(data=result.data["results"][0], images=result.images)


def test_adaptive_exposes_the_complete_tool_set(tmp_path) -> None:
    toolkit = make_toolkit(tmp_path, "adaptive")
    assert {definition.name for definition in toolkit.list_tools()} == ALL_TOOLS
    assert toolkit.planner_builtin_tools is False


def test_adaptive_returns_images_only_from_view_env(tmp_path) -> None:
    toolkit = make_toolkit(tmp_path, "adaptive")
    opening = toolkit.initial_observation()
    moved = execute_item(toolkit, "move_to", {"xyz": [0.1, -0.2, 0.95]})
    segmented = toolkit.execute_tool("segment", {"prompt": "cube"})
    assert opening.data["task"] == "put the cube in the bowl"
    assert opening.data["state"]["step"] == 0
    assert opening.images == moved.images == segmented.images == []
    assert "mask_id" in segmented.data and "found" not in segmented.data
    view = toolkit.execute_tool("view_env", {})
    assert len(view.images) == 2
    assert view.data["camera"] == "all"
    assert view.data["position_unit"] == "m"
    assert view.data["capture_frame_version"] == view.data["frame_version"] == 0
    toolkit.close()
    rollout = json.loads((tmp_path / "run" / "rollout.json").read_text())
    assert rollout["images_sent"] == 2


@pytest.mark.parametrize("mode", ("image", "text"))
def test_removed_observation_modes_are_rejected(tmp_path, mode) -> None:
    parser = argparse.ArgumentParser()
    robot_spec.get_robot_spec().add_cli_args(parser, use_dashboard=True)
    with pytest.raises(SystemExit):
        parser.parse_args(["--observation-mode", mode])
    with pytest.raises(ValueError, match="only adaptive"):
        prompt_bundle.system_prompt({"observation_mode": mode})
    with pytest.raises(ValueError, match="only adaptive"):
        make_toolkit(tmp_path, mode)
    from robots.rl.rl_episode import build_planner

    with pytest.raises(ValueError, match="only adaptive"):
        build_planner(
            base_url="http://127.0.0.1:1/v1",
            model_name="policy",
            max_tokens=32768,
            request_timeout_s=1,
            episode_timeout_s=1,
            observation_mode=mode,
            history_compaction=None,
            count_tokens=None,
        )


def test_pure_translation_and_pure_rotation_hold_the_other_pose_part(tmp_path) -> None:
    toolkit = make_toolkit(tmp_path, "adaptive")
    start = toolkit.initial_observation().data["state"]["pose"]

    moved = execute_item(toolkit, "move_to", {"xyz": [0.2, -0.1, 0.9]}).data
    turned = execute_item(toolkit, "move_to", {"orientation_rpy": [2.8, 0.3, 1.2]}).data

    assert moved["status"] == turned["status"] == "reached"
    assert moved["pose"]["xyz"] == pytest.approx([0.2, -0.1, 0.9], abs=0.006)
    assert moved["pose"]["orientation_rpy"] == pytest.approx(
        start["orientation_rpy"], abs=0.03
    )
    assert turned["pose"]["xyz"] == pytest.approx(moved["pose"]["xyz"], abs=0.006)
    target = Rotation.from_euler("xyz", [2.8, 0.3, 1.2])
    actual = Rotation.from_euler("xyz", turned["pose"]["orientation_rpy"])
    assert (target * actual.inv()).magnitude() < 0.02
    assert turned["step"] == moved["step"] + 1
    assert set(moved) == {
        "status",
        "step",
        "pose",
        "opening",
        "terminated",
        "truncated",
        "position_error_m",
        "rotation_error_rad",
        "env_steps",
        "frame_version",
        "start_frame_version",
        "action",
        "index",
    }


def test_closing_on_an_object_reports_and_keeps_the_measured_opening(tmp_path) -> None:
    env = FakeEnv(block_gap=0.04)
    toolkit = make_toolkit(tmp_path, "adaptive", env=env)

    closed = execute_item(toolkit, "set_gripper", {"opening": 0.0}).data
    actions_before_lift = len(env.actions)
    lifted = execute_item(toolkit, "move_to", {"xyz": [0.1, -0.2, 1.1]}).data

    assert closed["status"] == "partial"
    assert closed["reason"] == "blocked"
    assert closed["opening"] == pytest.approx(0.5, abs=0.01)
    assert lifted["status"] == "reached"
    assert lifted["opening"] == pytest.approx(0.5, abs=0.01)
    assert all(action[6] == 0 for action in env.actions[actions_before_lift:])


def test_reopening_after_a_squeezed_grasp_is_not_reported_blocked(tmp_path) -> None:
    toolkit = make_toolkit(tmp_path, "adaptive", env=FakeEnv(block_gap=0.04))

    execute_item(toolkit, "set_gripper", {"opening": 0.0})
    opened = execute_item(toolkit, "set_gripper", {"opening": 1.0}).data

    assert opened["status"] == "reached"
    assert opened["opening"] == pytest.approx(1.0, abs=0.01)


def test_residual_jitter_within_tolerance_counts_as_reached(tmp_path) -> None:
    toolkit = make_toolkit(tmp_path, "adaptive", env=FakeEnv(jitter_m=0.0015))

    moved = execute_item(toolkit, "move_to", {"xyz": [0.1, -0.2, 0.9]}).data

    assert moved["status"] == "reached"
    assert moved["pose"]["xyz"] == pytest.approx([0.1, -0.2, 0.9], abs=0.004)


def test_rejected_requests_reuse_the_latest_observation(tmp_path) -> None:
    toolkit = make_toolkit(tmp_path, "adaptive")

    rejected = execute_item(toolkit, "move_to", {})

    assert rejected.is_error and "No tool executed" in rejected.to_text()
    assert rejected.images == []
    assert len(toolkit.state.records()) == 1


def test_environment_success_ends_motion_and_refuses_later_actions(tmp_path) -> None:
    toolkit = make_toolkit(tmp_path, "adaptive", env=FakeEnv(success_below_z=0.95))

    moved = execute_item(toolkit, "move_to", {"xyz": [0.1, -0.2, 0.9]}).data
    refused = execute_item(toolkit, "set_gripper", {"opening": 1.0}).data
    finished = toolkit.completion_result()

    assert moved["status"] == "partial"
    assert moved["reason"] == "terminated"
    assert moved["terminated"] is True
    assert refused["executed_count"] == 0
    assert refused["stop_reason"] == "terminated"
    assert toolkit.state.latest_step == moved["step"]
    assert finished == {"status": "completed", "success": True, "truncated": False}
    assert toolkit.solved()


def test_segment_and_back_project_share_steps_boxes_and_masks(tmp_path) -> None:
    sam3 = FakeSam3(bar_mask(), box=[3.6, 7.2, 27.5, 11.4])
    toolkit = make_toolkit(tmp_path, "adaptive", sam3=sam3)

    found = toolkit.execute_tool("segment", {"prompt": "cube"}).data
    by_mask = toolkit.execute_tool("back_project", {"mask_id": found["mask_id"]}).data
    by_box = toolkit.execute_tool(
        "back_project", {"box": [8, 4, 12, 28], "step": 0}
    ).data
    by_pixel = toolkit.execute_tool("back_project", {"pixel": [9, 9], "step": 0}).data

    assert found["mask_id"] == "m_0_agentview_1"
    assert found["box"] == [7, 3, 12, 28]
    assert found["top_z_m"] == pytest.approx(1.0, abs=1e-3)
    assert found["size_xyz_m"][2] == pytest.approx(0.0, abs=1e-3)
    assert found["axis_yaw"] == pytest.approx(0.0, abs=1e-3)
    assert by_mask["world_xyz"] == found["world_xyz"]
    for field in (
        "camera",
        "step",
        "position_unit",
        "frame_version",
        "capture_frame_version",
    ):
        assert by_mask[field] == found[field]
    assert by_box["world_xyz"] == pytest.approx(found["world_xyz"])
    assert by_pixel["world_xyz"][2] == pytest.approx(1.0, abs=1e-3)
    for rejected in (
        ("back_project", {"pixel": [9, 9]}),
        ("back_project", {"mask_id": found["mask_id"], "step": 0}),
        ("back_project", {"box": [8, 4, 12, 28], "step": 0, "z_range_m": [1.5, 2]}),
        ("segment", {"point": [9, 9]}),
    ):
        assert "error" in toolkit.execute_tool(*rejected).data


def test_mask_without_depth_keeps_the_mask_and_reports_missing_geometry(
    tmp_path,
) -> None:
    toolkit = make_toolkit(tmp_path, "adaptive", env=FakeEnv(depth_m=math.nan))

    found = toolkit.execute_tool("segment", {"prompt": "cube"}).data

    assert "found" not in found and "score" not in found
    assert found["box"] == [8, 4, 12, 28]
    assert found["world_xyz"] is None
    assert found["geometry_unavailable"]
    assert (
        "error"
        in toolkit.execute_tool("back_project", {"mask_id": found["mask_id"]}).data
    )


@pytest.mark.parametrize(
    "angles",
    [
        [0, 0, 0],
        [math.pi, 0, 0.4],
        [0.2, -0.3, 1.1],
        [-math.pi, 0, 0],
        [0.3, math.pi / 2, 0.7],
    ],
)
def test_rpy_output_round_trips_full_rotation_at_wrap_and_singularity(angles):
    matrix = Rotation.from_euler("xyz", angles).as_matrix()
    restored = Rotation.from_euler("xyz", geometry.rpy_of(matrix)).as_matrix()
    assert Rotation.from_matrix(restored @ matrix.T).magnitude() < 2e-6


def test_surface_direction_is_reported_only_for_elongated_points() -> None:
    along = np.linspace(-0.1, 0.1, 41)
    across = np.linspace(-0.01, 0.01, 5)
    a, b = np.meshgrid(along, across)
    heading = 2.0  # equivalent to 2.0 - pi as an undirected axis
    bar = np.column_stack(
        [
            a.ravel() * math.cos(heading) - b.ravel() * math.sin(heading),
            a.ravel() * math.sin(heading) + b.ravel() * math.cos(heading),
            np.full(a.size, 0.3),
        ]
    )
    grid = np.linspace(-0.1, 0.1, 21)
    x, y = np.meshgrid(grid, grid)
    square = np.column_stack([x.ravel(), y.ravel(), np.full(x.size, 0.3)])

    assert geometry.surface_geometry(bar)["axis_yaw"] == pytest.approx(
        2.0 - math.pi, abs=1e-3
    )
    assert geometry.surface_geometry(square)["axis_yaw"] is None
    assert geometry.surface_geometry(bar[:5]) is None


@pytest.mark.parametrize("variant", prompt_bundle.PROMPT_VARIANTS)
def test_prompts_carry_only_the_interface_contract(variant) -> None:
    prompts = robot_spec.get_robot_spec().prompts

    variables = {"observation_mode": "adaptive", "prompt": variant}
    system = prompts.render("system", variables=variables)
    text = system + prompts.render("user", variables=variables)
    assert not any(
        word in text.lower()
        for word in ("pi0", "memory", "guide", "recipe", "seed", "suite")
    )
    # No tools beyond the robot's own, and nothing a planner cannot call.
    assert not any(word in text for word in ("read_image", "shell", "bash", "python"))
    mentioned = {name for name in ALL_TOOLS if name in text}
    assert mentioned <= ALL_TOOLS
    assert "Positions use the world frame" in system
    assert "exactly one tool call" in system


def test_only_fixed_simple_prompt_is_supported() -> None:
    simple = prompt_bundle.system_prompt({"prompt": "simple"})
    assert simple == prompt_bundle.system_prompt({})
    assert "Images are supplied only through view_env" in simple
    for old in ("base", "base_gripper", "base_gripper_camera", "fancy"):
        with pytest.raises(ValueError, match="fixed prompt"):
            prompt_bundle.system_prompt({"prompt": old})


@pytest.mark.parametrize(
    "environment,workflow",
    [
        ("libero", "episode"),
        ("robocasa", "episode"),
        ("robotwin", "episode"),
    ],
)
def test_supported_combinations_use_native_tasks_without_prompt_files(
    environment, workflow, tmp_path
):
    args = argparse.Namespace(
        env=environment,
        workflow=workflow,
        suite="libero_goal",
        task=1,
        task_name="example_task",
        seed=0,
        observation_mode="adaptive",
        prompt="simple",
        output_dir=tmp_path,
    )
    config = robot_spec.get_robot_spec().parse_config(args)
    assert "task_prompt" not in config.prompt_vars
    assert robot_spec.get_robot_spec().prompts.render(
        "system", variables=config.prompt_vars
    )


@pytest.mark.parametrize("environment", ["robocasa", "robotwin"])
def test_unsupported_zap_fails_before_accessing_native_runtime(environment, tmp_path):
    args = argparse.Namespace(
        env=environment, workflow="zap_turn", observation_mode="adaptive"
    )
    with pytest.raises(ValueError, match="supports only episode"):
        robot_spec.get_robot_spec().parse_config(args)
    with pytest.raises(ValueError, match="supports only episode"):
        RLToolkit(
            runtime_kwargs={},
            dashboard_events=NullDashboardEventSink(),
            memory=MemoryManager(tmp_path / "memory"),
            observation_mode="adaptive",
            output_dir=tmp_path / "run",
            environment=environment,
            workflow="zap_turn",
        )
    assert not (tmp_path / "run").exists()


def test_fixed_prompt_reaches_the_prompt_variables() -> None:
    args = argparse.Namespace(
        suite="libero_goal",
        task=1,
        seed=0,
        observation_mode="adaptive",
        prompt="simple",
        output_dir=None,
    )
    config = robot_spec.get_robot_spec().parse_config(args)
    assert config.prompt_vars == {
        "observation_mode": "adaptive",
        "prompt": "simple",
        "environment": "libero",
        "workflow": "episode",
        "world_cord": "base",
        "position_unit": "m",
        "reasoning": True,
        "image_size": 1024,
        "action_mode": "batch",
        "max_actions_per_call": 3,
        "reasoning_history": -1,
        "image_history": -1,
        "perception_mode": "single",
        "max_observe_calls": 8,
        "observe_concurrency": 8,
    }
    assert config.task_desc["prompt"] == "simple"


@pytest.mark.parametrize(
    "override",
    [
        {"planner": "codex"},
        {"planner": "claude_code"},
        {"tui": True},
        {"dashboard": True},
        {"memory_dir": "/memory"},
        {"no_images": True},
    ],
)
def test_options_that_would_change_the_rollout_unnoticed_are_rejected(
    override,
) -> None:
    args = argparse.Namespace(
        planner="api",
        dashboard=False,
        tui=False,
        memory_dir=None,
        memory_repo=None,
        no_images=False,
        observation_mode="adaptive",
    )
    validate = robot_spec.get_robot_spec().validate_args

    validate(args)
    vars(args).update(override)
    with pytest.raises(ValueError):
        validate(args)


def test_planner_sees_only_the_robot_tools(tmp_path) -> None:
    """The api planner adds no built-in tools, so frames come only from view_env."""
    from pydantic_ai.models.function import DeltaToolCall, FunctionModel

    from rpent.planner.api_loop import ApiAgentLoop

    requests = []

    async def stream(messages, info):
        requests.append(
            {tool.name for tool in [*info.function_tools, *info.output_tools]}
        )
        yield {
            0: DeltaToolCall(
                name="execute_actions",
                json_args='{"actions":[{"action":"move_to","xyz":[0.1,-0.2,0.9]}]}',
            )
        }

    toolkit = make_toolkit(tmp_path, "adaptive", env=FakeEnv(success_below_z=0.95))
    try:
        result = ApiAgentLoop(
            model=FunctionModel(stream_function=stream),
            dashboard_events=NullDashboardEventSink(),
            compaction=None,
            sequential_tools=True,
        ).solve(system_prompt="s", user_message="u", toolkit=toolkit, max_turns=2)
        assert result.finish_result["success"]
    finally:
        toolkit.close()
    assert requests == [ALL_TOOLS]


def test_rollouts_serialise_tool_calls() -> None:
    assert robot_spec.get_robot_spec().sequential_tools is True


def test_episode_planner_is_a_thinking_qwen_policy_client() -> None:
    from robots.rl import rl_episode
    from rpent.planner.qwen import QwenChatModel

    planner, client = rl_episode.build_planner(
        base_url="http://127.0.0.1:1/v1",
        model_name="policy",
        max_tokens=32768,
        request_timeout_s=1,
        episode_timeout_s=1,
        observation_mode="adaptive",
        history_compaction=None,
        count_tokens=None,
    )
    assert isinstance(planner.model, QwenChatModel)
    assert planner.sequential_tools is True and planner.no_images is False
    settings = planner.model_settings
    assert settings["extra_body"]["chat_template_kwargs"]["enable_thinking"] is True
    assert settings["extra_body"]["top_k"] == 20
    assert settings["parallel_tool_calls"] is False
    assert "openai_reasoning_effort" not in settings


def test_training_sampling_and_request_audit_reach_the_native_planner(tmp_path):
    import asyncio

    from robots.rl.rl_episode import build_planner

    planner, client = build_planner(
        base_url="http://127.0.0.1:1/v1",
        model_name="policy",
        max_tokens=32768,
        request_timeout_s=1,
        episode_timeout_s=1,
        observation_mode="adaptive",
        history_compaction=None,
        count_tokens=None,
        sampling_params={
            "temperature": 1,
            "top_p": 1,
            "top_k": -1,
            "min_p": 0,
            "presence_penalty": 0,
            "repetition_penalty": 1,
        },
        request_trace_dir=tmp_path / "api_trace",
    )
    try:
        settings = planner.model_settings
        assert settings["top_p"] == 1 and settings["presence_penalty"] == 0
        assert settings["extra_body"]["top_k"] == -1
        assert settings["extra_body"]["chat_template_kwargs"]["enable_thinking"]
        assert planner.model.request_trace.directory == tmp_path / "api_trace"
    finally:
        asyncio.run(client.close())


def test_trainer_keeps_sam_service_failure_separate_from_task_failure(tmp_path):
    from robots.rl.rl_episode import episode_outcome
    from rpent.planner.base import PlannerResult

    toolkit = make_toolkit(tmp_path, "adaptive")
    result = PlannerResult(stats={"turns_used": 1})
    try:
        toolkit.execute_tool("segment", {})
        assert episode_outcome(result, toolkit, 80)["terminal"] is True

        def broken(*args, **kwargs):
            raise ConnectionError("SAM3 unavailable")

        toolkit._primitives._sam3.segment = broken
        toolkit.execute_tool("segment", {"prompt": "cube"})
        outcome = episode_outcome(result, toolkit, 80)
        assert not outcome["terminal"] and not outcome["success"]
        assert outcome["error_kind"] == "tool_transport"
        assert outcome["service_errors"][0]["error"]["type"] == "ConnectionError"
    finally:
        toolkit.close()


def test_concurrent_runtime_clients_follow_their_servers_announced_ports(tmp_path):
    script = tmp_path / "server.py"
    script.write_text(
        "import argparse\n"
        "from rpent.utils.rpc import RpcFacade\n"
        "parser = argparse.ArgumentParser()\n"
        "parser.add_argument('--identity', type=int)\n"
        "args, _ = parser.parse_known_args()\n"
        "facade = RpcFacade()\n"
        "facade._rpc['identity'] = lambda: args.identity\n"
        # The child selects its port, so the client must discover it.
        "facade.serve(transport='http', host='127.0.0.1', port=0, parent_watch=True)\n"
    )

    def spawn(index):
        output = tmp_path / str(index)
        output.mkdir()
        (output / "env.log").write_text("RPC server listening on http://127.0.0.1:1\n")
        return robot_spec._spawn_server(
            argparse.Namespace(cuda_device=None),
            output,
            "env",
            script,
            ["--identity", str(index)],
        )

    results, errors = [], []
    try:
        with ThreadPoolExecutor(max_workers=8) as pool:
            futures = [pool.submit(spawn, index) for index in range(8)]
            for future in futures:
                try:
                    results.append(future.result())
                except Exception as error:
                    errors.append(error)
        if errors:
            raise errors[0]
        assert [client.call("identity") for _, client in results] == list(range(8))
    finally:
        for daemon, _ in results:
            daemon.stop(timeout=2)


def test_runtime_reports_a_server_that_exits_before_announcing(tmp_path):
    script = tmp_path / "crash.py"
    script.write_text("raise SystemExit(7)\n")
    with pytest.raises(RuntimeError, match="did not announce its endpoint"):
        robot_spec._spawn_server(
            argparse.Namespace(cuda_device=None), tmp_path, "env", script, []
        )


@pytest.mark.parametrize(
    "workflow", ["episode_image", "pear", "zap_turn", "zap_episode", "zap_subtask"]
)
def test_removed_workflows_fail_before_creating_environment(tmp_path, workflow):
    with pytest.raises(ValueError, match="supports only episode"):
        make_toolkit(tmp_path, "adaptive", workflow=workflow)
    assert not (tmp_path / "run").exists()


@pytest.mark.parametrize("size", [512, 256])
def test_display_pixel_uses_original_native_depth(tmp_path, monkeypatch, size):
    from robots.rl.images import map_pixel

    monkeypatch.setattr(tools, "IMAGE_SIZE", 1024)
    toolkit = make_toolkit(tmp_path, "adaptive", image_size=size)
    toolkit.initial_observation()
    view = toolkit.execute_tool("view_env", {})
    assert view.data["images"][0]["size"] == [size, size]
    point = [size // 2 - 1, size // 2 - 1]
    mapped = map_pixel(point, (size, size), (1024, 1024))
    expected = toolkit.state.load("agentview_world.npz", step=0)[mapped[0], mapped[1]]
    result = toolkit.execute_tool("back_project", {"pixel": point, "step": 0})
    np.testing.assert_allclose(result.data["world_xyz"], expected, atol=0.0001)
    assert toolkit.state.load("agentview_world.npz", step=0).shape[:2] == (1024, 1024)
    toolkit.close()


@pytest.mark.parametrize("removed", ["multiturn", "zap"])
def test_removed_workflow_names_are_rejected(removed):
    parser = argparse.ArgumentParser()
    robot_spec.get_robot_spec().add_cli_args(parser, use_dashboard=False)
    with pytest.raises(SystemExit):
        parser.parse_args(["--workflow", removed])
