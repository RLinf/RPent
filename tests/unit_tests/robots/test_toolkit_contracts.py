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

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

from robots.behavior import robot_spec as behavior_robot_spec
from robots.behavior import toolkit as behavior_toolkit
from robots.behavior.dino_v2.client import BehaviorDinoClient
from robots.behavior.dino_v2.encoder import DINOV2_DIMENSION
from robots.behavior.env_client import BehaviorEnvClient
from robots.behavior.rlinf_env import (
    GRIPPER_CLOSE_COMMAND,
    GRIPPER_COMMAND_CONTROL_CYCLES,
    GRIPPER_OPEN_COMMAND,
    OfficialBehaviorBackend,
)
from robots.behavior.robot_spec import get_toolkit
from robots.behavior.schemas import (
    ACTION_DIM,
    BEHAVIOR_TOOL_NAMES,
    ENV_ACTION_SEGMENTS,
    MOVE_TO_SPEC,
    RAW_PROPRIO_SEGMENTS,
    validate_action_chunk,
)
from robots.behavior.tools import BehaviorPrimitives
from robots.libero import robot_spec as libero_robot_spec
from robots.libero import toolkit as libero_toolkit
from robots.robocasa import robot_spec as robocasa_robot_spec
from robots.robocasa import toolkit as robocasa_toolkit
from robots.robotwin import robot_spec as robotwin_robot_spec
from robots.robotwin import toolkit as robotwin_toolkit
from rpent.dashboard.events import NullDashboardEventSink
from rpent.robots import RunConfig


def _run_config(memory_dir: Path, *, recipe_tag: str = "cell-s0") -> RunConfig:
    return RunConfig(
        recipe_tag=recipe_tag,
        output_dir=memory_dir.parent / "run",
        prompt_vars={"memory_dir": str(memory_dir)},
        task_desc={},
    )


@pytest.mark.parametrize(
    ("robot_spec", "toolkit_module", "toolkit_name", "configured_leaf"),
    [
        (robocasa_robot_spec, robocasa_toolkit, "RoboCasaToolkit", "memory"),
        (robotwin_robot_spec, robotwin_toolkit, "RoboTwinToolkit", "memory"),
    ],
)
def test_evaluation_toolkit_factories_use_configured_read_only_memory(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    robot_spec: Any,
    toolkit_module: Any,
    toolkit_name: str,
    configured_leaf: str,
) -> None:
    captured: dict[str, Any] = {}

    def fake_toolkit(**kwargs: Any) -> SimpleNamespace:
        captured.update(kwargs)
        return SimpleNamespace(**kwargs)

    monkeypatch.setattr(toolkit_module, toolkit_name, fake_toolkit)
    resources_dir = tmp_path / robot_spec.__name__
    configured_dir = resources_dir / configured_leaf
    memory_dir = resources_dir / "memory"
    config = _run_config(configured_dir)
    if robot_spec is robocasa_robot_spec:
        (memory_dir / "global").mkdir(parents=True)
        (memory_dir / "global/GLOBAL_MEMORY.md").write_text("# Global memory\n")
        config.output_dir.mkdir(parents=True)
        config.task_desc["task_name"] = "OpenDrawer"

    toolkit = robot_spec.get_toolkit(
        runtime_kwargs={"env": "offline"},
        dashboard_events=NullDashboardEventSink(),
        config=config,
    )

    assert toolkit.memory.root == memory_dir.resolve()
    write = toolkit.memory.get_common_tool_bindings()["write_text_file"][1]
    with pytest.raises(PermissionError, match="writing to memory is denied"):
        write(str(memory_dir / "global" / "strategy.md"), "changed")
    assert captured["runtime_kwargs"] == {"env": "offline"}


@pytest.mark.parametrize(
    ("robot_name", "robot_spec", "toolkit_module", "toolkit_name"),
    [
        ("libero", libero_robot_spec, libero_toolkit, "LiberoToolkit"),
        ("robotwin", robotwin_robot_spec, robotwin_toolkit, "RoboTwinToolkit"),
    ],
)
def test_toolkit_factories_fall_back_to_each_robot_memory_root(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    robot_name: str,
    robot_spec: Any,
    toolkit_module: Any,
    toolkit_name: str,
) -> None:
    default_memory = tmp_path / robot_name / "memory"
    monkeypatch.setattr(
        robot_spec,
        "get_memory_dir",
        lambda requested_robot: (
            default_memory if requested_robot == robot_name else None
        ),
    )
    monkeypatch.setattr(
        toolkit_module,
        toolkit_name,
        lambda **kwargs: SimpleNamespace(**kwargs),
    )
    config = RunConfig(
        recipe_tag="cell-s0",
        output_dir=tmp_path / "run",
        prompt_vars={},
        task_desc={},
    )

    toolkit = robot_spec.get_toolkit(
        runtime_kwargs={},
        dashboard_events=NullDashboardEventSink(),
        config=config,
    )

    assert toolkit.memory.root == default_memory.resolve()


@pytest.mark.parametrize(
    ("mode", "write_allowed"),
    [("eval", False), ("explore", True)],
)
def test_behavior_toolkit_uses_one_official_memory_manager(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    mode: str,
    write_allowed: bool,
) -> None:
    captured: dict[str, Any] = {}

    def fake_toolkit(**kwargs: Any) -> SimpleNamespace:
        captured.update(kwargs)
        return SimpleNamespace(**kwargs)

    monkeypatch.setattr(behavior_toolkit, "BehaviorToolkit", fake_toolkit)
    memory_dir = tmp_path / "memory"
    recipe_tag = "turning_on_radio_s1"
    config = RunConfig(
        recipe_tag=recipe_tag,
        output_dir=tmp_path / "run",
        prompt_vars={
            "mode": mode,
            "memory_dir": str(memory_dir),
        },
        task_desc={},
    )

    toolkit = behavior_robot_spec.get_toolkit(
        runtime_kwargs={},
        dashboard_events=NullDashboardEventSink(),
        config=config,
        mode="exploration" if mode == "explore" else "evaluation",
    )

    assert toolkit.memory is captured["memory"]
    assert toolkit.memory.root == memory_dir.resolve()
    write = toolkit.memory.get_common_tool_bindings()["write_text_file"][1]
    destination = memory_dir / "_internal" / "inbox" / recipe_tag / "wip" / "notes.md"
    if write_allowed:
        write(str(destination), "evidence")
        assert destination.read_text() == "evidence"
    else:
        with pytest.raises(PermissionError, match="writing to memory is denied"):
            write(str(destination), "evidence")

    component_names = {
        item["name"]
        for item in behavior_robot_spec.BEHAVIOR_DASHBOARD_SPEC["runtime_components"]
    }
    assert component_names == {"env", "vla", "dino"}


EXPECTED_TOOLS = (
    "pi0_nav_pick",
    "observe",
    "pixel_to_world",
    "navigate_to",
    "move_to",
    "rotate_wrist",
    "close",
    "open",
    "press",
)


class _FakeEnv:
    total_env_steps = 0
    official_success_latched = False
    official_success_receipt = None

    def __init__(self) -> None:
        self.last_move: dict[str, Any] | None = None
        self.gripper_calls: list[tuple[str, dict[str, Any]]] = []

    def move_to(self, **kwargs: Any) -> dict[str, Any]:
        self.last_move = kwargs
        return {"status": "failed", "stop_reason": "motion_unavailable"}

    def close_gripper(self, **kwargs: Any) -> dict[str, Any]:
        self.gripper_calls.append(("close_gripper", kwargs))
        return {"status": "success"}

    def open_gripper(self, **kwargs: Any) -> dict[str, Any]:
        self.gripper_calls.append(("open_gripper", kwargs))
        return {"status": "success"}


class _FakeRpcClient:
    def __init__(self) -> None:
        self.calls: list[tuple[str, dict[str, Any]]] = []

    def call(
        self,
        method: str,
        *,
        args: tuple = (),
        kwargs: dict[str, Any] | None = None,
        timeout_s: float | None = None,
    ) -> dict[str, Any]:
        self.calls.append((method, kwargs or {}))
        if method == "env.get_env_meta":
            return {}
        if method == "dino.get_meta":
            return {
                "runtime": "behavior_dino",
                "dimension": DINOV2_DIMENSION,
            }
        return {"status": "success"}


class _FakeOfficialBehaviorEnv:
    def __init__(
        self,
        *_args: Any,
        terminated_on_step: int | None = None,
        success_on_step: int | None = None,
        **_kwargs: Any,
    ) -> None:
        self.actions: list[np.ndarray] = []
        self.closed = False
        self.terminated_on_step = terminated_on_step
        self.success_on_step = success_on_step
        self.raw = np.zeros(256, dtype=np.float32)
        self.raw[RAW_PROPRIO_SEGMENTS["trunk"]] = np.asarray(
            [0.11, 0.12, 0.13, 0.14],
            dtype=np.float32,
        )
        self.raw[RAW_PROPRIO_SEGMENTS["left_arm"]] = np.linspace(
            0.21,
            0.27,
            7,
            dtype=np.float32,
        )
        self.raw[RAW_PROPRIO_SEGMENTS["right_arm"]] = np.linspace(
            -0.31,
            -0.37,
            7,
            dtype=np.float32,
        )
        self.raw[RAW_PROPRIO_SEGMENTS["left_gripper"]] = 0.05
        self.raw[RAW_PROPRIO_SEGMENTS["right_gripper"]] = 0.06

    def _obs(self) -> dict[str, Any]:
        return {
            "main_images": np.zeros((8, 8, 3), dtype=np.uint8),
            "wrist_images": np.zeros((2, 8, 8, 3), dtype=np.uint8),
            "states": self.raw.copy(),
            "task_descriptions": "turn on the radio",
        }

    def env_reset(self) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        return [self._obs()], [{"done": {"success": False}}]

    def env_chunk_step(
        self,
        chunk_actions: Any,
    ) -> tuple[
        list[list[dict[str, Any]]],
        np.ndarray,
        np.ndarray,
        np.ndarray,
        list[list[dict[str, Any]]],
    ]:
        if hasattr(chunk_actions, "detach"):
            action_array = chunk_actions.detach().cpu().numpy().astype(np.float32)
        else:
            action_array = np.asarray(chunk_actions, dtype=np.float32)
        assert action_array.shape == (1, 1, ACTION_DIM)
        self.actions.append(action_array[0, 0].copy())
        step_index = len(self.actions)
        terminated = self.terminated_on_step == step_index
        success = self.success_on_step == step_index
        return (
            [[self._obs()]],
            np.asarray([[0.0]], dtype=np.float32),
            np.asarray([[terminated]], dtype=bool),
            np.asarray([[False]], dtype=bool),
            [[{"done": {"success": success}}]],
        )

    def close(self) -> None:
        self.closed = True


def _both_hand_request() -> dict[str, Any]:
    return {
        "hand": "both",
        "targets": {
            "left": {"delta_xyz": [0.01, 0.0, 0.0], "frame": "world"},
            "right": {"delta_xyz": [-0.01, 0.0, 0.0], "frame": "eef"},
        },
        "visual_hand_checks": {
            "left": {
                "camera": "left_wrist",
                "frame_id": "left-frame",
                "selected_hand": "left",
                "assessment": "selected_hand_visually_confirmed",
            },
            "right": {
                "camera": "right_wrist",
                "frame_id": "right-frame",
                "selected_hand": "right",
                "assessment": "selected_hand_visually_confirmed",
            },
        },
    }


def _visual_check(hand: str) -> dict[str, str]:
    return {
        "camera": f"{hand}_wrist",
        "frame_id": f"{hand}-frame",
        "selected_hand": hand,
        "assessment": "selected_hand_visually_confirmed",
    }


def _official_backend(
    tmp_path: Path,
    fake_env: _FakeOfficialBehaviorEnv | None = None,
) -> tuple[OfficialBehaviorBackend, _FakeOfficialBehaviorEnv]:
    env = fake_env or _FakeOfficialBehaviorEnv()
    backend = OfficialBehaviorBackend(
        meta={
            "task_name": "turning_on_radio",
            "task_language": "turn on the radio",
            "activity_definition_id": 0,
            "activity_instance_id": 242,
            "public_seed": 0,
            "scene_model": "house_double_floor_lower",
            "max_episode_steps": 64,
        },
        output_dir=tmp_path,
        behavior_env_cls=lambda *_args, **_kwargs: env,
        cfg=object(),
    )
    backend.reset()
    return backend, env


def test_public_behavior_surface_is_exactly_nine_tools() -> None:
    assert BEHAVIOR_TOOL_NAMES == EXPECTED_TOOLS


def test_pixel_projection_uses_observed_depth_and_rejects_old_frame(tmp_path):
    backend, _ = _official_backend(tmp_path)
    frames = {
        camera: {
            "rgb": np.zeros((3, 3, 3), dtype=np.uint8),
            "depth": np.ones((3, 3), dtype=np.float32) * 2,
            "intrinsic": np.array([[2.0, 0, 1], [0, 2, 1], [0, 0, 1]]),
            "camera_to_world": np.eye(4),
        }
        for camera in ("head", "left_wrist", "right_wrist")
    }
    backend._call_actor = lambda _function: frames
    observation = backend.observe(camera="head")
    result = backend.pixel_to_world(
        camera="head", frame_id=observation["frame_id"], u=1, v=1
    )
    assert result["world_xyz"] == [0.0, 0.0, -2.0]
    assert result["task_success"] is False
    assert (
        backend._projection(result["projection_id"])["world_xyz"] == result["world_xyz"]
    )
    backend.chunk_step(backend._hold_action_from_current_proprio()[None, :])
    assert (
        backend.pixel_to_world(
            camera="head", frame_id=observation["frame_id"], u=1, v=1
        )["stop_reason"]
        == "stale_frame"
    )


@pytest.mark.parametrize("hand", ["left", "both"])
def test_planned_motion_preserves_unselected_joint_commands(tmp_path, hand):
    from types import SimpleNamespace

    backend, env = _official_backend(tmp_path)
    hold = backend._hold_action_from_current_proprio().copy()
    hands = ("left", "right") if hand == "both" else (hand,)
    poses = {
        h: {"position": np.zeros(3), "quaternion_xyzw": [0, 0, 0, 1]}
        for h in ("left", "right")
    }
    state = {"hands": poses, "control_dt": 1 / 60}
    backend._call_actor = lambda _: state
    backend._get_motion_state = lambda: state
    names = [f"{h}_arm_joint{i}" for h in hands for i in range(1, 8)]
    trajectory = np.ones((2, len(names)), dtype=np.float32) * 0.1
    backend._motion_planner = SimpleNamespace(
        plan=lambda *_: {
            "success": True,
            "joint_names": names,
            "positions": trajectory,
            "dt": 1 / 60,
        }
    )
    request = (
        _both_hand_request()
        if hand == "both"
        else {"hand": hand, "target": {"delta_xyz": [0, 0, 0], "frame": "world"}}
    )
    if hand == "both":
        for target in request["targets"].values():
            target["delta_xyz"] = [0, 0, 0]
    result = backend.move_to(**request)
    assert result["primitive_success"] is True
    assert result["task_success"] is False
    for action in env.actions:
        expected = hold.copy()
        for selected in hands:
            expected[ENV_ACTION_SEGMENTS[f"{selected}_arm"]] = 0.1
        np.testing.assert_allclose(action, expected)


def test_failed_plan_does_not_execute_actions(tmp_path):
    from types import SimpleNamespace

    backend, env = _official_backend(tmp_path)
    backend._call_actor = lambda _: {
        "hands": {"left": {"position": np.zeros(3), "quaternion_xyzw": [0, 0, 0, 1]}}
    }
    backend._motion_planner = SimpleNamespace(
        plan=lambda *_: {
            "success": False,
            "stop_reason": "planning_failed",
            "details": "collision",
        }
    )
    result = backend.move_to(
        hand="left", target={"delta_xyz": [0, 0, 0.03], "frame": "world"}
    )
    assert result["stop_reason"] == "planning_failed"
    assert not env.actions


def test_navigation_swept_footprint_rejects_obstacles():
    from robots.behavior.motion import navigation_collision

    state = {
        "base_position": np.zeros(3),
        "robot_aabb": [[-0.1, -0.1, 0], [0.1, 0.1, 0.5]],
        "obstacles": {"box": {"low": [0.4, -0.1, 0.1], "high": [0.6, 0.1, 0.3]}},
    }
    assert navigation_collision(state, np.array([1.0, 0, 0])) == "box"
    assert navigation_collision(state, np.array([-0.1, 0, 0])) is None


def test_rotate_requires_selected_hand_confirmation_before_motion(tmp_path):
    backend, env = _official_backend(tmp_path)
    with pytest.raises(ValueError, match="visual_hand_check"):
        backend.rotate_wrist(hand="left", angle_deg=5)
    with pytest.raises(ValueError, match="visual_hand_check"):
        backend.rotate_wrist(
            hand="left", angle_deg=5, visual_hand_check=_visual_check("right")
        )
    assert not env.actions


def test_navigation_brake_propagates_raw_success(tmp_path):
    backend, env = _official_backend(
        tmp_path, _FakeOfficialBehaviorEnv(success_on_step=1)
    )
    backend._call_actor = lambda _: {
        "base_position": np.zeros(3),
        "base_quaternion_xyzw": [0, 0, 0, 1],
        "robot_aabb": [[-0.1, -0.1, 0], [0.1, 0.1, 0.5]],
        "obstacles": {},
        "control_dt": 1 / 60,
    }
    result = backend.navigate_to(
        relative_motion={
            "kind": "translation",
            "direction": "forward",
            "distance_m": 0.001,
        }
    )
    assert len(env.actions) == 1
    assert result["task_success"] is True
    assert result["stop_reason"] == "official_task_success"


def test_motion_result_refreshes_policy_observation_without_extra_rpc(tmp_path):
    observation = {
        "main_images": np.zeros((16, 16, 3), dtype=np.uint8),
        "states": np.arange(256, dtype=np.float32),
    }

    class Env:
        def close_gripper(self, **kwargs):
            return {
                "primitive_success": True,
                "task_success": False,
                "_observation": observation,
                "info": {"done": {"success": False}},
            }

        def current_observation(self):
            raise AssertionError("post-action frame must come from the executed action")

    primitives = BehaviorPrimitives(
        env=Env(), task_name="turning_on_radio", output_dir=tmp_path
    )
    result = primitives.close(hand="left", visual_hand_check=_visual_check("left"))
    assert primitives.current_observation is observation
    assert "_observation" not in result
    assert result["task_success"] is False


def test_gripper_close_builds_hold_action_and_target_command(tmp_path: Path) -> None:
    backend, env = _official_backend(tmp_path)

    result = backend.close(hand="left", visual_hand_check=_visual_check("left"))
    chunk = validate_action_chunk(np.stack(env.actions, axis=0))
    first = chunk[0]

    assert result["status"] == "ok"
    assert result["primitive_success"] is True
    assert result["task_success"] is False
    assert result["stop_reason"] == "requested_actions_completed"
    assert result["visual_hand_check"] == _visual_check("left")
    assert result["visual_hand_check_verification"] == "not_verified"
    assert result["action_shape"] == [1, 23]
    assert result["action_chunk_shape"] == [GRIPPER_COMMAND_CONTROL_CYCLES, 23]
    assert chunk.shape == (GRIPPER_COMMAND_CONTROL_CYCLES, 23)
    assert np.allclose(chunk, first)
    assert np.allclose(first[ENV_ACTION_SEGMENTS["base"]], 0.0)
    assert np.allclose(
        first[ENV_ACTION_SEGMENTS["trunk"]],
        env.raw[RAW_PROPRIO_SEGMENTS["trunk"]],
    )
    assert np.allclose(
        first[ENV_ACTION_SEGMENTS["left_arm"]],
        env.raw[RAW_PROPRIO_SEGMENTS["left_arm"]],
    )
    assert np.allclose(
        first[ENV_ACTION_SEGMENTS["right_arm"]],
        env.raw[RAW_PROPRIO_SEGMENTS["right_arm"]],
    )
    assert first[ENV_ACTION_SEGMENTS["left_gripper"]][0] == GRIPPER_CLOSE_COMMAND
    assert first[ENV_ACTION_SEGMENTS["right_gripper"]][0] == GRIPPER_OPEN_COMMAND


def test_gripper_latch_holds_non_target_hand_and_open_echoes_release_check(
    tmp_path: Path,
) -> None:
    backend, env = _official_backend(tmp_path)

    backend.close(hand="left", visual_hand_check=_visual_check("left"))
    env.actions.clear()
    release_check = {
        "camera": "head",
        "frame_id": "release-frame",
        "assessment": "target_visibly_released",
    }
    result = backend.open(
        hand="right",
        visual_hand_check=_visual_check("right"),
        release_visual_check=release_check,
    )
    first = validate_action_chunk(np.stack(env.actions, axis=0))[0]

    assert result["status"] == "ok"
    assert result["release_visual_check"] == release_check
    assert result["release_visual_check_verification"] == "not_verified"
    assert first[ENV_ACTION_SEGMENTS["left_gripper"]][0] == GRIPPER_CLOSE_COMMAND
    assert first[ENV_ACTION_SEGMENTS["right_gripper"]][0] == GRIPPER_OPEN_COMMAND


def test_gripper_task_success_uses_only_official_done_success(tmp_path: Path) -> None:
    terminated_env = _FakeOfficialBehaviorEnv(terminated_on_step=1)
    backend, env = _official_backend(tmp_path / "terminated", terminated_env)

    result = backend.close(hand="left", visual_hand_check=_visual_check("left"))

    assert result["stop_reason"] == "terminated"
    assert result["primitive_success"] is True
    assert result["task_success"] is False
    assert backend.official_success_latched is False
    assert len(env.actions) == 1
    rejected = backend.open(hand="left", visual_hand_check=_visual_check("left"))
    assert rejected["stop_reason"] == "episode_ended"
    assert len(env.actions) == 1

    success_env = _FakeOfficialBehaviorEnv(success_on_step=1)
    success_backend, _success_env = _official_backend(tmp_path / "success", success_env)
    success = success_backend.open(
        hand="right",
        visual_hand_check=_visual_check("right"),
    )

    assert success["stop_reason"] == "official_task_success"
    assert success["task_success"] is True
    assert success["official_success_receipt"]["source"] == 'info["done"]["success"]'


def test_numpy_boolean_cannot_become_official_success(tmp_path: Path) -> None:
    from robots.behavior.terminal_success import (
        make_raw_success_receipt,
        official_task_success,
    )

    info = {"done": {"success": np.bool_(True)}}
    assert official_task_success(info) is False
    assert make_raw_success_receipt(info, env_step=1) is None

    backend, env = _official_backend(tmp_path)
    with pytest.raises(TypeError, match="must be the Python boolean True"):
        backend._note_info(info)

    original_chunk_step = env.env_chunk_step

    def numpy_success(actions: Any) -> tuple:
        observations, rewards, terms, truncs, infos = original_chunk_step(actions)
        infos[-1][0]["done"]["success"] = np.bool_(True)
        return observations, rewards, terms, truncs, infos

    env.env_chunk_step = numpy_success
    with pytest.raises(TypeError, match="must be the Python boolean True"):
        backend.chunk_step(backend._hold_action_from_current_proprio()[None, :])
    assert backend.official_success_latched is False


def test_gripper_execution_error_uses_motion_error_envelope(tmp_path: Path) -> None:
    backend, env = _official_backend(tmp_path)
    assert backend._last_obs is not None
    backend._last_obs["states"][RAW_PROPRIO_SEGMENTS["trunk"]] = np.nan

    result = backend.open(hand="left", visual_hand_check=_visual_check("left"))

    assert result["status"] == "failed"
    assert result["primitive_success"] is False
    assert result["task_success"] is False
    assert result["stop_reason"] == "error"
    assert "NaN or infinity" in result["error"]
    assert env.actions == []


def test_backend_lifecycle_close_still_closes_env_without_primitive_args(
    tmp_path: Path,
) -> None:
    backend, env = _official_backend(tmp_path)

    assert backend.close() == {"status": "ok", "closed": True}
    assert env.closed is True
    assert backend.close() == {
        "status": "ok",
        "closed": True,
        "already_closed": True,
    }


@pytest.mark.parametrize(
    "ending", ["contact", "terminated", "official_task_success", "duration_limit"]
)
def test_press_executes_bounded_hold_actions_and_reports_raw_success(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, ending: str
) -> None:
    backend, env = _official_backend(
        tmp_path,
        _FakeOfficialBehaviorEnv(
            terminated_on_step=1 if ending == "terminated" else None,
            success_on_step=1 if ending == "official_task_success" else None,
        ),
    )

    def state():
        return {
            "control_dt": 0.05,
            "hands": {
                "left": {
                    "position": np.array([0.0, 0.0, len(env.actions) * 0.0005]),
                    "approach_direction": np.array([0.0, 0.0, 1.0]),
                    "joint_positions": env.raw[RAW_PROPRIO_SEGMENTS["left_arm"]],
                    "joint_lower_limits": np.full(7, -3.0),
                    "joint_upper_limits": np.full(7, 3.0),
                    "jacobian": np.eye(6, 7),
                    "contacts": ["button"]
                    if ending == "contact" and env.actions
                    else [],
                }
            },
        }

    monkeypatch.setattr(backend, "_get_motion_state", state)
    hold = backend._hold_action_from_current_proprio()
    result = backend.press(
        hand="left", visual_hand_check=_visual_check("left"), duration_s=0.1
    )
    assert result["stop_reason"] == ending
    assert result["task_success"] is (ending == "official_task_success")
    assert result["primitive_success"] is (
        ending in {"contact", "official_task_success"}
    )
    assert result["executed_steps"] == (2 if ending == "duration_limit" else 1)
    untouched = np.ones(23, dtype=bool)
    untouched[ENV_ACTION_SEGMENTS["left_arm"]] = False
    for action in env.actions:
        assert validate_action_chunk(action[None, :]).shape == (1, 23)
        np.testing.assert_array_equal(action[untouched], hold[untouched])
    assert result["visual_hand_check_verification"] == "not_verified"


@pytest.mark.parametrize("duration", [0, -1, 11, float("nan"), float("inf")])
def test_press_rejects_invalid_duration_before_motion(tmp_path: Path, duration: float):
    backend, env = _official_backend(tmp_path)
    with pytest.raises(ValueError, match="duration_s"):
        backend.press(
            hand="left", visual_hand_check=_visual_check("left"), duration_s=duration
        )
    assert not env.actions


def test_behavior_toolkit_factory_maps_shared_modes(tmp_path: Path) -> None:
    config = RunConfig(
        recipe_tag="turning_on_radio_s0",
        output_dir=tmp_path / "run",
        prompt_vars={
            "mode": "eval",
            "memory_dir": str(tmp_path / "memory"),
            "task_name": "turning_on_radio",
            "public_seed": 0,
            "max_episode_steps": 64,
        },
        task_desc={},
    )
    dashboard_events = NullDashboardEventSink()

    evaluation = get_toolkit(
        runtime_kwargs={},
        dashboard_events=dashboard_events,
        config=config,
        mode="evaluation",
    )
    exploration = get_toolkit(
        runtime_kwargs={"output_dir": tmp_path / "session"},
        dashboard_events=dashboard_events,
        config=config,
        mode="exploration",
    )
    config.prompt_vars["mode"] = "explore"
    inherited = get_toolkit(
        runtime_kwargs={},
        dashboard_events=dashboard_events,
        config=config,
    )

    assert evaluation.primitives.behavior_phase == "eval"
    assert exploration.primitives.behavior_phase == "explore"
    assert inherited.primitives.behavior_phase == "eval"
    assert evaluation.memory._memory_access == "read_only"
    assert exploration.memory._memory_access == "inbox_write"
    assert exploration.primitives.output_dir == tmp_path / "session"
    with pytest.raises(ValueError, match="one attempt per session"):
        get_toolkit(
            runtime_kwargs={},
            dashboard_events=dashboard_events,
            config=config,
            mode="exploration",
            attempts_per_session=1,
        )
    with pytest.raises(ValueError, match="unsupported BEHAVIOR toolkit mode"):
        get_toolkit(
            runtime_kwargs={},
            dashboard_events=dashboard_events,
            config=config,
            mode="unsupported",
        )


@pytest.mark.parametrize("construction_fails", [False, True])
def test_behavior_toolkit_owns_session_env_not_shared_models(
    tmp_path, monkeypatch, construction_fails
):
    from robots.behavior import runtime

    stopped = []
    launched = []
    model = object()
    dino = object()
    args = SimpleNamespace()
    config = RunConfig(
        recipe_tag="turning_on_radio_s0",
        output_dir=tmp_path,
        prompt_vars={
            "memory_dir": str(tmp_path / "memory"),
            "task_name": "turning_on_radio",
            "public_seed": 0,
        },
        task_desc={},
    )

    def init_runtime(received_args, output_dir, dashboard_events, components):
        assert received_args is args
        assert components == {"env"}
        launched.append(output_dir)
        return [SimpleNamespace(stop=lambda: stopped.append(output_dir))], {
            "env": _FakeEnv()
        }

    monkeypatch.setattr(runtime, "init_runtime", init_runtime)
    if construction_fails:

        def fail(**kwargs):
            raise RuntimeError("toolkit construction failed")

        monkeypatch.setattr(behavior_toolkit, "BehaviorToolkit", fail)
    kwargs = {"env_args": args, "model": model, "dino_component": dino}
    for index in range(2):
        output = tmp_path / f"session_{index}"
        if construction_fails:
            with pytest.raises(RuntimeError, match="toolkit construction failed"):
                get_toolkit(
                    runtime_kwargs=kwargs,
                    dashboard_events=NullDashboardEventSink(),
                    config=config,
                    mode="exploration",
                    state_output_dir=output,
                )
        else:
            toolkit = get_toolkit(
                runtime_kwargs=kwargs,
                dashboard_events=NullDashboardEventSink(),
                config=config,
                mode="exploration",
                state_output_dir=output,
            )
            assert toolkit.primitives.model is model
            assert toolkit.primitives.dino_component is dino
            toolkit.close()
            toolkit.close()
        assert stopped == launched
    assert len(stopped) == 2
    assert kwargs == {"env_args": args, "model": model, "dino_component": dino}


def test_behavior_clients_and_tools_use_explicit_component_rpc_names() -> None:
    rpc = _FakeRpcClient()
    client = BehaviorEnvClient(rpc, expected_meta={})
    dino_client = BehaviorDinoClient(rpc, expected_meta={"runtime": "behavior_dino"})
    env = _FakeEnv()
    primitives = BehaviorPrimitives(env=env, task_name="turning_on_radio")

    client.close_gripper(hand="right")
    client.open_gripper(hand="right")
    primitives.close(hand="right")
    primitives.open(hand="right")

    assert rpc.calls == [
        ("env.get_env_meta", {}),
        ("dino.get_meta", {}),
        ("env.close_gripper", {"hand": "right"}),
        ("env.open_gripper", {"hand": "right"}),
    ]
    assert dino_client.server_meta["dimension"] == DINOV2_DIMENSION
    assert env.gripper_calls == [
        ("close_gripper", {"hand": "right"}),
        ("open_gripper", {"hand": "right"}),
    ]
    assert "env.close_gripper" in BehaviorEnvClient._TIMEOUT_S
    assert "env.open_gripper" in BehaviorEnvClient._TIMEOUT_S
    assert "env.close" not in BehaviorEnvClient._TIMEOUT_S
    assert "env.open" not in BehaviorEnvClient._TIMEOUT_S


def test_move_to_contract_separates_single_and_dual_hand_branches() -> None:
    schema = MOVE_TO_SPEC["input_schema"]
    single, dual = schema["oneOf"]
    assert schema["properties"]["hand"]["enum"] == ["left", "right", "both"]
    assert single["properties"]["hand"] == {"enum": ["left", "right"]}
    assert single["required"] == ["hand", "target"]
    assert dual["properties"]["hand"] == {"const": "both"}
    assert dual["required"] == ["hand", "targets", "visual_hand_checks"]

    env = _FakeEnv()
    primitives = BehaviorPrimitives(env=env, task_name="turning_on_radio")
    for hand in ("left", "right"):
        request = {"hand": hand, "target": {"delta_xyz": [0.0, 0.0, 0.01]}}
        primitives.move_to(**request)
        assert env.last_move == request

    both = _both_hand_request()
    primitives.move_to(**both)
    assert env.last_move == both
    invalid = _both_hand_request()
    invalid["targets"] = {"left": invalid["targets"]["left"]}
    with pytest.raises(ValueError, match="exactly left and right"):
        primitives.move_to(**invalid)
