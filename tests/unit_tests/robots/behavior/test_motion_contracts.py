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

from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

from robots.behavior.schemas import (
    BEHAVIOR_TOOL_NAMES,
    ENV_ACTION_SEGMENTS,
    MOVE_TO_SPEC,
)
from robots.behavior.tools import BehaviorPrimitives
from tests.unit_tests.robots.behavior._backend_fakes import (
    _both_hand_request,
    _FakeOfficialBehaviorEnv,
    _official_backend,
    _visual_check,
)

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


def test_public_behavior_surface_is_exactly_nine_tools() -> None:
    assert BEHAVIOR_TOOL_NAMES == EXPECTED_TOOLS


def test_robot_config_separates_world_and_self_collision_padding(tmp_path):
    import yaml

    from robots.behavior.motion import build_robot_config

    names = [f"{h}_arm_joint{i}" for h in ("left", "right") for i in range(1, 8)]
    urdf = tmp_path / "robot.urdf"
    urdf.write_text(
        '<robot name="test">'
        + "".join(f'<joint name="{n}" type="revolute"/>' for n in names)
        + "</robot>"
    )
    source = {
        "collision_link_names": ["base_link", "left_arm_link1"],
        "collision_spheres": {},
        "collision_sphere_buffer": 0.002,
        "self_collision_buffer": {"base_link": 0.02},
        "self_collision_ignore": {},
        "extra_links": {},
        "extra_collision_spheres": {},
        "cspace": {
            "joint_names": names,
            "cspace_distance_weight": [1] * 14,
            "null_space_weight": [1] * 14,
        },
    }
    config_file = tmp_path / "collision.yaml"
    config_file.write_text(yaml.safe_dump({"robot_cfg": {"kinematics": source}}))
    result = build_robot_config(
        {
            "collision_config_path": str(config_file),
            "urdf_path": str(urdf),
            "joint_positions": dict.fromkeys(names, 0),
        }
    )["kinematics"]
    assert result["collision_sphere_buffer"] == 0.002
    assert result["self_collision_buffer"]["base_link"] == pytest.approx(0.018)
    assert result["self_collision_buffer"]["left_arm_link1"] == -0.002
    assert result["self_collision_ignore"] == source["self_collision_ignore"]


@pytest.mark.parametrize("terminal", ["terminated", "truncated", "success"])
def test_chunk_step_rejects_actions_after_episode_end(tmp_path, terminal):
    trigger = {f"{terminal}_on_step": 1}
    backend, env = _official_backend(tmp_path, _FakeOfficialBehaviorEnv(**trigger))
    action = backend._hold_action_from_current_proprio()

    _, _, terminated, truncated, info = backend.chunk_step(action[None, :])

    assert terminated is (terminal == "terminated")
    assert truncated is (terminal == "truncated")
    assert info["done"]["success"] is (terminal == "success")
    assert backend.official_success_latched is (terminal == "success")
    with pytest.raises(RuntimeError, match="episode already ended"):
        backend.chunk_step(action[None, :])
    assert len(env.actions) == 1


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
@pytest.mark.parametrize("terminal_success", [False, True])
def test_planned_motion_preserves_unselected_joint_commands(
    tmp_path, hand, terminal_success
):
    from types import SimpleNamespace

    backend, env = _official_backend(
        tmp_path,
        _FakeOfficialBehaviorEnv(
            terminated_on_step=1 if terminal_success else None,
            success_on_step=1 if terminal_success else None,
        ),
    )
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
    assert result["task_success"] is terminal_success
    assert result["terminated"] is terminal_success
    assert result["truncated"] is False
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
        tmp_path, _FakeOfficialBehaviorEnv(success_on_step=1, terminated_on_step=1)
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
    assert result["terminated"] is True
    assert result["truncated"] is False
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


def test_move_to_contract_separates_single_and_dual_hand_branches() -> None:
    schema = MOVE_TO_SPEC["input_schema"]
    single, dual = schema["oneOf"]
    assert schema["properties"]["hand"]["enum"] == ["left", "right", "both"]
    assert single["properties"]["hand"] == {"enum": ["left", "right"]}
    assert single["required"] == ["hand", "target"]
    assert dual["properties"]["hand"] == {"const": "both"}
    assert dual["required"] == ["hand", "targets", "visual_hand_checks"]

    env = SimpleNamespace(last_move=None)

    def move_to(**kwargs: Any) -> dict[str, Any]:
        env.last_move = kwargs
        return {"status": "failed", "stop_reason": "motion_unavailable"}

    env.move_to = move_to
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
