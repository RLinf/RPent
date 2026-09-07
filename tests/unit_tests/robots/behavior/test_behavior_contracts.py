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

"""Offline contracts for the BEHAVIOR toolkit."""

from __future__ import annotations

import json
import os
import socket
import sys
import threading
import time
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from robots.behavior import build_memory_cli
from robots.behavior.dino_v2.client import BehaviorDinoClient
from robots.behavior.dino_v2.encoder import DINOV2_DIMENSION
from robots.behavior.dino_v2.server import BehaviorDinoFacade
from robots.behavior.env_client import BehaviorEnvClient
from robots.behavior.env_server import BehaviorEnvFacade
from robots.behavior.rlinf_env import (
    GRIPPER_CLOSE_COMMAND,
    GRIPPER_COMMAND_CONTROL_CYCLES,
    GRIPPER_OPEN_COMMAND,
    OfficialBehaviorBackend,
)
from robots.behavior.robot_spec import get_toolkit
from robots.behavior.schemas import (
    BEHAVIOR_TOOL_NAMES,
    ENV_ACTION_SEGMENTS,
    MOVE_TO_SPEC,
    RAW_PROPRIO_SEGMENTS,
    validate_action_chunk,
)
from robots.behavior.toolkit import BehaviorToolkit
from robots.behavior.tools import BehaviorPrimitives
from rpent.dashboard.events import NullDashboardEventSink
from rpent.memory import MemoryManager
from rpent.robots import RunConfig
from rpent.utils.daemon import ProcessDaemon, pick_free_port
from rpent.utils.rpc.http_rpc import HttpRpcClient

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


def test_episode_memory_is_structured_in_public_results():
    from robots.behavior.memory.index import empty_episode_memory_index

    class Encoder:
        def encode_batch(self, images):
            vector = np.zeros(DINOV2_DIMENSION, dtype=np.float32)
            vector[0] = 1
            return [vector if image is not None else None for image in images]

    class Env:
        def observe(self, **kwargs):
            return {"status": "ok", "info": {"done": {"success": False}}}

    primitives = BehaviorPrimitives(
        env=Env(),
        initial_observation={"main_images": np.zeros((8, 8, 3), dtype=np.uint8)},
        episode_memory_index=empty_episode_memory_index(),
        dino_component=Encoder(),
    )
    decision = primitives.snapshot()["episode_memory"]
    assert isinstance(decision, dict)
    assert decision["decision"] == "record_new"
    assert decision["candidate_count_after_task_filter"] == 0
    assert decision["stage_inference"] is None
    assert primitives.observe(camera="head")["episode_memory"] == decision
    assert json.loads(json.dumps(decision)) == decision


@pytest.mark.parametrize(
    ("task_name", "mode", "seed", "instruction"),
    [
        (
            "turning_on_radio",
            "explore",
            0,
            "Turn on the radio receiver that's on the table in the living room.",
        ),
        (
            "picking_up_trash",
            "explore",
            0,
            "Put the three can of soda from the living room inside the tash can "
            "in the kitchen.",
        ),
        (
            "picking_up_trash",
            "eval",
            10,
            "Put the three can of soda from the living room inside the tash can "
            "in the kitchen.",
        ),
    ],
)
@pytest.mark.parametrize("batched", [False, True])
def test_runtime_preserves_rlinf_task_language(
    tmp_path: Path,
    task_name: str,
    mode: str,
    seed: int,
    instruction: str,
    batched: bool,
) -> None:
    import argparse

    from robots.behavior import runtime

    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir")
    runtime.add_cli_args(parser, use_dashboard=False)
    args = parser.parse_args(
        [
            "--task-name",
            task_name,
            "--public-seed",
            str(seed),
            "--behavior-mode",
            mode,
            "--behavior-repo",
            str(tmp_path / "RLinf"),
            "--output-dir",
            str(tmp_path / "output"),
        ]
    )
    config = runtime.parse_config(args)
    meta = runtime.env_runtime_contract(args)
    assert meta["task_language"] == instruction
    for key in ("task_language", "task_instruction", "instruction"):
        assert config.prompt_vars[key] == instruction

    class Rpc:
        language = instruction

        def call(self, method, **kwargs):
            if method == "env.get_env_meta":
                return meta
            assert method == "env.reset"
            text = [self.language] if batched else self.language
            return {"task_descriptions": text}, {"done": {"success": False}}

    rpc = Rpc()
    connected = runtime._connect_env(args, rpc, config.output_dir)
    expected = [instruction] if batched else instruction
    assert connected["initial_observation"]["task_descriptions"] == expected
    assert connected["env"].official_success_latched is False

    rpc.language = "Put the three soda cans from the living room inside the trash can in the kitchen."
    with pytest.raises(RuntimeError, match="task language does not match TaskSpec"):
        runtime._connect_env(args, rpc, config.output_dir)


def test_env_endpoint_discovery_uses_actual_bind_and_ignores_old_log(tmp_path: Path):
    from robots.behavior.runtime import _wait_for_server_endpoint

    log = tmp_path / "env.log"
    log.write_text("RPC server listening on http://127.0.0.1:1\n")
    offset = log.stat().st_size
    daemon = ProcessDaemon(
        "test_env",
        [
            sys.executable,
            "-c",
            "from types import SimpleNamespace; "
            "from robots.behavior.env_server import BehaviorEnvFacade; "
            "print('Ray started; no application endpoint yet', flush=True); "
            "BehaviorEnvFacade(backend=SimpleNamespace(close=lambda: None), "
            "meta={'task_language': 'test'}).serve("
            "transport='http', host='127.0.0.1', port=0, parent_watch=True)",
        ],
        log_path=str(log),
    )
    daemon.start()
    rpc = None
    try:
        endpoint = _wait_for_server_endpoint(daemon, log_offset=offset)
        assert endpoint != "http://127.0.0.1:1"
        rpc = HttpRpcClient(endpoint)
        assert rpc.call("healthz") == {"status": "ok"}
        assert rpc.call("env.get_env_meta") == {"task_language": "test"}
        assert rpc.call("shutdown") == {"ok": True}
    finally:
        if rpc is not None:
            rpc.close()
        daemon.stop()


def test_env_endpoint_discovery_reports_early_exit(tmp_path: Path):
    from types import SimpleNamespace

    from robots.behavior.runtime import _wait_for_server_endpoint

    log = tmp_path / "env.log"
    log.write_text("initialization failed\n")
    daemon = SimpleNamespace(log_path=str(log), name="test_env", poll=lambda: 2)
    with pytest.raises(RuntimeError, match="exited with code 2"):
        _wait_for_server_endpoint(daemon)


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


class _FakeModel:
    def predict(self, observation: dict[str, Any], *, options: dict[str, Any]) -> Any:
        assert observation["task_descriptions"] == "turn on the radio"
        assert options == {"mode": "eval"}
        return np.zeros((32, 23), dtype=np.float32)


class _FakeChunkEnv:
    total_env_steps = 0
    official_success_latched = False
    official_success_receipt = None

    def __init__(self) -> None:
        self.return_all_frames: list[bool] = []

    def chunk_step(
        self,
        actions: Any,
        *,
        return_all_frames: bool = False,
    ) -> tuple[Any, float, bool, bool, dict[str, Any]]:
        array = np.asarray(actions)
        self.return_all_frames.append(bool(return_all_frames))
        self.total_env_steps += int(array.shape[0])
        frames = [
            {
                "main_images": np.full((16, 16, 3), idx, dtype=np.uint8),
                "task_descriptions": "turn on the radio",
            }
            for idx in range(int(array.shape[0]))
        ]
        obs: Any = frames if return_all_frames else frames[-1]
        return (
            obs,
            0.0,
            False,
            False,
            {
                "executed_steps": int(array.shape[0]),
                "_rpent": {"total_env_steps": self.total_env_steps},
            },
        )


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

    def reset_raw(self, *, env_idx: int = 0) -> tuple[dict[str, Any], dict[str, Any]]:
        assert env_idx == 0
        return self._obs(), {"done": {"success": False}}

    def step_raw(
        self,
        action: Any,
        *,
        env_idx: int = 0,
    ) -> tuple[dict[str, Any], float, bool, bool, dict[str, Any]]:
        assert env_idx == 0
        action_array = np.asarray(action, dtype=np.float32)
        self.actions.append(action_array.copy())
        step_index = len(self.actions)
        terminated = self.terminated_on_step == step_index
        success = self.success_on_step == step_index
        return self._obs(), 0.0, terminated, False, {"done": {"success": success}}

    def close(self) -> None:
        self.closed = True


class _ThreadRecordingBehaviorEnvFacade(BehaviorEnvFacade):
    def __init__(self) -> None:
        super().__init__(backend=object(), meta={"task_language": "test"})
        self.serve_thread_id: int | None = None
        self.business_thread_id: int | None = None

    def serve(self, **kwargs: Any) -> None:
        self.serve_thread_id = threading.get_ident()
        super().serve(**kwargs)

    def get_env_meta(self) -> dict[str, Any]:
        self.business_thread_id = threading.get_ident()
        return super().get_env_meta()


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


def _port_accepts_connections(port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
        probe.settimeout(0.1)
        return probe.connect_ex(("127.0.0.1", port)) == 0


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


@pytest.mark.parametrize(
    ("extra_args", "message"),
    [
        (
            [
                "--behavior-mode",
                "explore",
                "--explore-attempts-per-session",
                "1",
            ],
            "BEHAVIOR explore runs one attempt per session; use --explore-sessions",
        ),
        ([], "BEHAVIOR --explore requires --behavior-mode explore"),
        (
            ["--behavior-mode", "explore", "--dashboard"],
            "BEHAVIOR --explore is CLI-only",
        ),
        (
            ["--behavior-mode", "explore", "--env-endpoint", "127.0.0.1:1"],
            "BEHAVIOR explore requires an owned env sidecar; omit --env-endpoint",
        ),
    ],
)
def test_behavior_explore_rejects_incompatible_options(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    extra_args: list[str],
    message: str,
) -> None:
    from rpent.cli import main as cli

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "rpent",
            "--robot",
            "behavior",
            "--task-name",
            "turning_on_radio",
            "--public-seed",
            "0",
            "--explore",
            *extra_args,
        ],
    )

    with pytest.raises(SystemExit) as exc_info:
        cli.main()

    assert exc_info.value.code == 2
    assert message in capsys.readouterr().err


def test_behavior_toolkit_factory_maps_shared_modes(tmp_path: Path) -> None:
    config = RunConfig(
        recipe_tag="turning_on_radio_s0",
        output_dir=tmp_path / "run",
        prompt_vars={
            "behavior_mode": "eval",
            "memory_dir": str(tmp_path / "memory"),
            "task_name": "turning_on_radio",
            "public_seed": 0,
            "max_episode_steps": 64,
        },
        task_desc={},
    )
    dashboard_events = NullDashboardEventSink()

    evaluation = get_toolkit(
        primitives_kwargs={},
        dashboard_events=dashboard_events,
        config=config,
        mode="evaluation",
    )
    exploration = get_toolkit(
        primitives_kwargs={"output_dir": tmp_path / "session"},
        dashboard_events=dashboard_events,
        config=config,
        mode="exploration",
    )
    config.prompt_vars["behavior_mode"] = "explore"
    inherited = get_toolkit(
        primitives_kwargs={},
        dashboard_events=dashboard_events,
        config=config,
    )

    assert evaluation.primitives.behavior_phase == "eval"
    assert exploration.primitives.behavior_phase == "explore"
    assert inherited.primitives.behavior_phase == "explore"
    assert evaluation.memory._memory_access == "read_only"
    assert exploration.memory._memory_access == "inbox_write"
    assert exploration.primitives.output_dir == tmp_path / "session"
    with pytest.raises(ValueError, match="one attempt per session"):
        get_toolkit(
            primitives_kwargs={},
            dashboard_events=dashboard_events,
            config=config,
            mode="exploration",
            attempts_per_session=1,
        )
    with pytest.raises(ValueError, match="unsupported BEHAVIOR toolkit mode"):
        get_toolkit(
            primitives_kwargs={},
            dashboard_events=dashboard_events,
            config=config,
            mode="unsupported",
        )


def test_behavior_external_sidecar_python_gets_source_checkout(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from robots.behavior import runtime

    monkeypatch.setenv("PYTHONPATH", "/tmp/existing-pythonpath")

    env = runtime._behavior_subprocess_env(
        cuda_device="2",
        ROBOT_PLATFORM="BEHAVIOR",
    )

    assert env["ROBOT_PLATFORM"] == "BEHAVIOR"
    assert env["CUDA_VISIBLE_DEVICES"] == "2"
    assert env["PYTHONPATH"].split(os.pathsep)[:2] == [
        str(runtime.get_repo_root()),
        "/tmp/existing-pythonpath",
    ]


def test_behavior_facades_use_default_healthz_and_registered_metadata() -> None:
    facade = BehaviorEnvFacade(backend=object(), meta={"task_language": "test"})
    dino = BehaviorDinoFacade(
        encoder=object(),
        meta={"runtime": "behavior_dino", "dimension": DINOV2_DIMENSION},
    )

    assert facade._dispatch("healthz", (), {}) == {"status": "ok"}
    assert facade._dispatch("env.get_env_meta", (), {}) == {"task_language": "test"}
    assert "env.close_gripper" in facade._rpc
    assert "env.open_gripper" in facade._rpc
    assert "env.close" not in facade._rpc
    assert "env.open" not in facade._rpc
    with pytest.raises(ValueError, match="requires primitive arguments"):
        facade.close_gripper()
    assert dino._dispatch("healthz", (), {}) == {"status": "ok"}
    dino_meta = dino._dispatch("dino.get_meta", (), {})
    assert dino_meta["runtime"] == "behavior_dino"
    assert dino_meta["dimension"] == DINOV2_DIMENSION
    assert isinstance(dino_meta["pid"], int)


def test_behavior_env_facade_serve_dispatches_business_calls_on_serving_thread() -> (
    None
):
    facade = _ThreadRecordingBehaviorEnvFacade()
    port = pick_free_port()
    thread = threading.Thread(
        target=facade.serve,
        kwargs={
            "transport": "http",
            "host": "127.0.0.1",
            "port": port,
        },
        daemon=True,
    )
    thread.start()
    client = HttpRpcClient(f"http://127.0.0.1:{port}")

    try:
        deadline = time.monotonic() + 3.0
        while True:
            try:
                assert client.call("healthz", timeout_s=0.5) == {"status": "ok"}
                break
            except Exception:
                if time.monotonic() >= deadline:
                    raise
                time.sleep(0.01)

        assert client.call("env.get_env_meta", timeout_s=1.0) == {
            "task_language": "test"
        }
        assert facade.business_thread_id == facade.serve_thread_id
        assert facade.business_thread_id != threading.get_ident()
        assert client.call("shutdown", timeout_s=1.0) == {"ok": True}
    finally:
        client.close()
        facade._shutdown_event.set()
        thread.join(timeout=3.0)

    assert not thread.is_alive()
    assert facade._closed
    assert not _port_accepts_connections(port)


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


def test_finish_writes_terminal_receipt(tmp_path: Path) -> None:
    output_dir = tmp_path / "run"
    toolkit = BehaviorToolkit(
        primitives_kwargs={
            "task_name": "turning_on_radio",
            "output_dir": output_dir,
        },
        dashboard_events=NullDashboardEventSink(),
        memory=MemoryManager(tmp_path / "memory"),
    )

    result = toolkit.execute_tool(
        "finish", {"status": "incomplete", "summary": "bounded test"}
    )
    receipt = json.loads((output_dir / "terminal_receipt.json").read_text())

    assert result.is_finish is True
    assert receipt["_finish"] is True
    assert receipt["kind"] == "behavior_finish_terminal_receipt"
    assert receipt["planner_status"] == "incomplete"
    assert receipt["summary"] == "bounded test"


def test_receipt_is_session_artifact_and_recipe_is_run_artifact(
    tmp_path: Path,
) -> None:
    run_dir = tmp_path / "run"
    session_dir = run_dir / "sessions" / "session_001"
    toolkit = BehaviorToolkit(
        primitives_kwargs={
            "task_name": "turning_on_radio",
            "output_dir": run_dir,
        },
        dashboard_events=NullDashboardEventSink(),
        memory=MemoryManager(tmp_path / "memory"),
        config=RunConfig(
            recipe_tag="turning_on_radio_s0",
            output_dir=run_dir,
            prompt_vars={
                "task_name": "turning_on_radio",
                "public_seed": 0,
                "memory_dir": str(tmp_path / "memory"),
            },
            task_desc={},
        ),
        state_output_dir=session_dir,
    )
    run_audit = run_dir / "turning_on_radio_s0.json"
    run_audit.parent.mkdir(parents=True, exist_ok=True)
    run_audit.write_text('{"audit": true}\n')

    toolkit.execute_tool("finish", {"status": "incomplete", "summary": "done"})
    assert (session_dir / "terminal_receipt.json").is_file()
    assert json.loads((session_dir / "states.json").read_text())["run_artifacts"] == [
        "terminal_receipt.json"
    ]
    assert run_audit.read_text() == '{"audit": true}\n'

    toolkit.primitives._official_success_latched = True
    with toolkit.state.record_step(
        state={"task_success": True},
        terminated=True,
        command={"action": "future_stateful_command", "arg": 1},
        result={"ok": True},
        elapsed_s=0.1,
    ):
        pass
    with toolkit.state.record_step(
        state={"task_success": True},
        terminated=True,
        command={"action": "bad_command"},
        result={"error": "failed"},
        elapsed_s=0.1,
    ):
        pass

    recipe_path = Path(toolkit.write_recipe("turning_on_radio_s0") or "")
    assert recipe_path == run_dir / "recipe_turning_on_radio_s0.jsonl"
    assert not (session_dir / "recipe_turning_on_radio_s0.jsonl").exists()
    commands = [
        json.loads(line)
        for line in recipe_path.read_text().splitlines()
        if line.strip()
    ]
    assert commands == [{"action": "future_stateful_command", "arg": 1}]
    assert "command" not in commands[0]


def test_unsolved_behavior_session_does_not_write_recipe(tmp_path: Path) -> None:
    toolkit = BehaviorToolkit(
        primitives_kwargs={
            "task_name": "turning_on_radio",
            "output_dir": tmp_path / "run",
        },
        dashboard_events=NullDashboardEventSink(),
        memory=MemoryManager(tmp_path / "memory"),
    )
    with toolkit.state.record_step(
        state={"task_success": False},
        command={"action": "pi0_nav_pick", "instruction": "turn on the radio"},
        result={"ok": True},
        elapsed_s=0.1,
    ):
        pass

    assert toolkit.write_recipe("turning_on_radio_s0") is None
    assert not (tmp_path / "run" / "recipe_turning_on_radio_s0.jsonl").exists()


def test_behavior_pi0_chunk_records_streaming_episode_video(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    session_dir = run_dir / "sessions" / "session_001"
    toolkit = BehaviorToolkit(
        primitives_kwargs={
            "env": _FakeChunkEnv(),
            "model": _FakeModel(),
            "task_name": "turning_on_radio",
            "public_seed": 0,
            "max_episode_steps": 64,
            "initial_observation": {
                "main_images": np.zeros((16, 16, 3), dtype=np.uint8),
                "task_descriptions": "turn on the radio",
            },
        },
        dashboard_events=NullDashboardEventSink(),
        memory=MemoryManager(tmp_path / "memory"),
        config=RunConfig(
            recipe_tag="turning_on_radio_s0",
            output_dir=run_dir,
            prompt_vars={
                "behavior_mode": "explore",
                "task_name": "turning_on_radio",
                "public_seed": 0,
                "memory_dir": str(tmp_path / "memory"),
            },
            task_desc={},
        ),
        state_output_dir=session_dir,
    )

    result = toolkit.primitives.pi0_nav_pick(
        instruction="turn on the radio",
        chunks=1,
    )
    toolkit.execute_tool("finish", {"status": "incomplete", "summary": "done"})

    assert result["chunks_used"] == 1
    assert result["env_steps_used"] == 32
    assert toolkit.primitives.env.return_all_frames == [True]
    video_path = session_dir / "episode.mp4"
    assert video_path.is_file()
    assert video_path.stat().st_size > 0
    assert (
        "episode.mp4"
        in json.loads((session_dir / "states.json").read_text())["run_artifacts"]
    )


def test_behavior_build_memory_cli_wraps_existing_catalog_compiler(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    captured: dict[str, Any] = {}

    def fake_compile_runtime_catalog(**kwargs: Any) -> dict[str, Any]:
        captured.update(kwargs)
        return {"artifact_dir": str(kwargs["output_dir"]), "preliminary": True}

    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "7")
    monkeypatch.setattr(
        build_memory_cli,
        "compile_runtime_catalog",
        fake_compile_runtime_catalog,
    )

    rc = build_memory_cli.main(
        [
            "--selection-manifest",
            str(tmp_path / "selection.json"),
            "--output-dir",
            str(tmp_path / "catalog"),
            "--video-root",
            str(tmp_path / "videos"),
            "--rollups-dir",
            str(tmp_path / "rollups"),
            "--source-archive",
            str(tmp_path / "dinov2.tar.gz"),
            "--weights",
            str(tmp_path / "dinov2.pth"),
            "--cache-dir",
            str(tmp_path / "cache"),
            "--cuda-device",
            "2",
            "--batch-size",
            "8",
        ]
    )

    assert rc == 0
    assert json.loads(capsys.readouterr().out) == {
        "artifact_dir": str((tmp_path / "catalog").resolve()),
        "preliminary": True,
    }
    assert captured["selection_manifest"] == (tmp_path / "selection.json").resolve()
    assert captured["output_dir"] == (tmp_path / "catalog").resolve()
    assert captured["video_roots"] == ((tmp_path / "videos").resolve(),)
    assert captured["rollups_dir"] == (tmp_path / "rollups").resolve()
    assert captured["source_archive"] == (tmp_path / "dinov2.tar.gz").resolve()
    assert captured["weights"] == (tmp_path / "dinov2.pth").resolve()
    assert captured["cache_dir"] == (tmp_path / "cache").resolve()
    assert captured["batch_size"] == 8
    assert os.environ["CUDA_VISIBLE_DEVICES"] == "2"


def test_behavior_build_memory_cli_rejects_multi_cuda_device() -> None:
    with pytest.raises(SystemExit) as exc_info:
        build_memory_cli.main(
            [
                "--selection-manifest",
                "selection.json",
                "--output-dir",
                "catalog",
                "--video-root",
                "videos",
                "--rollups-dir",
                "rollups",
                "--source-archive",
                "dinov2.tar.gz",
                "--weights",
                "dinov2.pth",
                "--cuda-device",
                "2,7",
            ]
        )
    assert exc_info.value.code == 2
