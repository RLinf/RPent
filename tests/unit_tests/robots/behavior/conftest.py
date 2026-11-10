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

"""Helpers shared by the BEHAVIOR unit-test modules."""

from __future__ import annotations

import argparse
from copy import deepcopy
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from robots.behavior.env_client import BehaviorEnvClient
from robots.behavior.env_server import BehaviorEnvFacade
from robots.behavior.primitives import BehaviorPrimitives
from robots.behavior.robot_spec import get_robot_spec
from robots.behavior.toolkit import BehaviorToolkit
from rpent.dashboard.events import NullDashboardEventSink
from rpent.memory import MemoryManager
from rpent.robots.components.sam3_client import Sam3Result


def _args(tmp_path: Path, *extra: str) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default=str(tmp_path / "run"))
    parser.add_argument("--memory-dir", default=None)
    parser.add_argument("--memory-profile", default=None)
    get_robot_spec().add_cli_args(parser, False)
    return parser.parse_args(
        [
            "--task-id",
            "40",
            "--task-language",
            "make popcorn",
            "--execution-horizon",
            "2",
            "--env-endpoint",
            "http://localhost:21000",
            "--vla-endpoint",
            "http://localhost:21001",
            *extra,
        ]
    )


_GEOMETRY_CAMERAS = (
    "robot_r1::robot_r1:zed_link:Camera:0::rgb",
    "robot_r1::robot_r1:left_realsense_link:Camera:0::rgb",
    "robot_r1::robot_r1:right_realsense_link:Camera:0::rgb",
)


def _geometry_observation(
    offset: float = 0, depth: float = 2, size: int = 224
) -> dict[str, Any]:
    raw: dict[str, Any] = {
        camera: np.zeros((1, size, size, 3), dtype=np.uint8)
        for camera in _GEOMETRY_CAMERAS
    }
    raw.update(
        {
            camera.replace("::rgb", "::depth_linear"): np.full(
                (1, size, size, 1), depth, dtype=np.float32
            )
            for camera in _GEOMETRY_CAMERAS
        }
    )
    raw["robot_r1::proprio"] = np.zeros((1, 256), dtype=np.float32)
    # Public evaluator order is left wrist, right wrist, head.
    raw["robot_r1::cam_rel_poses"] = np.array(
        [[x + offset, 0, 0, 0, 0, 0, 1] for x in (1, 2, 3)], dtype=np.float32
    ).reshape(1, 21)
    raw["task_id"] = np.array([40])
    return raw


class _GeometryEnvironment:
    def __init__(self, raw: dict[str, Any]) -> None:
        self.raw = raw
        self.reads = 0
        self.actions = 0

    def get_observation(self) -> dict[str, Any]:
        self.reads += 1
        return self.raw

    def chunk_step(self, actions: np.ndarray) -> tuple:
        self.actions += len(actions)
        self.raw = _geometry_observation(offset=10, depth=4)
        return self.raw, None, None, None, {"executed_actions": len(actions)}


class _GeometryVLA:
    def infer(self, observation: dict[str, Any], prompt: str) -> np.ndarray:
        return np.zeros((64, 23), dtype=np.float32)


class _SAM:
    def __init__(self):
        self.calls = []

    def segment(self, image, **kwargs):
        self.calls.append((image, kwargs))
        mask = np.zeros((224, 224), dtype=bool)
        mask[112, 112] = True
        return Sam3Result(found=True, mask=mask, mask_shape=mask.shape, score=0.9)


def _template_toolkit(tmp_path, allowed, *, sam=None, **options):
    return BehaviorToolkit(
        runtime_kwargs={
            "env": _GeometryEnvironment(_geometry_observation()),
            "model": _GeometryVLA(),
            "sam3_client": sam,
            "task_id": 40,
            "task_language": "make popcorn",
            "execution_horizon": 32,
            "instruction_mode": "main",
            "max_chunks_per_call": 1,
            "tool_allowlist": allowed,
            **options,
        },
        dashboard_events=NullDashboardEventSink(),
        memory=MemoryManager(tmp_path / "memory"),
        state_output_dir=tmp_path / "state",
    )


class _MotionEnvironment:
    def __init__(self):
        proprio = np.zeros(61)
        proprio[3:10] = np.arange(7) * 0.1
        proprio[28:35] = np.arange(7) * -0.1
        proprio[17:20] = [0.4, 0.2, 0.8]
        proprio[42:45] = [0.4, -0.2, 0.8]
        proprio[[23, 48]] = 1
        proprio[24:26] = proprio[49:51] = 0.05
        proprio[53:57] = [0.1, -0.2, 0.3, 0.4]
        self.raw = {"r1::proprio": proprio}
        for camera in ("zed_link", "left_realsense_link", "right_realsense_link"):
            self.raw[f"r1::{camera}:Camera:0::rgb"] = np.zeros((4, 5, 3), np.uint8)
        self.actions = []
        self.ended = False
        self.feedback = None
        self.invalid_after = False
        self.fail = False
        self.generation = 0
        self.observation_index = 0
        self.previous_proprio = None
        self.issued = 0
        self.confirmed = 0
        self.received_tokens = []

    def get_observation(self):
        return deepcopy(self.raw)

    def session_status(self):
        return {"ended": self.ended}

    def get_motion_snapshot(self):
        token = {
            "generation": self.generation,
            "observation_index": self.observation_index,
        }
        return {
            "observation": self.get_observation(),
            "observation_token": token,
            "motion_context": {
                "source": "confirmed_public_control_observations",
                "control_dt_s": 1 / 30,
                "current_token": token,
                "previous_token": {
                    **token,
                    "observation_index": self.observation_index - 1,
                }
                if self.previous_proprio is not None
                else None,
                "current_proprio": self.raw["r1::proprio"].copy(),
                "previous_proprio": deepcopy(self.previous_proprio),
                "pair_confirmed": self.previous_proprio is not None,
                "initial_bootstrap": self.observation_index == 0,
                "issued_actions": self.issued,
                "confirmed_actions": self.confirmed,
                "unconfirmed_actions": self.issued - self.confirmed,
                "actions_pending": False,
            },
        }

    def chunk_step(
        self, actions, *, return_all_frames=False, expected_observation_token=None
    ):
        if expected_observation_token is not None:
            assert (
                expected_observation_token
                == self.get_motion_snapshot()["observation_token"]
            )
        self.received_tokens.append(deepcopy(expected_observation_token))
        self.actions.append(actions.copy())
        if self.fail:
            raise TimeoutError("execution outcome unknown")
        info = (
            {
                "issued_actions": len(actions),
                "confirmed_actions": len(actions),
                "unconfirmed_actions": 0,
                "executed_actions": len(actions),
                "observation_after_last_issued_action": True,
                "stop_reason": "chunk_completed",
            }
            if self.feedback is None
            else dict(self.feedback)
        )
        issued, confirmed = info.get("issued_actions"), info.get("confirmed_actions")
        if type(issued) is int and 0 <= issued <= len(actions):
            self.issued += issued
        if type(confirmed) is int and 0 <= confirmed <= len(actions):
            for action in actions[:confirmed]:
                self.previous_proprio = self.raw["r1::proprio"].copy()
                self.raw["r1::proprio"][3:10] = action[7:14]
                self.raw["r1::proprio"][28:35] = action[15:22]
                self.observation_index += 1
                self.confirmed += 1
        snapshot = self.get_motion_snapshot()
        info["observation_token"] = snapshot["observation_token"]
        info["motion_context"] = snapshot["motion_context"]
        returned = self.get_observation()
        if self.invalid_after:
            returned["r1::proprio"] = np.zeros(256)
        return returned, None, None, None, info


class _MotionPlanner:
    def __init__(self):
        self.calls = []
        self.mutate = lambda result, observation: None

    def plan_arm_path(
        self,
        *,
        arm,
        target_pose,
        observation,
        motion_context,
        allow_orientation_relax=False,
    ):
        self.calls.append((arm, target_pose, observation))
        proprio = observation["r1::proprio"]
        arm_slice = slice(3, 10) if arm == "left" else slice(28, 35)
        other_slice = slice(28, 35) if arm == "left" else slice(3, 10)
        start = proprio[arm_slice]
        result = {
            "status": "Success",
            "position": start + np.arange(1, 5)[:, None] * 0.005,
            "joint_names": [f"{arm}_arm_joint{i}" for i in range(1, 8)],
            "frame": "robot_base",
            "quaternion_order": "xyzw",
            "dt_s": 1 / 30,
            "control_dt_s": 1 / 30,
            "start_arm_qpos": start.copy(),
            "held_trunk_qpos": proprio[53:57].copy(),
            "held_other_arm_qpos": proprio[other_slice].copy(),
            "motion_state_contract": {"stationary": True},
            "held_finger_qpos": {
                "left": proprio[24:26].tolist(),
                "right": proprio[49:51].tolist(),
            },
            "joint_position_limits_rad": np.tile([-3.0, 3.0], (7, 1)),
            "joint_velocity_limits_rad_s": np.full(7, 0.5),
            "fk_comparison": {
                arm: {"position_error_m": 0.001, "orientation_error_rad": 0.002}
                for arm in ("left", "right")
            },
            "collision_checked": False,
            "collision_scope": "public_depth_and_self_collision",
        }
        self.mutate(result, observation)
        return result


class _MotionRpc:
    def __init__(self, facade):
        self.facade = facade
        self.calls = []

    def call(self, method, args=(), kwargs=None, timeout_s=None):
        self.calls.append(method)
        return self.facade._dispatch(method, args, kwargs or {})


class _MotionVLA:
    def __init__(self):
        self.calls = 0

    def infer(self, observation, prompt):
        self.calls += 1
        proprio = observation["r1::proprio"]
        actions = np.zeros((2, 23), dtype=np.float32)
        actions[:, 3:7] = proprio[53:57]
        actions[:, 7:14] = proprio[3:10]
        actions[:, 15:22] = proprio[28:35]
        actions[:, [14, 22]] = 1
        return actions


@pytest.fixture
def motion():
    backend, planner, model = _MotionEnvironment(), _MotionPlanner(), _MotionVLA()
    facade = BehaviorEnvFacade(backend, task_id=40, motion_planner=planner)
    rpc = _MotionRpc(facade)
    client = BehaviorEnvClient(rpc, task_id=40)
    primitives = BehaviorPrimitives(
        env=client,
        model=model,
        task_id=40,
        task_language="make popcorn",
        check_cancelled=lambda: None,
        execution_horizon=2,
        instruction_mode="main",
    )
    return primitives, backend, planner, rpc, model


def _establish_prior_hold(motion):
    """Fixture setup: one prior confirmed hold, outside the move_to under test.

    Path tests inject feedback after this setup so those faults still address
    the native path or its optional target holds, rather than the new initial
    sampling action. Keep actual public history and command state; only reset
    the fixture's per-test call lists. No production function skips admission.
    """
    primitives, backend, _, _, _ = motion
    from robots.behavior.tools import extract_robot_state

    native, _ = primitives._current_hold_action(
        extract_robot_state(backend.raw["r1::proprio"])
    )
    primitives._send_actions(
        native.reshape(1, 23),
        expected_observation_token=backend.get_motion_snapshot()["observation_token"],
    )
    backend.actions.clear()
    backend.received_tokens.clear()


def _motion_target(backend, arm="left"):
    start = 17 if arm == "left" else 42
    return backend.raw["r1::proprio"][start : start + 3].tolist()


def _refresh_reply(backend, reply):
    _, reward, terminated, truncated, info = reply
    snapshot = backend.get_motion_snapshot()
    info.update(
        observation_token=snapshot["observation_token"],
        motion_context=snapshot["motion_context"],
    )
    return backend.get_observation(), reward, terminated, truncated, info


def _follow_planned_pose(backend, planner):
    """Let confirmed fake path execution reach its requested EEF pose."""
    step = backend.chunk_step

    def following_step(actions, **kwargs):
        reply = step(actions, **kwargs)
        if planner.calls and reply[4].get("confirmed_actions") == len(actions):
            arm, target, _ = planner.calls[-1]
            start = 17 if arm == "left" else 42
            backend.raw["r1::proprio"][start : start + 7] = target
        return _refresh_reply(backend, reply)

    backend.chunk_step = following_step


class _GripperEnvironment:
    def __init__(self):
        proprio = np.zeros(61)
        proprio[3:10] = np.arange(7) * 0.1
        proprio[28:35] = np.arange(7) * -0.1
        proprio[[23, 48]] = 1
        proprio[24:26] = [0.04, 0.04]
        proprio[49:51] = [0.01, 0.01]
        proprio[53:57] = [0.1, -0.2, 0.3, 0.4]
        self.raw = {"r1::proprio": proprio}
        for camera in ("zed_link", "left_realsense_link", "right_realsense_link"):
            self.raw[f"r1::{camera}:Camera:0::rgb"] = np.zeros((4, 5, 3), np.uint8)
        self.actions = []
        self.feedback = None
        self.ended = False
        self.terminated = None
        self.fail = False
        self.invalid_after = False
        self.post_finger_qpos = None
        self.post_finger_qvel = None

    def get_observation(self):
        return deepcopy(self.raw)

    def get_session_status(self):
        return {"ended": self.ended}

    def chunk_step(self, actions):
        self.actions.append(actions.copy())
        if self.fail:
            raise TimeoutError("execution outcome unknown")
        info = (
            {
                "issued_actions": len(actions),
                "confirmed_actions": len(actions),
                "unconfirmed_actions": 0,
                "executed_actions": len(actions),
                "observation_after_last_issued_action": True,
                "stop_reason": "chunk_completed",
            }
            if self.feedback is None
            else dict(self.feedback)
        )
        returned = self.get_observation()
        if self.post_finger_qpos is not None:
            returned["r1::proprio"][24:26] = self.post_finger_qpos
        if self.post_finger_qvel is not None:
            returned["r1::proprio"][26:28] = self.post_finger_qvel
        if self.invalid_after:
            returned["r1::proprio"] = np.zeros(256)
        return returned, None, self.terminated, None, info


class _GripperModel:
    def __init__(self):
        self.calls = 0

    def infer(self, observation, prompt):
        self.calls += 1
        actions = np.zeros((2, 23), dtype=np.float32)
        actions[:, 14] = [-0.2, -0.6]
        actions[:, 22] = [0.3, 0.6]
        return actions


@pytest.fixture
def gripper():
    env, model = _GripperEnvironment(), _GripperModel()
    primitives = BehaviorPrimitives(
        env=env,
        model=model,
        task_id=40,
        task_language="make popcorn",
        check_cancelled=lambda: None,
        execution_horizon=2,
        instruction_mode="main",
    )
    return primitives, env, model


class LocalRpc:
    """Exercise the production RPC routes without opening a network listener."""

    def __init__(self, facade):
        self.facade = facade
        self.calls = []

    def call(self, method, args=(), kwargs=None, timeout_s=None):
        self.calls.append(method)
        return self.facade._dispatch(method, args, kwargs or {})


class FakeEnvBackend:
    def __init__(self):
        self.observation = {"fixture_frame": 0}
        self.actions = []
        self.info = {"executed_actions": 0}

    def get_observation(self):
        return self.observation

    def step(self, action):
        self.actions.append(action)
        self.observation = {"fixture_frame": len(self.actions)}
        self.info = {"executed_actions": 1}
        return self.observation, None, None, None, self.info

    def chunk_step(self, actions, *, return_all_frames=False):
        frames = []
        for action in actions:
            self.step(action)
            frames.append(self.observation)
        self.info = {"executed_actions": len(actions), "backend_note": "completed"}
        return (
            frames if return_all_frames else self.observation,
            None,
            None,
            None,
            self.info,
        )
