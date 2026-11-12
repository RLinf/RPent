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

"""Shared task arguments, saved camera frames and local Env RPC fixtures."""

from __future__ import annotations

import argparse
from copy import deepcopy
from pathlib import Path
from typing import Any

import numpy as np

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
    def predict(self, observation: dict[str, Any]) -> np.ndarray:
        return np.zeros((64, 23), dtype=np.float32)


class _SAM:
    def __init__(self):
        self.calls = []

    def segment(self, image, **kwargs):
        self.calls.append((image, kwargs))
        mask = np.zeros((224, 224), dtype=bool)
        mask[112, 112] = True
        return Sam3Result(found=True, mask=mask, mask_shape=mask.shape, score=0.9)


def _template_toolkit(tmp_path, allowed, *, env=None, sam=None, **options):
    if env is None:
        env = _GeometryEnvironment(_geometry_observation())
    return BehaviorToolkit(
        runtime_kwargs={
            "env": env,
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

    def session_status(self):
        return {"ended": None, "reason": None, "outcome": "unknown"}

    def get_motion_snapshot(self):
        raise RuntimeError("motion_history_unavailable: fixture has no motion history")

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

    def predict(self, observation):
        self.calls += 1
        actions = np.zeros((2, 23), dtype=np.float32)
        actions[:, 14] = [-0.2, -0.6]
        actions[:, 22] = [0.3, 0.6]
        return actions
