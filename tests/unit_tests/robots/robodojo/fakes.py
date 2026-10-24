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

"""Shared native environment, model metadata and deterministic policy test doubles."""

import argparse
from types import SimpleNamespace

import numpy as np

from robots.robodojo import robot_spec
from robots.robodojo.awomo05_contract import (
    CAMERAS,
    POLICY_METHODS,
    SESSION_METHODS,
    STATE_FIELDS,
)
from robots.robodojo.env_client import RoboDojoEnvClient


def metadata():
    return {
        "policy_name": "Awomo05",
        "env_cfg_type": "arx_x5",
        "action_type": "joint",
        "action_dim": 14,
        "action_layout": "native_joint_dict",
        "camera_order": list(CAMERAS),
        "input_color_order": "bgr",
        "control_hz": 25,
        "history_enabled": True,
        "action_horizon": 16,
        "replan_steps": 8,
        "methods": list(POLICY_METHODS),
        "stateful": True,
        "rpc_methods": list(SESSION_METHODS),
    }


def args(*flags):
    parser = argparse.ArgumentParser()
    robot_spec._add_cli_args(parser, True)
    return parser.parse_args(flags)


class FakeEnv(RoboDojoEnvClient):
    def __init__(self, limit=20, done_at=None):
        self.frame = 0
        self.limit = limit
        self.done_at = done_at
        self.closed = False
        self.after_step = lambda: None
        super().__init__(SimpleNamespace(call=self._rpc), expected_meta={})

    def get_obs(self):
        return {
            "instruction": self.get_task_language(),
            "state": {
                "frame": self.frame,
                **{
                    key: np.full(size, self.frame, dtype=float)
                    for key, size in STATE_FIELDS.items()
                },
            },
            "vision": {
                name: {
                    "color": np.full((2, 2, 3), self.frame % 256, np.uint8),
                    "depth": np.ones((2, 2)),
                    "intrinsic_matrix": np.eye(3),
                    "extrinsic_matrix": np.eye(4),
                }
                for name in ("cam_head", "cam_left_wrist", "cam_right_wrist")
            },
        }

    def get_task_language(self):
        return "stack the blocks"

    def get_status(self):
        return {"step": self.frame, "step_limit": self.limit}

    def _native_step(self, action, *, expected_step=None):
        before = self.frame
        if expected_step is not None:
            assert expected_step == before
        if "expected_step" in action:
            assert action["expected_step"] == self.frame + 1
        if "left_arm_joint_state" in action:
            assert action["left_arm_joint_state"][0] == self.frame + 1
        self.frame += 1
        self.after_step()
        return (
            self.get_obs(),
            0.0,
            self.frame == self.done_at,
            {
                "status": self.get_status(),
                "step_before": before,
                "step_after": self.frame,
                "executed_actions": 1,
                "terminated": self.frame == self.done_at,
                "truncated": self.frame >= self.limit,
            },
        )

    def _native_reset(self):
        self.frame = 0
        return self.get_obs()

    def get_motion_state(self):
        return {f"{arm}_gripper": 0.5 for arm in ("left", "right")}

    def _rpc(self, method, *, kwargs=None, **options):
        kwargs = kwargs or {}
        if method == "env.get_env_meta":
            return {}
        if method == "env.reset":
            return self._native_reset()
        if method == "env.close":
            self.closed = True
            return None
        if method == "env.step":
            return self._native_step(
                kwargs["flat_action"], expected_step=kwargs.get("expected_step")
            )
        if method in ("env.step_joint_update", "env.hold_step"):
            obs, _, _, info = self._native_step(
                {}, expected_step=kwargs.get("expected_step")
            )
            return {**info, "episode_status": info["status"], "observation": obs}
        raise AssertionError(method)


class EpisodePolicy:
    """Fixed eight-action chunks; counts in direct execution tests use this fixture."""

    def __init__(self):
        self.events = []
        self.step = None
        self.last_action_info = {}

    def reset_session(self):
        self.events.append(("reset",))

    def update_obs(self, obs, step):
        assert obs["instruction"] == "stack the blocks"
        assert obs["state"]["frame"] == step
        self.step = step
        self.events.append(("observe", step))

    def get_action(self):
        self.events.append(("infer", self.step))
        self.last_action_info = {"upstream": {"latency_ms": 123.5}}
        return [{"expected_step": self.step + i} for i in range(1, 9)]

    def end_episode(self):
        self.events.append(("end",))
