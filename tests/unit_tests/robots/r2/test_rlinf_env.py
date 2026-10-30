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
"""Offline tests for the R2 adapter over a fake RLinf connection."""

from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest

from robots.r2.rlinf_env import R2AgentEnv
from robots.r2.runtime_config import BaseLimits, CameraCalibration, load_site_config

#: TCP = arm joints[:3] + offset, rpy = joints[3:6]: a stand-in for R2's FK.
_TCP_OFFSET = {"left": np.array([0.4, 0.2, 0.9]), "right": np.array([0.4, -0.2, 0.9])}


class FakeHealth:
    def __init__(self, healthy: bool = True) -> None:
        self.healthy = healthy

    def to_dict(self):
        return {
            "healthy": self.healthy,
            "error_message": "" if self.healthy else "e-stop",
        }


class FakeConnection:
    camera_names = ("head",)
    body_dof = 21
    end_effector_type = "gripper"
    enable_depth = True

    def __init__(self) -> None:
        self.joints = {
            "waist": np.zeros(5),
            "neck": np.zeros(2),
            "left_arm": np.zeros(7),
            "right_arm": np.zeros(7),
        }
        self.grippers = {"left": 1.0, "right": 1.0}
        self.moves: list[dict] = []
        self.complete = True
        self.joint_group_specs = {
            "neck": SimpleNamespace(
                low=np.array([-1.5, -0.7]), high=np.array([1.5, 0.35])
            )
        }

    def get_state(self):
        return SimpleNamespace(
            waist_positions=self.joints["waist"].copy(),
            neck_positions=self.joints["neck"].copy(),
            left_joint_positions=self.joints["left_arm"].copy(),
            right_joint_positions=self.joints["right_arm"].copy(),
            left_gripper_position=np.array([self.grippers["left"]]),
            right_gripper_position=np.array([self.grippers["right"]]),
            base_pose=np.zeros(3),
        )

    def forward_kinematics(self, q):
        q = np.asarray(q)
        return {
            f"{arm}_tcp_pose": np.concatenate(
                [joints[:3] + _TCP_OFFSET[arm], joints[3:6]]
            )
            for arm, joints in (("left", q[5:12]), ("right", q[12:19]))
        }

    def inverse_kinematics(self, *, left_tcp_pose=None, right_tcp_pose=None):
        solved = {}
        for arm, pose in (("left", left_tcp_pose), ("right", right_tcp_pose)):
            q = self.joints[f"{arm}_arm"].copy()
            if pose is not None:
                if pose[2] > 1.15:
                    raise ValueError(f"R2 {arm} arm has no IK solution")
                q[:3] = pose[:3] - _TCP_OFFSET[arm]
                q[3:6] = pose[3:]
            solved[f"{arm}_joint_positions"] = q
        return solved

    def move_body_smooth(self, **kwargs):
        self.moves.append(kwargs)
        for group in ("left_arm", "right_arm", "neck"):
            if group in kwargs:
                self.joints[group] = np.asarray(kwargs[group], dtype=float)
        for arm in ("left", "right"):
            if f"{arm}_gripper" in kwargs:
                self.grippers[arm] = kwargs[f"{arm}_gripper"]
        return self.complete

    def diagnose(self):
        return FakeHealth(self.complete)

    def check_camera(self, name, timeout):
        return True

    def get_camera(self, name):
        return np.full((4, 6, 3), 7, dtype=np.uint8)

    def get_camera_depth(self, name):
        return np.full((4, 6), 0.812, dtype=np.float32)

    def move_base_pulse(self, velocity, duration):
        self.moves.append({"base": list(velocity), "duration": duration})


class FakeRlinfEnv:
    def __init__(self) -> None:
        self.resets = 0
        self.closed = 0

    def reset(self):
        self.resets += 1
        return {}, {}

    def go_to_rest(self):
        raise RuntimeError("R2 did not reach its rest pose")

    def close(self):
        self.closed += 1


def _env(site=None, urdf=None):
    # The example's URDF lives on the robot; tests use their own or none.
    site = replace(
        site or load_site_config(),
        urdf=None if urdf is None else str(urdf),
        cameras={"head": CameraCalibration(frame="cam", intrinsics=None)},
    )
    connection = FakeConnection()
    rlinf_env = FakeRlinfEnv()
    env = R2AgentEnv(
        site,
        env=rlinf_env,
        connection=connection,
        joint_names={
            "waist": [f"J_W{i}" for i in range(5)],
            "neck": ["J_NeckYaw", "J_NeckPitch"],
            "left_arm": [f"J_L{i}" for i in range(7)],
            "right_arm": [f"J_R{i}" for i in range(7)],
        },
    )
    return env, connection, rlinf_env


def test_move_to_refuses_targets_outside_the_workspace_or_too_far_without_moving():
    env, connection, _ = _env()
    connection.joints["left_arm"][0] = 0.2

    beyond = env.move_to("left", [0.8, 0.2, 0.9])
    far = env.move_to("left", [0.6, 0.2, 1.25])

    assert beyond["ok"] is False and "outside the left workspace" in beyond["error"]
    assert far["ok"] is False and "exceeds max_move_m" in far["error"]
    assert connection.moves == []


def test_relative_moves_compose_with_the_measured_tcp():
    env, connection, _ = _env()

    moved = env.move_delta("left", [0.05, 0.0, -0.02])
    rotated = env.rotate_delta("left", [0.0, 0.0, 0.3])

    assert moved["ok"] is True
    np.testing.assert_allclose(moved["final_tcp"][:3], [0.45, 0.2, 0.88], atol=1e-4)
    np.testing.assert_allclose(
        rotated["final_tcp"], [0.45, 0.2, 0.88, 0, 0, 0.3], atol=1e-4
    )
    assert rotated["rotation_error_rad"] == pytest.approx(0.0, abs=1e-4)
    assert all(move["max_velocity"] == 0.5 for move in connection.moves)
    assert env.move_delta("left", [0.2, 0.0, 0.0])["ok"] is False
    assert env.rotate_delta("left", [0.0, 0.0, 1.0])["ok"] is False


def test_unreachable_and_incomplete_moves_report_why():
    env, connection, _ = _env()

    unreachable = env.move_to("left", [0.4, 0.2, 1.16], [0.0, 0.0, 0.0])
    connection.complete = False
    stopped = env.move_delta("right", [0.0, 0.0, 0.05])

    assert unreachable["ok"] is False and "no IK solution" in unreachable["error"]
    assert stopped["ok"] is False and stopped["health"]["healthy"] is False


def test_gripper_head_and_home_validate_and_report_measured_values():
    env, connection, _ = _env()

    closed = env.set_gripper("right", 0.0)
    head = env.look(0.2, -0.5)
    too_low = env.look(0.0, -1.0)
    home = env.go_home()

    assert closed == {
        "ok": True,
        "arm": "right",
        "target_opening": 0.0,
        "measured_opening": 0.0,
    }
    assert head["neck"] == [0.2, -0.5]
    assert too_low["ok"] is False and "outside the limits" in too_low["error"]
    assert home["ok"] is False and "rest pose" in home["error"]
    with pytest.raises(ValueError, match="opening must be within"):
        env.set_gripper("left", 1.5)


def test_base_motion_needs_the_site_to_allow_it():
    env, connection, _ = _env()
    assert env.move_base(0.1, 0.0, 0.0, 1.0)["ok"] is False

    site = replace(
        load_site_config(),
        base=BaseLimits(
            allow_motion=True,
            max_linear_speed=0.2,
            max_angular_speed=0.3,
            max_duration=2.0,
        ),
    )
    env, connection, _ = _env(site)

    assert env.move_base(0.3, 0.0, 0.0, 1.0)["ok"] is False
    assert env.move_base(0.1, 0.0, 0.1, 1.0)["ok"] is True
    assert connection.moves == [{"base": [0.1, 0.0, 0.1], "duration": 1.0}]
    assert env.get_env_meta()["base_motion"] is True


def test_observe_returns_millimetre_depth_and_urdf_camera_poses(tiny_urdf):
    env, connection, _ = _env(urdf=tiny_urdf)
    connection.joints["neck"] = np.array([np.pi / 2, 0.0])

    observation = env.observe()

    head = observation["views"]["head"]
    assert head["rgb"].shape == (4, 6, 3)
    assert head["depth_mm"].dtype == np.uint16 and int(head["depth_mm"][0, 0]) == 812
    meta = observation["camera_meta"]["cameras"]["head"]
    assert meta["image_shape"] == [4, 6] and meta["depth_shape"] == [4, 6]
    np.testing.assert_allclose(
        np.asarray(meta["T_base_camera"])[:3, 3], [0, 0.1, 1], atol=1e-6
    )
    state = observation["robot_state"]
    assert state["arms"]["left"]["tcp_xyz"] == [0.4, 0.2, 0.9]
    assert state["neck"]["yaw"] == pytest.approx(np.pi / 2, abs=1e-4)


def test_reset_starts_from_rest_and_close_is_idempotent():
    env, _, rlinf_env = _env()

    assert env.reset()["ok"] is True
    env.close()
    env.close()

    assert (rlinf_env.resets, rlinf_env.closed) == (1, 1)
