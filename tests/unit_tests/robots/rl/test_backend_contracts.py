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

"""Native frame conversion, paired control and authoritative task outcomes."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from robots.rl.backends import RoboCasaPrimitives, RoboTwinPrimitives
from robots.rl.server import planner_command_pose
from rpent.session import EnvState


@pytest.mark.parametrize("mount_yaw", [0.02, 0.01])
def test_native_planner_command_corrects_fixed_kinematic_mount_rotation(mount_yaw):
    class Pose:
        def __init__(self, position, quaternion=None):
            self.p = np.asarray(position)
            self.rotation = (
                Rotation.identity()
                if quaternion is None
                else Rotation.from_quat(quaternion, scalar_first=True)
            )

        @property
        def q(self):
            return self.rotation.as_quat(scalar_first=True)

        def inv(self):
            rotation = self.rotation.inv()
            return Pose(-rotation.apply(self.p), rotation.as_quat(scalar_first=True))

        def __mul__(self, other):
            return Pose(
                self.p + self.rotation.apply(other.p),
                (self.rotation * other.rotation).as_quat(scalar_first=True),
            )

    origin = Pose(
        [0, -0.65, 0], Rotation.from_euler("z", np.pi / 2).as_quat(scalar_first=True)
    )
    bias = np.array([-0.23, -0.3, -0.78])
    nominal = origin * Pose(-bias)
    actual = nominal * Pose(
        [0.005, 0.002, 0],
        Rotation.from_euler("z", mount_yaw).as_quat(scalar_first=True),
    )
    target = Pose(
        [0.1, -0.2, 0.9],
        Rotation.from_euler("xyz", [0.1, 0.2, 0.3]).as_quat(scalar_first=True),
    )
    command = planner_command_pose(
        SimpleNamespace(robot_origion_pose=origin, frame_bias=bias),
        actual,
        [*target.p, *target.q],
    )
    # Native planner converts the command through its nominal base; actual FK
    # places that relative goal at the measured kinematic base.
    realised = actual * nominal.inv() * Pose(command[:3], command[3:])
    np.testing.assert_allclose(realised.p, target.p, atol=1e-12)
    np.testing.assert_allclose(
        realised.rotation.as_matrix(), target.rotation.as_matrix(), atol=1e-12
    )


class Twin:
    def __init__(self, *, failed_arm=None):
        self.failed_arm = failed_arm
        state = {"qpos_target14": np.r_[np.zeros(6), 1, np.zeros(6), 1]}
        for arm in ("left", "right"):
            state[f"{arm}_tcp_pose"] = [0, 0, 0.4, 1, 0, 0, 0]
            state[f"{arm}_eef_pose"] = [0, 0, 0.3, 1, 0, 0, 0]
            state[f"{arm}_gripper"] = 1
        self.last_info = {
            "robot_state": state,
            "episode_status": {"eval_success": False},
        }
        self.plans, self.chunks = [], []
        self._client = SimpleNamespace(
            call=lambda method: {
                side: {"opening": 1.0, "commanded_opening": 1.0}
                for side in ("left", "right")
            }
        )

    def plan_arm_path(self, arm, pose):
        self.plans.append((arm, pose))
        if arm == self.failed_arm:
            return {"status": "Fail", "position": None}
        count = 3 if arm == "left" else 5
        return {
            "status": "Success",
            "position": np.linspace(np.zeros(6), np.ones(6), count),
        }

    def chunk_step(self, actions, *, action_type):
        assert action_type == "qpos"
        self.chunks.append(actions)
        for side, pose in self.plans:
            self.last_info["robot_state"][f"{side}_tcp_pose"] = [
                *np.asarray(pose[:3]) + [0, 0, 0.1],
                *pose[3:],
            ]
        return None, None, None, None, {"executed_actions": len(actions)}


def twin_primitives(tmp_path, **kwargs):
    env = Twin(**kwargs)
    return RoboTwinPrimitives(
        env=env,
        sam3_client=None,
        state=EnvState(tmp_path),
        check_cancelled=lambda: None,
    )


@pytest.mark.parametrize("homogeneous", [False, True])
def test_native_camera_backprojection_accepts_sapien_extrinsic_shape(
    tmp_path, homogeneous
):
    primitive = twin_primitives(tmp_path)
    extrinsic = np.eye(4)
    extrinsic[:3, 3] = [-1, -2, -3]
    primitive.env.render_camera = lambda *args, **kwargs: (
        np.zeros((2, 2, 3), dtype=np.uint8),
        np.ones((2, 2)),
    )
    primitive.env.get_camera_meta = lambda camera: {
        "extrinsic_cv": extrinsic if homogeneous else extrinsic[:3],
        "intrinsic_K": np.eye(3),
    }
    _, world, calibration = primitive.capture("head")
    np.testing.assert_allclose(world[0, 0], [1, 2, 4])
    np.testing.assert_allclose(world[1, 1], [2, 3, 4])
    np.testing.assert_allclose(
        np.asarray(calibration["extrinsic_cam2world"])[:3, 3], [1, 2, 3]
    )


def test_paired_plans_transform_tcp_to_native_eef_and_hold_both_grippers(tmp_path):
    primitive = twin_primitives(tmp_path)
    result = primitive.move_to("left", [0.1, 0.2, 0.5], other_xyz=[0.2, -0.2, 0.6])
    assert result.data["status"] == "reached"
    np.testing.assert_allclose(primitive.env.plans[0][1][:3], [0.1, 0.2, 0.4])
    np.testing.assert_allclose(primitive.env.plans[1][1][:3], [0.2, -0.2, 0.5])
    actions = primitive.env.chunks[0]
    assert actions.shape == (5, 14)
    np.testing.assert_allclose(actions[:, [6, 13]], 1)
    np.testing.assert_allclose(actions[-1, :6], 1)
    np.testing.assert_allclose(actions[-1, 7:13], 1)
    assert primitive.native_success() is False
    assert primitive.finish().data["success"] is False


def test_failed_second_plan_executes_neither_arm(tmp_path):
    primitive = twin_primitives(tmp_path, failed_arm="right")
    result = primitive.move_to("left", [0.1, 0.2, 0.5], other_xyz=[0.2, -0.2, 0.6])
    assert result.data["reason"] == "plan_failed"
    assert len(primitive.env.plans) == 2
    assert primitive.env.chunks == [] and primitive.env_steps == 0


def test_motion_holds_final_waypoint_until_measured_joint_drive_settles(tmp_path):
    primitive = twin_primitives(tmp_path)
    execute = primitive.env.chunk_step

    def lagging_drive(actions, *, action_type):
        result = execute(actions, action_type=action_type)
        if len(primitive.env.chunks) == 1:
            primitive.env.last_info["robot_state"]["left_tcp_pose"][2] -= 0.005
        return result

    primitive.env.chunk_step = lagging_drive
    result = primitive.move_to("left", [0.1, 0.2, 0.5])
    assert result.data["status"] == "reached"
    assert result.data["settling_steps"] == 10
    assert len(primitive.env.chunks) == 2
    np.testing.assert_allclose(
        primitive.env.chunks[1],
        np.repeat(primitive.env.chunks[0][-1][None], 10, axis=0),
    )


def test_measured_gripper_state_does_not_report_the_command_as_measurement(tmp_path):
    primitive = twin_primitives(tmp_path)
    primitive.env._client = SimpleNamespace(
        call=lambda method: {
            side: {"opening": 0.4, "commanded_opening": 0.0, "joint_position": 0.012}
            for side in ("left", "right")
        }
    )
    assert primitive.robot_state()["left"]["opening"] == 0.4
    result = primitive.set_gripper("left", 0.0)
    assert result.data["status"] == "partial"
    assert result.data["grippers"]["left"]["opening"] == 0.4


def test_gripper_pair_leaves_arm_targets_unchanged(tmp_path):
    primitive = twin_primitives(tmp_path)
    primitive.set_gripper("right", 0.2, other_opening=0.5)
    actions = primitive.env.chunks[0]
    np.testing.assert_allclose(actions[:, :6], 0)
    np.testing.assert_allclose(actions[:, 7:13], 0)
    np.testing.assert_allclose(actions[-1, [6, 13]], [0.5, 0.2])


def test_arm_labels_project_measured_tcp_into_the_named_camera(tmp_path):
    primitive = twin_primitives(tmp_path)
    primitive.env.last_info["robot_state"]["left_tcp_pose"][:3] = [0, 0, 1]
    primitive.env.last_info["robot_state"]["right_tcp_pose"][:3] = [0.1, 0, 1]
    image = np.zeros((100, 100, 3), dtype=np.uint8)
    calibration = {
        "intrinsic_K": [[100, 0, 20], [0, 100, 50], [0, 0, 1]],
        "extrinsic_cam2world": np.eye(4),
    }
    annotated = primitive.annotate(image, calibration)
    assert annotated.any() and not image.any()
    assert annotated[50, 15, 1] > 0  # left TCP circle, cyan


def casa_primitives(tmp_path):
    meta = {
        "gripper_max_gap": 0.08,
        "parts": {
            "right": {
                "input_min": [-1] * 6,
                "input_max": [1] * 6,
                "output_min": [-0.05] * 3 + [-0.5] * 3,
                "output_max": [0.05] * 3 + [0.5] * 3,
                "input_ref_frame": "base",
            },
            "base": {"actuator_min": [-2, -4, -5], "actuator_max": [2, 4, 5]},
        },
        "slices": {"right": [0, 6], "base": [7, 10]},
        "control_freq": 20,
    }
    rotation = Rotation.from_euler("z", np.pi / 2).as_matrix()
    frame = {"arm_origin_rotation": rotation, "base_theta": np.pi / 2}
    actions = []
    env = SimpleNamespace(
        gripper_qpos=[0.04, -0.04],
        eef_pos=np.array([0.0, 0.0, 1.0]),
        eef_quat=np.array([0.0, 0.0, 0.0, 1.0]),
        check_success=lambda: False,
        current_raw_obs={
            "robot0_base_pos": [0, 0, 0],
            "robot0_base_quat": [0, 0, 0, 1],
        },
        step=lambda action: actions.append(action.copy()),
        _client=SimpleNamespace(
            call=lambda method: meta if method == "env.control_meta" else frame
        ),
    )
    primitive = RoboCasaPrimitives(
        env=env,
        sam3_client=None,
        state=EnvState(tmp_path),
        check_cancelled=lambda: None,
    )
    return primitive, actions


def test_casa_world_delta_uses_the_actual_arm_controller_origin(tmp_path):
    primitive, actions = casa_primitives(tmp_path)
    primitive._step(np.array([0.01, 0, 0, 0, 0, 0, 0]))
    np.testing.assert_allclose(actions[0][:6], [0, -0.2, 0, 0, 0, 0], atol=1e-8)
    assert actions[0][6] == 0 and actions[0][-1] == -1


def test_casa_base_velocity_inverts_native_scaling_and_stops(tmp_path):
    primitive, actions = casa_primitives(tmp_path)
    desired = np.array([0.1, 0.2, 0.3])
    raw = primitive._base_velocity(desired)
    rotation = Rotation.from_euler("z", -np.pi / 2).as_matrix()
    np.testing.assert_allclose(np.diag([2, 4, 5]) @ rotation @ raw, rotation @ desired)
    result = primitive.move_base(*desired, duration_s=0.1)
    assert result.data["status"] == "commanded"
    assert len(actions) == 3  # two 20Hz commands plus explicit stop
    np.testing.assert_allclose(actions[-1][7:10], 0, atol=1e-8)
    assert primitive.native_success() is False


def test_casa_rejects_out_of_range_velocity_before_stepping(tmp_path):
    primitive, actions = casa_primitives(tmp_path)
    assert primitive.move_base(vx=3).is_error and actions == []


def test_casa_gripper_normalization_uses_physical_limits_not_reset_gap(tmp_path):
    primitive, _ = casa_primitives(tmp_path)
    primitive.env.gripper_qpos = [0.02, -0.02]
    assert primitive.measure().opening == 0.5
    primitive.env.gripper_qpos = [0.0, 0.0]
    assert primitive.measure().opening == 0.0
