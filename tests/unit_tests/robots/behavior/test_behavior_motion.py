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

"""R1Pro planning and native arm, base, gripper and wrist action results."""

from __future__ import annotations

import math
import sys
from copy import deepcopy
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest
import yaml
from scipy.spatial.transform import Rotation

from robots.behavior.env_client import BehaviorEnvClient
from robots.behavior.env_server import BehaviorEnvFacade
from robots.behavior.motion import R1ProMotionModel, R1ProMotionPlanner
from robots.behavior.primitives import (
    BASE_OUTPUT_SCALE,
    WRIST_JOINT_OFFSET,
    BehaviorPrimitives,
)
from rpent.tools.toolkit import ToolCancelled
from tests.unit_tests.robots.behavior.conftest import (
    ACTION_COUNT_FIELDS,
    LocalRpc,
    _execution_feedback,
    _GripperEnvironment,
    _GripperModel,
    _robot_observation,
)


class _NativeRobot:
    """Loaded robot double; getters never move its joints or episode clock."""

    model = "r1pro"
    usd_path = "/native/r1pro.usd"
    eef_link_names = {"left": "left_eef_link", "right": "right_eef_link"}

    def __init__(self, path):
        names = [
            f"base_footprint_{axis}_joint" for axis in ("x", "y", "z", "rx", "ry", "rz")
        ]
        names += list(R1ProMotionModel.joint_positions(np.zeros(61)))
        self.joints = {
            name: SimpleNamespace(
                lower_limit=0.0 if "finger" in name else -3.0,
                upper_limit=0.05 if "finger" in name else 3.0,
                max_velocity=1.0,
            )
            for name in names
        }
        self.proprio = np.zeros(61)
        self.base_positions = np.arange(6, dtype=float) * 0.1
        self.world_pose = (np.zeros(3), np.array([0.0, 0.0, 0.0, 1.0]))
        self.solver_pose = (np.zeros(3), np.array([0.0, 0.0, 0.0, 1.0]))
        self.links = {
            "base_footprint_x": SimpleNamespace(
                get_position_orientation=lambda: self.solver_pose
            )
        }
        self.curobo_path = {"arm": str(path)}

    def get_joint_positions(self):
        return np.r_[
            self.base_positions,
            list(R1ProMotionModel.joint_positions(self.proprio).values()),
        ]

    def get_position_orientation(self):
        return self.world_pose

    def get_relative_eef_pose(self, arm):
        return np.array([0.0, 0.0, 1.14]), np.array([0.0, 0.0, 0.0, 1.0])


@pytest.fixture
def native_robot(tmp_path):
    path = tmp_path / "r1pro_arm.yaml"
    path.write_text(
        yaml.safe_dump(
            {
                "robot_cfg": {
                    "kinematics": {
                        "base_link": "base_footprint_x",
                        "ee_link": "left_eef_link",
                        "link_names": ["right_eef_link"],
                        "collision_spheres": {
                            "left_arm_link1": [{"center": [0, 0, 0], "radius": 0.02}],
                            "right_arm_link1": [{"center": [0, 0, 0], "radius": 0.02}],
                        },
                        "collision_link_names": ["left_arm_link1", "right_arm_link1"],
                        "self_collision_ignore": {"base_link": ["torso_link1"]},
                        "lock_joints": {},
                        "cspace": {"max_acceleration": 15, "max_jerk": 500},
                    }
                }
            }
        )
    )
    return _NativeRobot(path)


@pytest.fixture
def motion_model(native_robot):
    model = R1ProMotionModel(lambda: native_robot)
    yield model
    model.close()


@pytest.mark.parametrize("arm", ["left", "right"])
def test_native_config_limits_and_locks_preserve_official_geometry(motion_model, arm):
    robot = motion_model.robot
    path = next(iter(robot.curobo_path.values()))
    from pathlib import Path

    original = Path(path).read_bytes()
    config = motion_model.robot_config(arm, robot.proprio)["kinematics"]
    active = [f"{arm}_arm_joint{i}" for i in range(1, 8)]
    assert config["cspace"]["joint_names"] == active
    assert config["ee_link"] == f"{arm}_eef_link"
    assert config["use_usd_kinematics"] is True
    assert len(config["lock_joints"]) == 21
    assert set(config["lock_joints"]) == set(robot.joints) - set(active)
    assert config["lock_joints"]["base_footprint_rz_joint"] == 0.5
    source = yaml.safe_load(original)["robot_cfg"]["kinematics"]
    assert config["collision_spheres"] == source["collision_spheres"]
    assert config["self_collision_ignore"] == source["self_collision_ignore"]
    assert Path(path).read_bytes() == original
    np.testing.assert_array_equal(
        motion_model.joint_limits(arm), np.tile([-3, 3], (7, 1))
    )
    robot.joints[active[0]].max_velocity = 0.3
    assert motion_model.velocity_limits(arm)[0] == 0.3


def test_motion_model_requires_native_provider_and_rebinds_after_scene_rebuild(
    native_robot, tmp_path
):
    with pytest.raises(TypeError, match="get_robot"):
        R1ProMotionModel(native_robot)
    current = [native_robot]
    model = R1ProMotionModel(lambda: current[0])
    assert model.robot is native_robot
    model._generators[("left", "cuda:0")] = object()
    current[0] = _NativeRobot(next(iter(native_robot.curobo_path.values())))
    assert model.robot is current[0]
    assert model._generators == {}
    model.close()


def test_native_model_rejects_stale_public_joint_snapshot(motion_model):
    state = motion_model.robot.proprio.copy()
    state[28] = 0.1
    with pytest.raises(ValueError, match="snapshot differs"):
        motion_model.robot_config("left", state)


def test_solver_frame_transform_uses_actual_robot_and_native_base_poses(motion_model):
    robot = motion_model.robot
    robot.world_pose = (
        np.array([2.0, -1.0, 0.3]),
        Rotation.from_euler("z", 0.6).as_quat(),
    )
    robot.solver_pose = (
        np.array([0.2, 0.1, 0.0]),
        Rotation.from_euler("z", -0.3).as_quat(),
    )
    goal = np.r_[
        [0.4, -0.2, 0.9], Rotation.from_euler("xyz", [0.1, 0.2, 0.3]).as_quat()
    ]
    native = motion_model.transform_pose(goal, to_solver=True)
    assert not np.allclose(native[:3], goal[:3])
    np.testing.assert_allclose(
        motion_model.transform_pose(native, to_solver=False), goal, atol=1e-12
    )


def test_physical_joint_map_uses_public_positions():
    values = R1ProMotionModel.joint_positions(np.arange(61, dtype=float))
    assert len(values) == 22
    assert values["torso_joint1"] == 53
    assert values["left_arm_joint1"] == 3
    assert values["right_arm_joint7"] == 34
    assert values["left_gripper_finger_joint2"] == 25
    assert values["right_gripper_finger_joint1"] == 49


def motion_context(state, previous=None):
    initial = previous is None
    index = 0 if initial else 1
    return {
        "source": "confirmed_public_control_observations",
        "control_dt_s": 1 / 30,
        "current_token": {"generation": 0, "observation_index": index},
        "previous_token": None
        if initial
        else {"generation": 0, "observation_index": 0},
        "current_proprio": state.copy(),
        "previous_proprio": None if initial else previous.copy(),
        "pair_confirmed": not initial,
        "initial_bootstrap": initial,
        "issued_actions": index,
        "confirmed_actions": index,
        "unconfirmed_actions": 0,
        "actions_pending": False,
    }


@pytest.fixture
def planning_request(motion_model):
    planner = R1ProMotionPlanner(model=motion_model, task_id=40)
    proprio = motion_model.robot.proprio
    proprio[24:26] = 0.05
    proprio[49:51] = 0.025
    for arm, offset in (("left", 17), ("right", 42)):
        position, quaternion = planner.model.robot.get_relative_eef_pose(arm)
        proprio[offset : offset + 3] = position
        proprio[offset + 3 : offset + 7] = quaternion
    raw = {
        "r1::proprio": proprio,
        "r1::cam_rel_poses": np.tile([0, 0, 0, 0, 0, 0, 1], 3),
    }
    for camera in ("zed_link", "left_realsense_link", "right_realsense_link"):
        raw[f"r1::{camera}:Camera:0::rgb"] = np.zeros((224, 224, 3), np.uint8)
        raw[f"r1::{camera}:Camera:0::depth_linear"] = np.ones((224, 224))
    return planner, raw, proprio[17:24].copy()


@pytest.mark.parametrize(
    "fingers",
    [
        {"left": [0.02, 0.02], "right": [0.0248, 0.0252]},
        {"left": [0.01, 0.04], "right": [0.03, 0.01]},
    ],
)
def test_native_plan_preserves_measured_fingers_and_reports_collision_scope(
    planning_request, monkeypatch, fingers
):
    planner, raw, target = planning_request
    proprio = raw["r1::proprio"]
    proprio[24:26] = fingers["left"]
    proprio[49:51] = fingers["right"]
    before = proprio.copy()
    _fake_curobo(monkeypatch)
    solver = _IKSolver(target, proprio[3:10])
    monkeypatch.setattr(planner, "_ik_solver", lambda *args: solver)
    plan = planner.plan_arm_path(
        arm="left",
        target_pose=target,
        observation=raw,
        motion_context=motion_context(proprio),
    )
    assert plan["status"] == "Success"
    assert plan["held_finger_qpos"] == fingers
    assert plan["held_objects_modeled"] is plan["collision_checked"] is False
    assert plan["collision_scope"]["source"] == "none_ik_only"
    np.testing.assert_array_equal(proprio, before)


def test_invalid_finger_geometry_is_rejected_before_ik(planning_request, monkeypatch):
    planner, raw, target = planning_request
    raw["r1::proprio"][24:26] = 0.06

    def forbidden_solver(*args):
        raise AssertionError("invalid finger geometry reached the IK solver")

    monkeypatch.setattr(planner, "_ik_solver", forbidden_solver)
    plan = planner.plan_arm_path(
        arm="left",
        target_pose=target,
        observation=raw,
        motion_context=motion_context(raw["r1::proprio"]),
    )
    assert plan["status"] == "Unavailable"
    assert plan["code"] == "invalid_finger_geometry"


def _fake_curobo(monkeypatch):
    modules = {
        "curobo.types.math": {"Pose": SimpleNamespace},
        "curobo.rollout.cost.pose_cost": {"PoseCostMetric": _PoseCostMetric},
    }
    for name, attributes in modules.items():
        parts = name.split(".")
        for length in range(1, len(parts)):
            package = ".".join(parts[:length])
            if package not in sys.modules:
                monkeypatch.setitem(sys.modules, package, ModuleType(package))
        module = ModuleType(name)
        module.__dict__.update(attributes)
        monkeypatch.setitem(sys.modules, name, module)


class _PoseCostMetric(SimpleNamespace):
    @classmethod
    def reset_metric(cls):
        return cls(reach_partial_pose=False, reach_full_pose=True)


class _Tensor:
    def __init__(self, values):
        self.values = np.asarray(values)

    def clone(self):
        return _Tensor(self.values.copy())

    def detach(self):
        return self

    def cpu(self):
        return self

    def numpy(self):
        return self.values


class _IKSolver:
    """CPU IK double with an optional position-only solution."""

    tensor_args = SimpleNamespace(to_device=np.asarray)

    def __init__(self, target, start_q, relaxed_wxyz=None, relaxed_success=True):
        self.target = target
        self.primary_arm = "left"
        self.other_pose = target.copy()
        self.target_q = start_q
        self.relaxed_wxyz = relaxed_wxyz
        self.relaxed_success = relaxed_success
        self.metric = None

    def update_pose_cost_metric(self, metric):
        self.metric = metric

    def fk(self, q):
        q = np.asarray(q).reshape(-1, 7)
        # The current posture reports the true pose; any other q reports the
        # relaxed solution's pose.
        if self.relaxed_wxyz is None or np.allclose(q[0], self.target_q):
            return SimpleNamespace(
                ee_position=_Tensor(self.target[:3]),
                ee_quaternion=_Tensor(self.target[[6, 3, 4, 5]]),
                link_poses={
                    f"{arm}_eef_link": SimpleNamespace(
                        position=_Tensor(
                            (
                                self.target
                                if arm == self.primary_arm
                                else self.other_pose
                            )[:3].copy()
                        ),
                        quaternion=_Tensor(
                            (
                                self.target
                                if arm == self.primary_arm
                                else self.other_pose
                            )[[6, 3, 4, 5]].copy()
                        ),
                    )
                    for arm in ("left", "right")
                },
            )
        return SimpleNamespace(
            ee_position=_Tensor(self.target[:3]),
            ee_quaternion=_Tensor(self.relaxed_wxyz),
        )

    def solve_single(self, goal, seed_config=None, return_seeds=1, link_poses=None):
        other = "right_eef_link" if self.primary_arm == "left" else "left_eef_link"
        assert link_poses is not None
        assert set(link_poses) == {other}
        np.testing.assert_allclose(
            link_poses[other].position.values, self.other_pose[:3]
        )
        np.testing.assert_allclose(
            link_poses[other].quaternion.values, self.other_pose[[6, 3, 4, 5]]
        )
        relaxed = bool(getattr(self.metric, "reach_partial_pose", False))
        goal_q = np.asarray(seed_config).reshape(7) + (0.002 if relaxed else 0.001)
        success = (relaxed and self.relaxed_success) or self.relaxed_wxyz is None
        return SimpleNamespace(
            success=np.asarray([success]),
            position_error=np.asarray(
                [0.0 if success else (0.08 if relaxed else 0.04)]
            ),
            rotation_error=np.asarray([0.0 if success else (0.6 if relaxed else 0.3)]),
            js_solution=SimpleNamespace(
                get_ordered_joint_state=lambda names: SimpleNamespace(
                    position=_Tensor(goal_q[None, :])
                )
            ),
        )


@pytest.mark.parametrize("relaxed_success", [True, False])
def test_relaxed_orientation_retry_is_reported_not_hidden(
    planning_request, monkeypatch, relaxed_success
):
    planner, raw, target = planning_request
    _fake_curobo(monkeypatch)
    # A wrist turned 90 degrees about z relative to the requested orientation.
    turned = (
        Rotation.from_euler("z", 90, degrees=True) * Rotation.from_quat(target[3:])
    ).as_quat()
    solver = _IKSolver(
        target, raw["r1::proprio"][3:10], turned[[3, 0, 1, 2]], relaxed_success
    )
    monkeypatch.setattr(planner, "_ik_solver", lambda *args: solver)
    context = motion_context(raw["r1::proprio"])

    strict = planner.plan_arm_path(
        arm="left", target_pose=target, observation=raw, motion_context=context
    )
    assert strict["status"] == "Failed" and strict["code"] == "curobo_ik_failed"
    assert strict["orientation_relax_attempted"] is False
    relaxed = planner.plan_arm_path(
        arm="left",
        target_pose=target,
        observation=raw,
        motion_context=context,
        allow_orientation_relax=True,
    )
    assert solver.metric.reach_partial_pose is False
    if not relaxed_success:
        assert relaxed["status"] == "Failed"
        assert relaxed["orientation_relax_attempted"] is True
        for key in ("code", "position_error_m", "orientation_error_rad"):
            assert relaxed[key] == strict[key]
    else:
        assert relaxed["status"] == "Success"
        assert relaxed["orientation_relaxed"] is True
        assert relaxed["ik_orientation_error_rad"] == pytest.approx(np.pi / 2, abs=1e-6)
        assert relaxed["target_pose"] == target.tolist()


@pytest.mark.parametrize("arm", ["left", "right"])
def test_cached_warmup_link_goal_is_replaced_and_owns_its_fk_snapshot(
    planning_request, monkeypatch, arm
):
    planner, raw, _ = planning_request
    state = raw["r1::proprio"]
    offset, other_offset = (17, 42) if arm == "left" else (42, 17)
    qpos = state[3:10] if arm == "left" else state[28:35]
    target = state[offset : offset + 7].copy()
    solver = _IKSolver(target, qpos)
    solver.primary_arm = arm
    other = "right_eef_link" if arm == "left" else "left_eef_link"
    _fake_curobo(monkeypatch)
    monkeypatch.setattr(planner, "_ik_solver", lambda *args: solver)
    # Model the native solver's persistent warmup goal: None would retain it.
    cached = {"position": np.full(3, 7.0), "original_fk": None}
    original_fk, original_solve = solver.fk, solver.solve_single

    def fk(q):
        result = original_fk(q)
        cached["original_fk"] = result.link_poses[other]
        return result

    def solve(*args, **kwargs):
        links = kwargs.get("link_poses")
        if links is not None:
            cached["position"] = links[other].position.values.copy()
        result = original_solve(*args, **kwargs)
        result.success[:] = np.allclose(cached["position"], solver.other_pose[:3])
        # Native FK uses reusable buffers. Optimizing another candidate must
        # not move the already-supplied goal for the locked opposite arm.
        cached["original_fk"].position.values[:] += 2
        np.testing.assert_allclose(links[other].position.values, solver.other_pose[:3])
        return result

    monkeypatch.setattr(solver, "fk", fk)
    monkeypatch.setattr(solver, "solve_single", solve)
    for x_shift in (0.0, 0.012):
        state[other_offset] = target[0] + x_shift
        solver.other_pose = state[other_offset : other_offset + 7].copy()
        plan = planner.plan_arm_path(
            arm=arm,
            target_pose=target,
            observation=raw,
            motion_context=motion_context(state),
        )
        assert plan["status"] == "Success"
        np.testing.assert_allclose(cached["position"], solver.other_pose[:3])


def test_native_ik_cache_updates_locks_and_rebuilds_after_reset(
    native_robot, monkeypatch
):
    from pathlib import Path

    current = [native_robot]
    model = R1ProMotionModel(lambda: current[0])
    created = []

    class Generator:
        def __init__(self, robot, **kwargs):
            self.robot = robot
            self.path = kwargs["robot_cfg_path"]
            self.config = yaml.safe_load(Path(self.path).read_text())["robot_cfg"][
                "kinematics"
            ]
            self.mg = {
                "default": SimpleNamespace(
                    ik_solver=SimpleNamespace(
                        joint_names=self.config["cspace"]["joint_names"]
                    )
                )
            }
            self.updates = []
            created.append(self)

        def update_locked_joints(self, state, embodiment):
            assert embodiment == "default"
            self.updates.append(np.array(state.position, copy=True))

    modules = {
        "torch": {
            "as_tensor": lambda x, **kwargs: np.asarray(x),
            "float32": np.float32,
        },
        "curobo.types.state": {
            "JointState": lambda **kwargs: SimpleNamespace(**kwargs)
        },
        "omnigibson.action_primitives.curobo": {
            "CuRoboEmbodimentSelection": SimpleNamespace(DEFAULT="default"),
            "CuRoboMotionGenerator": Generator,
        },
    }
    for name, values in modules.items():
        parts = name.split(".")
        for count in range(1, len(parts)):
            package = ".".join(parts[:count])
            if package not in sys.modules:
                monkeypatch.setitem(sys.modules, package, ModuleType(package))
        module = ModuleType(name)
        module.__dict__.update(values)
        monkeypatch.setitem(sys.modules, name, module)
    solver = model.ik_solver("left", native_robot.proprio, "cpu")
    first = created[0]
    native_robot.proprio[28] = 0.3
    native_robot.proprio[49:51] = [0.02, 0.03]
    native_robot.base_positions[5] = 0.8
    assert model.ik_solver("left", native_robot.proprio, "cpu") is solver
    assert len(created) == 1 and len(first.updates) == 2
    np.testing.assert_array_equal(
        first.updates[-1][0], native_robot.get_joint_positions()
    )
    assert not np.array_equal(first.updates[0], first.updates[1])
    previous_path = first.path
    current[0] = _NativeRobot(next(iter(native_robot.curobo_path.values())))
    assert model.ik_solver("left", current[0].proprio, "cpu") is not solver
    assert created[1].robot is current[0]
    assert not Path(previous_path).exists()
    final_path = created[1].path
    model.close()
    assert not Path(final_path).exists()


def test_curobo_opposite_arm_fk_disagreement_refuses_plan(
    planning_request, monkeypatch
):
    planner, raw, target = planning_request
    _fake_curobo(monkeypatch)
    solver = _IKSolver(target, raw["r1::proprio"][3:10])
    original_fk = solver.fk

    def wrong_locked_geometry(q):
        value = original_fk(q)
        value.link_poses["right_eef_link"].position = _Tensor(target[:3] + [0.1, 0, 0])
        return value

    monkeypatch.setattr(solver, "fk", wrong_locked_geometry)
    monkeypatch.setattr(planner, "_ik_solver", lambda *args: solver)
    result = planner.plan_arm_path(
        arm="left",
        target_pose=target,
        observation=raw,
        motion_context=motion_context(raw["r1::proprio"]),
    )
    assert result["status"] == "Unavailable" and result["code"] == "curobo_fk_mismatch"
    assert result["fk_comparison"]["source"] == "native_usd_curobo_fk"
    assert result["fk_comparison"]["right"]["position_error_m"] == pytest.approx(0.1)


def test_cached_pose_metric_is_reset_even_if_relaxed_solve_raises(
    planning_request, monkeypatch
):
    planner, raw, target = planning_request
    _fake_curobo(monkeypatch)
    solver = _IKSolver(
        target, raw["r1::proprio"][3:10], relaxed_wxyz=np.array([1, 0, 0, 0])
    )
    original_solve = solver.solve_single

    def failing_relaxed_solve(*args, **kwargs):
        if getattr(solver.metric, "reach_partial_pose", False):
            raise RuntimeError("IK failed during relaxed retry")
        return original_solve(*args, **kwargs)

    monkeypatch.setattr(solver, "solve_single", failing_relaxed_solve)
    monkeypatch.setattr(planner, "_ik_solver", lambda *args: solver)
    with pytest.raises(RuntimeError, match="relaxed retry"):
        planner.plan_arm_path(
            arm="left",
            target_pose=target,
            observation=raw,
            motion_context=motion_context(raw["r1::proprio"]),
            allow_orientation_relax=True,
        )
    assert solver.metric.reach_partial_pose is False
    strict = planner.plan_arm_path(
        arm="left",
        target_pose=target,
        observation=raw,
        motion_context=motion_context(raw["r1::proprio"]),
    )
    assert (
        strict["status"] == "Failed" and strict["orientation_relax_attempted"] is False
    )


def test_path_cap_cannot_override_native_velocity_limit(planning_request, monkeypatch):
    planner, raw, target = planning_request
    _fake_curobo(monkeypatch)
    solver = _IKSolver(target, raw["r1::proprio"][3:10])
    monkeypatch.setattr(planner, "_ik_solver", lambda *args: solver)
    planner.model.robot.joints["left_arm_joint1"].max_velocity = 1e-6
    result = planner.plan_arm_path(
        arm="left",
        target_pose=target,
        observation=raw,
        motion_context=motion_context(raw["r1::proprio"]),
    )
    assert result["status"] == "Unavailable" and result["code"] == "path_too_long"


@pytest.mark.parametrize("arm,other_gripper", [("left", 1), ("right", 1), ("left", -1)])
def test_arm_path_preserves_other_body_targets_and_gripper_command(
    motion, arm, other_gripper
):
    primitives, backend, planner, _, model = motion
    commands = [1, 1]
    commands[1 if arm == "left" else 0] = other_gripper
    model.grippers = commands
    primitives.vla_act()
    if other_gripper == -1:
        # Contact can keep measured fingers open while their command closes.
        backend.raw["r1::proprio"][49:51] = [0.01, 0.04]
        backend.previous_proprio[49:51] = [0.01, 0.04]
        backend.raw["r1::proprio"][51:53] = -0.01
    before = backend.raw["r1::proprio"].copy()
    result = primitives.move_to(
        arm=arm, xyz=_motion_target(backend, arm), settle_steps=0
    )
    actions = backend.actions[-1]
    moving = slice(7, 14) if arm == "left" else slice(15, 22)
    observed = slice(3, 10) if arm == "left" else slice(28, 35)
    expected = np.tile(_held_action(before, commands), (4, 1))
    expected[:, moving] = before[observed] + np.arange(1, 5)[:, None] * 0.005
    np.testing.assert_allclose(actions, expected)
    assert not result.is_error and result.data["completed"] is True
    assert result.data["target_reached"] is True
    assert result.data["confirmed_actions"] == len(actions) == 4
    assert len(planner.calls) == model.calls == 1
    assert [len(chunk) for chunk in backend.actions] == [2, 4]


def test_target_outside_short_reposition_scope_never_reaches_solver(motion):
    primitives, backend, planner, _, _ = motion
    primitives.vla_act()
    target = np.asarray(_motion_target(backend)) + [0.301, 0, 0]
    result = primitives.move_to(arm="left", xyz=target.tolist())
    assert result.data["code"] == "target_outside_motion_scope"
    assert len(backend.actions) == 1 and planner.calls == []


@pytest.mark.parametrize(
    "failure,code",
    [
        ({"status": "Failed", "code": "unreachable"}, "unreachable"),
        ({"status": "Unavailable", "code": "depth_missing"}, "depth_missing"),
        (None, "stale_arm_plan"),
    ],
)
def test_failed_or_stale_plan_never_executes_or_falls_back_to_vla(
    motion, failure, code
):
    primitives, backend, planner, _, model = motion
    primitives.vla_act()

    def changed_plan(plan, observation):
        if failure is None:
            backend.raw["r1::proprio"][53] = 0.2
            backend.observation_index += 1
            backend.issued += 1
            backend.confirmed += 1
        else:
            plan.update(failure)

    planner.mutate = changed_plan
    result = primitives.move_to(arm="left", xyz=_motion_target(backend))
    assert result.is_error and result.data["code"] == code
    assert len(backend.actions) == model.calls == 1


def test_partial_terminal_execution_keeps_counts_and_never_claims_fresh_pose(motion):
    primitives, backend, _, _, model = motion
    primitives.vla_act()
    backend.feedback = {
        "issued_actions": 2,
        "confirmed_actions": 1,
        "unconfirmed_actions": 1,
        "executed_actions": None,
        "observation_after_last_issued_action": False,
        "stop_reason": "session_ended",
    }
    result = primitives.move_to(arm="left", xyz=_motion_target(backend))
    assert result.data["issued_actions"] == 2
    assert result.data["confirmed_actions"] == result.data["unconfirmed_actions"] == 1
    assert result.data["executed_actions"] is None
    assert result.data["target_reached"] is None
    assert result.data["position_error_m"] is None
    assert result.data["session_ended"] is True
    assert len(backend.actions) == 2 and model.calls == 1
    assert primitives.move_to(arm="left", xyz=_motion_target(backend)).is_error
    assert len(backend.actions) == 2


@pytest.mark.parametrize(
    "budget,stop", [(0, "target_not_reached"), (2, "target_hold_budget_exhausted")]
)
def test_unreached_pose_is_not_completed_after_optional_holds(motion, budget, stop):
    _establish_prior_hold(motion)
    primitives, backend, _, _, _ = motion
    original = backend.chunk_step

    def missed_pose(actions, **kwargs):
        observation, *rest = original(actions, **kwargs)
        observation["r1::proprio"][17] += 0.006
        rest[-1]["motion_context"]["current_proprio"] = observation[
            "r1::proprio"
        ].copy()
        return observation, *rest

    backend.chunk_step = missed_pose
    result = primitives.move_to(
        arm="left", xyz=_motion_target(backend), settle_steps=budget
    )
    assert result.is_error
    assert result.data["target_reached"] is result.data["completed"] is False
    assert result.data["stop_reason"] == stop
    assert [len(actions) for actions in backend.actions] == [4] + [1] * budget


def test_transport_failure_is_not_replayed_and_invalidates_last_command(motion):
    primitives, backend, _, _, _ = motion
    primitives.vla_act()
    backend.fail = True
    result = primitives.move_to(arm="left", xyz=_motion_target(backend))
    assert result.data["code"] == "execution_outcome_unknown"
    assert result.data["requested_actions"] == 4
    assert result.data["confirmed_actions"] is None
    backend.fail = False
    result = primitives.move_to(arm="left", xyz=_motion_target(backend))
    assert result.data["code"] == "command_state_unknown"
    assert len(backend.actions) == 2


@pytest.mark.parametrize(
    "tool,arm,delta,angle",
    [
        ("move_eef_delta", "left", [0.01, -0.012, 0.005], 0),
        ("move_eef_delta", "right", [0.01, -0.012, 0.005], 0),
        ("rotate_wrist", "left", [0, 0, 0], -15),
        ("rotate_wrist", "right", [0, 0, 0], 15),
    ],
)
def test_relative_arm_tools_use_current_pose_and_correct_base_frame(
    motion, tool, arm, delta, angle
):
    primitive, backend, planner, _, model = motion
    start = 17 if arm == "left" else 42
    original = Rotation.from_euler("xyz", [0.4, -0.3, 0.2])
    backend.raw["r1::proprio"][start + 3 : start + 7] = original.as_quat()
    _establish_prior_hold(motion)
    before = backend.raw["r1::proprio"].copy()
    primitive.last_observation = deepcopy(backend.raw)
    primitive.last_observation["r1::proprio"][start : start + 3] += 0.1
    _follow_planned_pose(backend, planner)
    args = (
        {"delta_xyz": delta} if tool == "move_eef_delta" else {"delta_yaw_deg": angle}
    )
    result = getattr(primitive, tool)(arm=arm, **args)
    goal = planner.calls[0][1]
    np.testing.assert_allclose(goal[:3], before[start : start + 3] + delta)
    expected = Rotation.from_rotvec([0, 0, np.deg2rad(angle)]) * original
    np.testing.assert_allclose(
        Rotation.from_quat(goal[3:]).as_matrix(), expected.as_matrix()
    )
    assert not result.is_error and result.data["completed"] is True
    assert result.data["target_reached"] is True
    assert [actions.shape for actions in backend.actions] == [(4, 23)]
    assert len(planner.calls) == 1 and model.calls == 0


def _make_primitives(env, model, cancelled=lambda: None):
    return BehaviorPrimitives(
        env=env,
        model=model,
        task_id=40,
        task_language="make popcorn",
        check_cancelled=cancelled,
        execution_horizon=2,
        instruction_mode="main",
    )


def _held_action(proprio, grippers):
    """Expected native23 hold, independently assembled from measured state."""
    return np.r_[
        np.zeros(3),
        proprio[53:57],
        proprio[3:10],
        grippers[0],
        proprio[28:35],
        grippers[1],
    ]


@pytest.mark.parametrize("axis", range(3))
@pytest.mark.parametrize("sign", [-1, 1])
def test_base_commands_use_correct_channels_and_confirm_stop(motion, axis, sign):
    _establish_prior_hold(motion)
    primitive, backend, _, _, model = motion
    velocity = np.zeros(3)
    velocity[axis] = sign * 0.1
    result = primitive.move_base(
        **dict(zip(("forward", "lateral", "turn"), velocity, strict=True)), steps=3
    ).data
    stack = np.concatenate(backend.actions)
    np.testing.assert_allclose(stack[:3, :3], np.tile(velocity, (3, 1)))
    np.testing.assert_equal(stack[-1, :3], 0)
    np.testing.assert_allclose(stack[:, 3:], np.tile(stack[0, 3:], (4, 1)))
    np.testing.assert_allclose(
        result["target_base_velocity_m_s_rad_s"], velocity * BASE_OUTPUT_SCALE
    )
    assert result["completed"] and result["stop_command_confirmed"]
    assert result["issued_actions"] == result["confirmed_actions"] == 4
    assert result["unconfirmed_actions"] == model.calls == 0
    assert [item["observation_index"] for item in backend.received_tokens] == [
        1,
        2,
        3,
        4,
    ]


def test_cancellation_preserves_executed_prefix_without_claiming_stop():
    armed = False

    def cancelled():
        if armed and backend.actions:
            raise ToolCancelled("cancelled by caller")

    runtime = _motion_runtime(cancelled)
    primitive, backend, _, _, _ = runtime
    _establish_prior_hold(runtime)
    armed = True
    result = primitive.move_base(forward=0.1, steps=3).data
    assert result["confirmed_actions"] == 1 and result["interrupted"]
    assert result["stop_reason"] == "tool_cancelled"
    assert len(backend.actions) == 1 and not result["stop_command_confirmed"]


@pytest.mark.parametrize(
    "method,kwargs",
    [
        ("move_to", {"arm": "left", "xyz": [float("nan"), 0, 0]}),
        ("set_gripper", {"arm": "left", "val": 2}),
        ("move_base", {"forward": 2}),
    ],
)
def test_invalid_motion_request_does_not_send_actions(motion, method, kwargs):
    primitives, backend, planner, _, model = motion
    with pytest.raises(ValueError):
        getattr(primitives, method)(**kwargs)
    assert backend.actions == planner.calls == []
    assert model.calls == 0


@pytest.mark.parametrize(
    "tool,arm,val,ramp",
    [
        ("set_gripper", "left", 1, [-0.2, 0.2, 0.6, 1.0]),
        ("set_gripper", "left", 0, [-0.7, -0.8, -0.9, -1.0]),
        ("set_gripper", "right", 1, [0.7, 0.8, 0.9, 1.0]),
        ("set_gripper", "right", 0, [0.2, -0.2, -0.6, -1.0]),
        ("release", "left", 1, [-0.2, 0.2, 0.6, 1.0]),
    ],
)
def test_gripper_actions_reach_correct_commands_and_hold_other_parts(
    gripper, tool, arm, val, ramp
):
    primitive, backend, model = gripper
    primitive.vla_act()
    before = backend.raw["r1::proprio"].copy()
    kwargs = {"val": val} if tool == "set_gripper" else {}
    result = getattr(primitive, tool)(arm=arm, steps=4, settle_steps=0, **kwargs)
    expected = np.tile(_held_action(before, [-0.6, 0.6]), (4, 1))
    expected[:, 14 if arm == "left" else 22] = ramp
    np.testing.assert_allclose(backend.actions[-1], expected)
    assert not result.is_error
    assert result.data["issued_actions"] == result.data["confirmed_actions"] == 4
    assert result.data["grasp_success"] is None
    assert model.calls == 1 and len(backend.actions) == 2


def _assert_not_issued(result):
    for name in ACTION_COUNT_FIELDS:
        assert result.data[name] == 0


def test_partial_execution_continues_from_last_issued_command_not_requested_target(
    gripper,
):
    primitives, env, _ = gripper
    primitives.vla_act()
    env.feedback = {
        "issued_actions": 2,
        "confirmed_actions": 1,
        "unconfirmed_actions": 1,
        "executed_actions": None,
        "observation_after_last_issued_action": False,
        "stop_reason": "partial_execution",
    }
    partial = primitives.set_gripper(arm="left", val=1, steps=4, settle_steps=0)
    assert partial.data["issued_actions"] == 2
    assert partial.data["confirmed_actions"] == 1
    assert partial.data["unconfirmed_actions"] == 1
    assert partial.data["executed_actions"] is None
    assert partial.data["finger_qpos_after_m"] is None
    assert partial.data["finger_target_position_error_m"] is None
    assert partial.data["finger_target_position_reached"] is None
    assert partial.data["finger_qvel_after_m_s"] is None
    assert partial.data["settled"] is None
    assert "success" not in partial.data
    env.feedback = None
    result = primitives.set_gripper(arm="left", val=0, steps=1)
    assert result.data["start_val"] == pytest.approx(0.6)
    assert env.actions[-1][0, 14] == -1


def test_missing_counts_stay_unknown_and_invalidate_next_hold_command(gripper):
    primitives, env, _ = gripper
    primitives.vla_act()
    env.feedback = {"observation_after_last_issued_action": False}
    result = primitives.release(arm="right")
    for key in ACTION_COUNT_FIELDS:
        assert result.data[key] is None
    assert result.data["finger_qpos_after_m"] is None
    count = len(env.actions)
    rejected = primitives.release(arm="left")
    assert rejected.data["code"] == "command_state_unknown"
    _assert_not_issued(rejected)
    assert len(env.actions) == count


@pytest.mark.parametrize(
    "tool,arguments",
    [
        ("set_gripper", {"val": 1}),
        ("release", {}),
        ("rotate_wrist_joint", {"joint": "yaw", "delta_deg": 1}),
    ],
)
@pytest.mark.parametrize(
    "invalid_counts",
    [
        {"issued_actions": 2},
        {"confirmed_actions": 2, "executed_actions": 2},
        {"unconfirmed_actions": 2},
        {"issued_actions": -1},
        {"issued_actions": True},
        {"issued_actions": 1.0},
        {"confirmed_actions": 0},
        {"unconfirmed_actions": 1},
        {"issued_actions": 0, "confirmed_actions": None},
        {
            "issued_actions": 0,
            "confirmed_actions": None,
            "executed_actions": None,
            "unconfirmed_actions": 1,
        },
        {"confirmed_actions": None, "executed_actions": 0},
    ],
)
def test_direct_joint_tools_reject_invalid_counts_without_replaying(
    gripper, tool, arguments, invalid_counts
):
    primitives, env, model = gripper
    before = env.get_observation()
    env.post_finger_qpos = [0.01, 0.01]
    env.feedback = {
        "issued_actions": 1,
        "confirmed_actions": 1,
        "unconfirmed_actions": 0,
        "executed_actions": 1,
        "observation_after_last_issued_action": True,
        "stop_reason": "chunk_completed",
        **invalid_counts,
    }
    result = getattr(primitives, tool)(arm="left", steps=1, settle_steps=0, **arguments)
    assert result.is_error
    assert result.data["requested_actions"] == 1
    assert result.data["code"] == "invalid_execution_feedback"
    assert result.data["stop_reason"] == "invalid_execution_feedback"
    for name in ACTION_COUNT_FIELDS:
        assert result.data[name] is None
    assert result.data["observation_fresh"] is False
    assert result.data["observation_after_last_issued_action"] is False
    if tool == "rotate_wrist_joint":
        assert result.data["measured_delta_deg"] is None
    else:
        assert result.data["finger_target_position_reached"] is None
        assert result.data["settled"] is None
    np.testing.assert_array_equal(
        primitives.last_observation["r1::proprio"], before["r1::proprio"]
    )
    rejected = primitives.release(arm="right", steps=1, settle_steps=0)
    assert rejected.data["code"] == "command_state_unknown"
    _assert_not_issued(rejected)
    assert len(env.actions) == 1 and model.calls == 0


@pytest.mark.parametrize(
    "tool,arguments",
    [
        ("set_gripper", {"val": 1}),
        ("release", {}),
        ("rotate_wrist_joint", {"joint": "yaw", "delta_deg": 1}),
    ],
)
@pytest.mark.parametrize(
    "counts", [(None, None, None, None), (1, None, 0, 1), (1, 0, 1, None)]
)
def test_direct_joint_tools_preserve_consistent_partial_or_missing_counts(
    gripper, tool, arguments, counts
):
    primitives, env, _ = gripper
    names = ACTION_COUNT_FIELDS
    env.feedback = dict(zip(names, counts), observation_after_last_issued_action=False)
    result = getattr(primitives, tool)(arm="left", steps=1, settle_steps=0, **arguments)
    assert not result.is_error
    assert tuple(result.data[name] for name in names) == counts
    assert result.data["requested_actions"] == 1
    assert len(env.actions) == 1


@pytest.mark.parametrize("arm", ["left", "right"])
@pytest.mark.parametrize("joint", ["roll", "pitch", "yaw"])
def test_wrist_joint_moves_only_requested_joint_and_preserves_hold(gripper, arm, joint):
    primitive, backend, model = gripper
    primitive.vla_act()
    before = backend.raw["r1::proprio"].copy()
    result = primitive.rotate_wrist_joint(
        arm=arm, joint=joint, delta_deg=30, steps=5, settle_steps=2
    )
    index = (7 if arm == "left" else 15) + WRIST_JOINT_OFFSET[joint]
    state_index = (3 if arm == "left" else 28) + WRIST_JOINT_OFFSET[joint]
    ramp = before[state_index] + math.radians(30) * np.arange(1, 6) / 5
    expected = np.tile(_held_action(before, [-0.6, 0.6]), (7, 1))
    expected[:5, index] = ramp
    expected[5:, index] = ramp[-1]
    np.testing.assert_allclose(backend.actions[-1], expected, atol=1e-6)
    assert not result.is_error
    assert result.data["issued_actions"] == result.data["confirmed_actions"] == 7
    assert model.calls == 1 and len(backend.actions) == 2


class _MotionEnvironment:
    def __init__(self):
        self.raw = _robot_observation()
        proprio = self.raw["r1::proprio"]
        proprio[17:20] = [0.4, 0.2, 0.8]
        proprio[42:45] = [0.4, -0.2, 0.8]
        proprio[24:26] = proprio[49:51] = 0.05
        self.actions = []
        self.feedback = None
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
        return {"ended": False}

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
        info = _execution_feedback(len(actions), self.feedback)
        issued, confirmed = info.get("issued_actions"), info.get("confirmed_actions")
        if type(issued) is int and 0 <= issued <= len(actions):
            self.issued += issued
        if type(confirmed) is int and 0 <= confirmed <= len(actions):
            for action in actions[:confirmed]:
                self.previous_proprio = self.raw["r1::proprio"].copy()
                self.raw["r1::proprio"][:3] = action[:3] * BASE_OUTPUT_SCALE
                self.raw["r1::proprio"][3:10] = action[7:14]
                self.raw["r1::proprio"][28:35] = action[15:22]
                self.observation_index += 1
                self.confirmed += 1
        snapshot = self.get_motion_snapshot()
        info["observation_token"] = snapshot["observation_token"]
        info["motion_context"] = snapshot["motion_context"]
        returned = self.get_observation()
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


class _MotionVLA:
    def __init__(self):
        self.calls = 0
        self.grippers = [1, 1]

    def predict(self, observation):
        self.calls += 1
        proprio = observation["r1::proprio"]
        actions = np.zeros((2, 23), dtype=np.float32)
        actions[:, 3:7] = proprio[53:57]
        actions[:, 7:14] = proprio[3:10]
        actions[:, 15:22] = proprio[28:35]
        actions[:, [14, 22]] = self.grippers
        return actions


def _motion_runtime(cancelled=lambda: None):
    backend, planner, model = _MotionEnvironment(), _MotionPlanner(), _MotionVLA()
    rpc = LocalRpc(BehaviorEnvFacade(backend, task_id=40, motion_planner=planner))
    primitives = _make_primitives(BehaviorEnvClient(rpc, task_id=40), model, cancelled)
    return primitives, backend, planner, rpc, model


@pytest.fixture
def motion():
    return _motion_runtime()


def _establish_prior_hold(motion):
    """Establish a confirmed command through the public gripper tool."""
    primitives, backend, _, _, _ = motion
    result = primitives.set_gripper(arm="left", val=1, steps=1, settle_steps=0)
    assert not result.is_error
    backend.actions.clear()
    backend.received_tokens.clear()


def _motion_target(backend, arm="left"):
    start = 17 if arm == "left" else 42
    return backend.raw["r1::proprio"][start : start + 3].tolist()


def _follow_planned_pose(backend, planner):
    """Return the requested EEF pose after the fake path is confirmed."""
    step = backend.chunk_step

    def following_step(actions, **kwargs):
        reply = step(actions, **kwargs)
        if planner.calls and reply[4].get("confirmed_actions") == len(actions):
            arm, target, _ = planner.calls[-1]
            start = 17 if arm == "left" else 42
            backend.raw["r1::proprio"][start : start + 7] = target
        snapshot = backend.get_motion_snapshot()
        reply[4].update(
            observation_token=snapshot["observation_token"],
            motion_context=snapshot["motion_context"],
        )
        return backend.get_observation(), *reply[1:]

    backend.chunk_step = following_step


@pytest.fixture
def gripper():
    backend, model = _GripperEnvironment(), _GripperModel()
    return _make_primitives(backend, model), backend, model


@pytest.mark.parametrize(
    "tool,relative,explicit_orientation,expected_relax",
    [
        ("move_to", False, False, True),
        ("move_to", False, True, False),
        ("move_eef_delta", True, False, True),
    ],
)
def test_initial_sampling_preserves_requested_orientation_policy(
    motion, monkeypatch, tool, relative, explicit_orientation, expected_relax
):
    primitives, backend, planner, _, _ = motion
    options = []
    plan = planner.plan_arm_path

    def capture(**kwargs):
        options.append(kwargs["allow_orientation_relax"])
        return plan(**kwargs)

    monkeypatch.setattr(planner, "plan_arm_path", capture)
    _follow_planned_pose(backend, planner)
    arguments = (
        {"delta_xyz": [0.01, 0, 0]} if relative else {"xyz": _motion_target(backend)}
    )
    if explicit_orientation:
        arguments["quat"] = [0, 0, 0, 1]
    result = getattr(primitives, tool)(arm="left", **arguments)
    assert not result.is_error, result.data
    assert options == [expected_relax]
    assert [len(chunk) for chunk in backend.actions] == [1, 4]
    assert result.data["confirmed_actions"] == 5


def test_arm_path_holds_the_final_planning_state_after_settling(motion, monkeypatch):
    primitives, backend, planner, _, _ = motion
    _establish_prior_hold(motion)
    backend.raw["r1::proprio"][53] += 0.03
    final_start = []

    def settle(state):
        backend.raw["r1::proprio"][53] += 0.02
        backend.raw["r1::proprio"][28:35] += 0.01
        backend.previous_proprio = backend.raw["r1::proprio"].copy()
        final_start.append(backend.previous_proprio.copy())
        return {"status": "stationary", "attempted": 0, "confirmed": 0}

    monkeypatch.setattr(primitives, "_auto_settle_before_planning", settle)
    _follow_planned_pose(backend, planner)
    result = primitives.move_to(arm="left", xyz=_motion_target(backend))
    assert not result.is_error, result.data
    assert len(final_start) == 1
    state = final_start[0]
    np.testing.assert_array_equal(planner.calls[0][2]["r1::proprio"], state)
    commands = backend.actions[-1]
    np.testing.assert_allclose(commands[:, 3:7], np.tile(state[53:57], (4, 1)))
    np.testing.assert_allclose(commands[:, 15:22], np.tile(state[28:35], (4, 1)))


@pytest.mark.parametrize("tool", ["set_gripper", "release"])
@pytest.mark.parametrize("steps", [5001, 10**9])
def test_gripper_schema_rejects_unbounded_steps(tool, steps):
    arguments = {"arm": "left", "steps": steps, "val": 1}
    with pytest.raises(ValueError):
        getattr(BehaviorPrimitives, tool).args_schema.model_validate(arguments)


@pytest.mark.parametrize("tool", ["set_gripper", "release"])
@pytest.mark.parametrize("steps,settle_steps", [(5001, 0), (5000, 1)])
def test_gripper_budget_is_checked_before_action_allocation(
    gripper, monkeypatch, tool, steps, settle_steps
):
    primitives, env, _ = gripper

    def forbidden_allocation(*args, **kwargs):
        raise AssertionError("oversized gripper request reached action allocation")

    monkeypatch.setattr(np, "tile", forbidden_allocation)
    arguments = {"arm": "left", "steps": steps, "settle_steps": settle_steps, "val": 1}
    with pytest.raises(ValueError, match="5000|budget"):
        getattr(primitives, tool)(**arguments)
    assert env.actions == []


@pytest.mark.parametrize("tool", ["set_gripper", "release"])
def test_gripper_exact_budget_is_accepted(gripper, tool):
    primitives, env, _ = gripper
    arguments = {"arm": "left", "steps": 4990, "settle_steps": 10, "val": 1}
    result = getattr(primitives, tool)(**arguments)
    assert not result.is_error
    assert env.actions[-1].shape == (5000, 23)
    assert result.data["confirmed_actions"] == 5000


def test_new_hold_still_rejects_asymmetric_fingers_without_command_history(gripper):
    primitives, env, _ = gripper
    env.raw["r1::proprio"][24:26] = [0.01, 0.04]
    result = primitives.set_gripper(arm="left", val=1, steps=1, settle_steps=0)
    assert result.data["code"] == "invalid_hold_state"
    assert result.data["issued_actions"] == 0
    assert env.actions == []
