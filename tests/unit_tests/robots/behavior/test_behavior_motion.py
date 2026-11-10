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

"""R1Pro motion model, CuRobo IK backend and arm/base motion tools."""

from __future__ import annotations

import json
import shutil
import sys
import tempfile
import unittest
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from pathlib import Path
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest
from pydantic import ValidationError
from scipy.spatial.transform import Rotation

from robots.behavior.env_client import BehaviorEnvClient
from robots.behavior.env_server import BehaviorEnvFacade
from robots.behavior.eval.evaluator import BridgeCancelled, EvaluatorStepBridge
from robots.behavior.motion import R1ProMotionModel, R1ProMotionPlanner
from robots.behavior.primitives import (
    BASE_OUTPUT_SCALE,
    CONTROL_DT_S,
    BehaviorPrimitives,
)
from robots.behavior.toolkit import BehaviorToolkit
from rpent.dashboard.events import NullDashboardEventSink
from rpent.memory import MemoryManager
from rpent.tools import iter_tools
from rpent.tools.toolkit import ToolCancelled
from rpent.utils.rpc.rpc_client import RpcError
from tests.unit_tests.robots.behavior.conftest import (
    _establish_prior_hold,
    _follow_planned_pose,
    _motion_target,
    _refresh_reply,
)


class TestBehaviorMotionModel(unittest.TestCase):
    def setUp(self):
        self.model = R1ProMotionModel()
        self.state = np.zeros(61)
        self.state[24:26] = [0.02, 0.03]
        self.state[49:51] = [0.035, 0.045]

    def test_zero_pose_and_sourced_eef_frame(self):
        expected = {
            "left": [-0.07951818, 0.25199935, 0.65659],
            "right": [-0.07951906, -0.25200065, 0.65659],
        }
        for arm in ("left", "right"):
            pose = self.model.forward(arm, self.state)
            np.testing.assert_allclose(pose["position"], expected[arm], atol=1e-10)
            np.testing.assert_allclose(
                Rotation.from_quat(pose["quaternion_xyzw"]).as_matrix(),
                np.diag([-1, 1, -1]),
                atol=1e-12,
            )
            self.assertEqual(pose["frame"], "robot_base")

    def test_torso_joint3_uses_urdf_negative_axis_once(self):
        self.state[55] = 0.3
        pivot = np.asarray([-0.079032, 0.0001005, 1.04265])
        old = np.asarray([-0.07951818, 0.25199935, 0.65659])
        rotation = Rotation.from_euler("y", -0.3)
        pose = self.model.forward("left", self.state)
        np.testing.assert_allclose(
            pose["position"], pivot + rotation.apply(old - pivot), atol=1e-10
        )
        np.testing.assert_allclose(
            Rotation.from_quat(pose["quaternion_xyzw"]).as_matrix(),
            rotation.as_matrix() @ np.diag([-1, 1, -1]),
            atol=1e-12,
        )

    def test_physical_joint_map_uses_public_positions(self):
        values = self.model.joint_positions(np.arange(61, dtype=float))
        assert len(values) == 22
        assert values["torso_joint1"] == 53
        assert values["left_arm_joint1"] == 3
        assert values["right_arm_joint7"] == 34
        assert values["left_gripper_finger_joint2"] == 25
        assert values["right_gripper_finger_joint1"] == 49

    def test_single_arm_config_locks_all_other_observed_joints(self):
        self.state[53:57] = [0.4, -0.5, -0.1, 0.1]
        self.state[3:10] = [-1, 0.4, 0.1, -0.7, 0.05, 0.2, 0.1]
        self.state[28:35] = [-0.8, -0.3, 0.2, -0.6, 0.1, 0.05, -0.1]
        values = self.model.joint_positions(self.state)
        for arm in ("left", "right"):
            config = self.model.robot_config(arm, self.state)["kinematics"]
            active = config["cspace"]["joint_names"]
            self.assertEqual(active, list(self.model.active_joint_names(arm)))
            self.assertEqual(len(active), 7)
            self.assertEqual(len(config["lock_joints"]), 15)
            self.assertEqual(set(active) | set(config["lock_joints"]), set(values))
            for name, q in config["lock_joints"].items():
                self.assertEqual(q, values[name])
            self.assertEqual(config["base_link"], "base_link")
            self.assertFalse(config["use_usd_kinematics"])
            self.assertNotIn("usd_flip_joints", config)
            self.assertNotIn("usd_flip_joint_limits", config)
            self.assertIn("left_arm_link7", config["collision_link_names"])
            self.assertIn("right_arm_link7", config["collision_link_names"])
            self.assertEqual(config["ee_link"], f"{arm}_eef_link")
            self.assertEqual(
                config["cspace"]["retract_config"], [values[name] for name in active]
            )
            json.dumps(config, allow_nan=False)

    def test_config_is_fresh_and_arm_limits_are_asymmetric(self):
        config = self.model.robot_config("left", self.state)
        config["kinematics"]["collision_spheres"]["base_link"].clear()
        self.assertTrue(
            self.model.robot_config("left", self.state)["kinematics"][
                "collision_spheres"
            ]["base_link"]
        )
        left = self.model.joint_limits("left")
        right = self.model.joint_limits("right")
        np.testing.assert_array_equal(left[1], [-0.1745, 3.1416])
        np.testing.assert_array_equal(right[1], [-3.1416, 0.1745])
        self.assertEqual(self.model.velocity_limits("left").shape, (7,))
        self.assertTrue(np.all(self.model.velocity_limits("left") > 0))
        left[:] = 0
        self.assertNotEqual(self.model.joint_limits("left")[1, 0], 0)

    def test_opposite_arm_and_finger_links_follow_observed_positions(self):
        before = self.model.link_transforms(self.state)
        self.state[28] = -0.5
        self.state[25] += 0.01
        after = self.model.link_transforms(self.state)
        np.testing.assert_allclose(before["left_eef_link"], after["left_eef_link"])
        self.assertGreater(
            np.linalg.norm(before["right_eef_link"] - after["right_eef_link"]), 0.1
        )
        # Second finger has negative local Y axis and positive measured opening.
        np.testing.assert_allclose(
            after["left_gripper_finger_link2"][:3, 3]
            - before["left_gripper_finger_link2"][:3, 3],
            [0, -0.01, 0],
            atol=1e-12,
        )

    def test_hold_candidates_are_new_symmetric_targets_not_changes_to_measured_geometry(
        self,
    ):
        self.state[24:26] = [0.0124, 0.0126]
        self.state[49:51] = [0.0375, 0.0375]
        before = self.state.copy()
        commands = self.model.gripper_hold_candidates(self.state)
        self.assertAlmostEqual(commands["left"], -0.5)
        self.assertAlmostEqual(commands["right"], 0.5)
        np.testing.assert_array_equal(self.state, before)
        locked = self.model.robot_config("left", self.state)["kinematics"][
            "lock_joints"
        ]
        self.assertEqual(locked["left_gripper_finger_joint1"], 0.0124)
        self.assertEqual(locked["left_gripper_finger_joint2"], 0.0126)

    def test_wheel_enclosure_covers_every_sampled_unobserved_angle(self):
        # Native radius .075, local offset z=-.005; arbitrary axle/steer angles.
        for enclosure in self.model.provenance["wheel_enclosures"]:
            center = np.asarray(enclosure["center"])
            for steer in np.linspace(-np.pi, np.pi, 7):
                for wheel in np.linspace(-np.pi, np.pi, 9):
                    offset = Rotation.from_euler("z", steer).apply(
                        Rotation.from_euler("y", wheel).apply([0, 0, -0.005])
                    )
                    outer_extent = np.linalg.norm(center + offset - center) + 0.075
                    self.assertLessEqual(outer_extent, enclosure["radius"] + 1e-12)

    def test_public_fk_comparison_handles_quaternion_sign(self):
        self.state[53:57] = [0.4, -0.5, -0.1, 0.1]
        for arm, start in (("left", 17), ("right", 42)):
            pose = self.model.forward(arm, self.state)
            self.state[start : start + 3] = pose["position"]
            self.state[start + 3 : start + 7] = -np.asarray(pose["quaternion_xyzw"])
        self.state[17] += 0.012
        comparison = self.model.compare_proprio(self.state)
        self.assertAlmostEqual(comparison["left"]["position_error_m"], 0.012)
        self.assertLess(comparison["right"]["position_error_m"], 1e-12)
        self.assertLess(comparison["left"]["orientation_error_rad"], 1e-12)
        self.assertLess(comparison["right"]["orientation_error_rad"], 1e-12)

    def test_motion_assets_have_no_virtual_base_or_mesh_dependencies(self):
        text = self.model.urdf_path.read_text()
        for forbidden in ("<mesh", "base_footprint", "steer_motor", "wheel_motor"):
            self.assertNotIn(forbidden, text)
        self.assertNotIn(
            "attached_object", json.dumps(self.model.robot_config("left", self.state))
        )

    def test_changed_asset_is_rejected_before_use(self):
        with tempfile.TemporaryDirectory() as temp:
            target = Path(temp) / "assets"
            shutil.copytree(self.model.asset_dir, target)
            with (target / "r1pro_base.urdf").open("a") as stream:
                stream.write("\n")
            with self.assertRaisesRegex(ValueError, "SHA256"):
                R1ProMotionModel(target)

    def test_bad_public_input_and_arm_are_rejected(self):
        for value in ([0] * 23, [np.nan] * 61, [True] * 61):
            with self.assertRaisesRegex(ValueError, "61 finite"):
                self.model.robot_config("left", value)
        with self.assertRaisesRegex(ValueError, "arm must"):
            self.model.robot_config("both", self.state)
        with self.assertRaisesRegex(ValueError, "quaternion"):
            self.model.compare_proprio(self.state)


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
def planning_request():
    planner = R1ProMotionPlanner(task_id=40, task_language="make popcorn")
    proprio = np.zeros(61)
    proprio[24:26] = 0.05
    proprio[49:51] = 0.025
    for arm, offset in (("left", 17), ("right", 42)):
        pose = planner.model.forward(arm, proprio)
        proprio[offset : offset + 3] = pose["position"]
        proprio[offset + 3 : offset + 7] = pose["quaternion_xyzw"]
    raw = {
        "r1::proprio": proprio,
        "r1::cam_rel_poses": np.tile([0, 0, 0, 0, 0, 0, 1], 3),
    }
    for camera in ("zed_link", "left_realsense_link", "right_realsense_link"):
        raw[f"r1::{camera}:Camera:0::rgb"] = np.zeros((224, 224, 3), np.uint8)
        raw[f"r1::{camera}:Camera:0::depth_linear"] = np.ones((224, 224))
    return planner, raw, proprio[17:24].copy()


def test_planner_rejects_missing_control_history_before_solver(
    planning_request, monkeypatch
):
    planner, raw, target = planning_request
    monkeypatch.setattr(
        planner, "_ik_solver", lambda *args: pytest.fail("no GPU expected")
    )
    result = planner.plan_arm_path(arm="left", target_pose=target, observation=raw)
    assert result["code"] == "motion_history_unavailable"


def test_confirmed_positions_override_residual_raw_joint_velocity_without_mutation():
    state = np.zeros(61)
    state[[10, 35, 57, 26, 51]] = 0.1
    previous = state.copy()
    previous[3] -= 0.00001
    context = motion_context(state, previous)
    before = state.copy()
    contract = R1ProMotionModel.planning_state_contract(state, context)
    assert contract["valid"] and contract["stationary"]
    assert contract["velocity_groups"]["left_arm"]["measured_max_abs"] == pytest.approx(
        0.0003
    )
    assert contract["raw_velocity_groups"]["left_arm"]["measured_max_abs"] == 0.1
    np.testing.assert_array_equal(state, before)
    np.testing.assert_array_equal(context["current_proprio"], before)


@pytest.mark.parametrize(
    "index,group",
    [
        (53, "trunk"),
        (3, "left_arm"),
        (28, "right_arm"),
        (24, "left_fingers"),
        (49, "right_fingers"),
    ],
)
def test_position_change_rejects_false_static_raw_velocity(index, group):
    state, previous = np.zeros(61), np.zeros(61)
    state[index] = 0.01
    contract = R1ProMotionModel.planning_state_contract(
        state, motion_context(state, previous)
    )
    assert contract["valid"] and not contract["stationary"]
    assert contract["unsettled_groups"] == [group]


def test_base_still_uses_public_velocity_without_base_pose_history():
    state = np.zeros(61)
    state[0] = 0.02
    contract = R1ProMotionModel.planning_state_contract(
        state, motion_context(state, state)
    )
    assert contract["unsettled_groups"] == ["base"]
    assert contract["velocity_groups"]["base"]["source"] == "public_base_qvel"


@pytest.mark.parametrize(
    "patch",
    [
        {"control_dt_s": 1 / 120},
        {"control_dt_s": 0.5},
        {"pair_confirmed": False},
        {"previous_token": {"generation": 1, "observation_index": 0}},
        {"previous_token": {"generation": 0, "observation_index": 1}},
        {"current_token": {"generation": 0, "observation_index": 3}},
        {"unconfirmed_actions": 1},
        {"actions_pending": True},
        {"previous_proprio": None},
        {"initial_bootstrap": True},
    ],
)
def test_motion_history_rejects_gap_wrong_generation_or_unknown_execution(patch):
    state = np.zeros(61)
    context = {**motion_context(state, state), **patch}
    contract = R1ProMotionModel.planning_state_contract(state, context)
    assert not contract["valid"] and not contract["stationary"]


def test_initial_bootstrap_cannot_be_reused_after_an_action_or_for_another_frame():
    state = np.zeros(61)
    context = motion_context(state)
    assert R1ProMotionModel.planning_state_contract(state, context)["valid"]
    context["issued_actions"] = context["confirmed_actions"] = 1
    assert not R1ProMotionModel.planning_state_contract(state, context)["valid"]
    context = motion_context(state, state)
    state[3] = 0.1
    assert not R1ProMotionModel.planning_state_contract(state, context)["valid"]


@pytest.mark.parametrize(
    ("index", "velocity", "group", "code"),
    [
        (0, 0.02, "base", "base_not_stationary"),
        (57, 0.03, "trunk", "motion_state_unsettled"),
        (10, 0.03, "left_arm", "motion_state_unsettled"),
        (35, -0.03, "right_arm", "motion_state_unsettled"),
        (26, 0.003, "left_fingers", "motion_state_unsettled"),
        (51, -0.01, "right_fingers", "motion_state_unsettled"),
    ],
)
def test_nonstatic_geometry_is_rejected_before_native_solver_allocation(
    planning_request, monkeypatch, index, velocity, group, code
):
    planner, raw, target = planning_request
    raw["r1::proprio"][index] = velocity
    monkeypatch.setattr(
        planner, "_ik_solver", lambda *args: pytest.fail("must reject before GPU setup")
    )
    result = planner.plan_arm_path(
        arm="left",
        target_pose=target,
        observation=raw,
        motion_context=motion_context(raw["r1::proprio"]),
    )
    assert result["status"] == "Unavailable" and result["code"] == code
    assert result["motion_state_contract"]["unsettled_groups"] == [group]


def test_other_fingers_that_cannot_share_one_hold_target_are_rejected(
    planning_request, monkeypatch
):
    planner, raw, target = planning_request
    raw["r1::proprio"][49:51] = [0.02, 0.03]
    monkeypatch.setattr(
        planner, "_ik_solver", lambda *args: pytest.fail("must reject before GPU setup")
    )
    result = planner.plan_arm_path(
        arm="left",
        target_pose=target,
        observation=raw,
        motion_context=motion_context(raw["r1::proprio"]),
    )
    assert result["code"] == "gripper_hold_unrepresentable"
    assert result["motion_state_contract"]["stationary"] is True


def test_native_plan_preserves_measured_fingers_and_reports_collision_scope(
    planning_request, monkeypatch
):
    planner, raw, target = planning_request
    proprio = raw["r1::proprio"]
    proprio[24:26] = 0.02
    proprio[49:51] = [0.0248, 0.0252]
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
    assert plan["held_finger_qpos"] == {"left": [0.02, 0.02], "right": [0.0248, 0.0252]}
    assert plan["held_gripper_native_commands"] == pytest.approx(
        {"left": -0.2, "right": 0.0}
    )
    assert plan["gripper_hold_source"] == "candidate_from_current_measurement"
    assert plan["held_objects_modeled"] is plan["collision_checked"] is False
    assert plan["collision_scope"]["source"] == "none_ik_only"
    np.testing.assert_array_equal(proprio, before)


def _fake_curobo(monkeypatch):
    modules = {
        "curobo.types.math": {"Pose": SimpleNamespace},
        "curobo.rollout.cost.pose_cost": {
            "PoseCostMetric": lambda **kwargs: SimpleNamespace(**kwargs)
        },
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


class _Tensor:
    def __init__(self, values):
        self.values = np.asarray(values)

    def detach(self):
        return self

    def cpu(self):
        return self

    def numpy(self):
        return self.values


class _IKSolver:
    """CPU IK double with an optional position-only solution."""

    tensor_args = SimpleNamespace(to_device=np.asarray)

    def __init__(self, target, start_q, relaxed_wxyz=None):
        self.target = target
        self.target_q = start_q
        self.relaxed_wxyz = relaxed_wxyz
        self.metric = None
        self.calls = []

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
            )
        return SimpleNamespace(
            ee_position=_Tensor(self.target[:3]),
            ee_quaternion=_Tensor(self.relaxed_wxyz),
        )

    def solve_single(self, goal, seed_config=None, return_seeds=1):
        relaxed = self.metric is not None
        self.calls.append("relaxed" if relaxed else "full")
        goal_q = np.asarray(seed_config).reshape(7) + (0.002 if relaxed else 0.001)
        success = relaxed or self.relaxed_wxyz is None
        return SimpleNamespace(
            success=np.asarray([success]),
            position_error=np.asarray([0.0 if success else 0.04]),
            rotation_error=np.asarray([0.0 if success else 0.3]),
            js_solution=SimpleNamespace(
                get_ordered_joint_state=lambda names: SimpleNamespace(
                    position=_Tensor(goal_q[None, :])
                )
            ),
        )


def test_relaxed_orientation_retry_is_reported_not_hidden(
    planning_request, monkeypatch
):
    planner, raw, target = planning_request
    _fake_curobo(monkeypatch)
    # A wrist turned 90 degrees about z relative to the requested orientation.
    turned = (
        Rotation.from_euler("z", 90, degrees=True) * Rotation.from_quat(target[3:])
    ).as_quat()
    solver = _IKSolver(target, raw["r1::proprio"][3:10], turned[[3, 0, 1, 2]])
    monkeypatch.setattr(planner, "_ik_solver", lambda *args: solver)
    context = motion_context(raw["r1::proprio"])

    strict = planner.plan_arm_path(
        arm="left", target_pose=target, observation=raw, motion_context=context
    )
    assert strict["status"] == "Failed" and strict["code"] == "curobo_ik_failed"
    assert strict["orientation_relax_attempted"] is False
    assert solver.calls == ["full"]
    # The retreat suggestion keeps height and pulls the hand toward the base.
    current = raw["r1::proprio"][17:20]
    retreat = np.asarray(strict["suggested_retreat_xyz"])
    assert retreat[2] == pytest.approx(current[2])
    assert np.linalg.norm(retreat[:2]) < np.linalg.norm(current[:2])

    solver.calls.clear()
    solver.metric = None
    relaxed = planner.plan_arm_path(
        arm="left",
        target_pose=target,
        observation=raw,
        motion_context=context,
        allow_orientation_relax=True,
    )
    assert solver.calls == ["full", "relaxed"]
    assert relaxed["status"] == "Success"
    assert relaxed["orientation_relaxed"] is True
    assert relaxed["ik_orientation_error_rad"] == pytest.approx(np.pi / 2, abs=1e-6)
    assert relaxed["target_pose"] == target.tolist()


@pytest.mark.parametrize("arm", ["left", "right"])
def test_native_planner_path_is_sent_unchanged_with_unplanned_components_held(
    motion, arm
):
    primitives, backend, planner, rpc, model = motion
    primitives.vla_act()
    before = backend.raw["r1::proprio"].copy()
    result = primitives.move_to(
        arm=arm, xyz=_motion_target(backend, arm), settle_steps=0
    )
    assert not result.is_error
    assert len(planner.calls) == 1 and model.calls == 1
    assert [len(chunk) for chunk in backend.actions] == [2, 4]
    assert "env.plan_arm_path" in rpc.calls
    actions = backend.actions[-1]
    moving = slice(7, 14) if arm == "left" else slice(15, 22)
    observed = slice(3, 10) if arm == "left" else slice(28, 35)
    other = slice(15, 22) if arm == "left" else slice(7, 14)
    other_observed = slice(28, 35) if arm == "left" else slice(3, 10)
    assert actions.shape == (4, 23)
    np.testing.assert_allclose(actions[:, :3], 0)
    np.testing.assert_allclose(actions[:, 3:7], np.tile(before[53:57], (4, 1)))
    np.testing.assert_allclose(
        actions[:, other], np.tile(before[other_observed], (4, 1))
    )
    np.testing.assert_allclose(actions[:, [14, 22]], 1)
    np.testing.assert_allclose(
        actions[:, moving], before[observed] + np.arange(1, 5)[:, None] * 0.005
    )
    assert result.data["target_reached"] is True
    assert result.data["issued_actions"] == result.data["confirmed_actions"] == 4
    assert result.data["unconfirmed_actions"] == 0
    assert result.data["actual_settle_steps"] == 0
    assert result.data["stop_reason"] == "target_reached"
    assert result.data["completed"] is True
    assert "success" not in result.data
    assert "env.plan_arm_path" not in rpc.facade._readonly_methods


def test_robot_base_xyzw_is_explicit_and_not_converted_to_world_or_wxyz(motion):
    primitives, backend, planner, _, _ = motion
    primitives.vla_act()
    quat = [0, 0, np.sqrt(0.5), np.sqrt(0.5)]
    result = primitives.move_to(
        arm="left", xyz=_motion_target(backend), quat=quat, settle_steps=3
    )
    np.testing.assert_allclose(planner.calls[-1][1], [0.4, 0.2, 0.8, *quat])
    assert result.data["orientation_error_rad"] == pytest.approx(np.pi / 2)
    assert result.data["target_reached"] is False


def test_registered_motion_default_matches_immediate_pose_completion(motion):
    primitives, _, _, _, _ = motion
    declaration = {item.name: item for item in iter_tools(primitives)}["move_to"]
    assert declaration.input_schema["properties"]["settle_steps"]["default"] == 0
    assert "completion gate" in declaration.description


@pytest.mark.parametrize("finger_velocity", [0.0, -0.01])
def test_opposite_gripper_close_command_is_carried_into_the_plan(
    motion, finger_velocity
):
    """A closing command on the other hand does not block planning.

    Command-versus-measurement disagreement is expected under contact (a held
    object keeps the fingers open); the plan carries the current command.
    """
    primitives, backend, planner, _, model = motion
    original = model.predict

    def closing_actions(observation):
        actions = original(observation)
        actions[:, 22] = -1
        return actions

    model.predict = closing_actions
    primitives.vla_act()
    backend.raw["r1::proprio"][51:53] = finger_velocity
    result = primitives.move_to(arm="left", xyz=_motion_target(backend))
    assert result.data.get("code") != "gripper_command_unsettled"
    assert planner.calls, "the request must reach the planner"


@pytest.mark.parametrize("channel", [0, 1, 2])
def test_moving_base_cannot_change_frame_during_arm_execution(motion, channel):
    primitives, backend, planner, _, _ = motion
    primitives.vla_act()
    backend.raw["r1::proprio"][channel] = 0.011
    result = primitives.move_to(arm="left", xyz=_motion_target(backend))
    assert result.data["code"] == "base_not_stationary"
    # Residual base motion is first damped with bounded zero-base holds; the
    # fake base never settles, so the holds are spent and the plan is refused.
    holds = backend.actions[1:]
    assert 1 <= len(holds) <= 12 and all(np.all(a[:, :3] == 0) for a in holds)
    assert result.data["auto_settle"]["status"] == "still_unsettled"
    assert planner.calls == []


def test_target_outside_short_reposition_scope_never_reaches_solver(motion):
    primitives, backend, planner, _, _ = motion
    primitives.vla_act()
    target = np.asarray(_motion_target(backend)) + [0.301, 0, 0]
    result = primitives.move_to(arm="left", xyz=target.tolist())
    assert result.data["code"] == "target_outside_motion_scope"
    assert len(backend.actions) == 1 and planner.calls == []


@pytest.mark.parametrize("mismatch", ["command", "measured"])
def test_closed_gripper_plans_with_held_object_carried(motion, mismatch):
    """IK-only planning has no path collision model, so a held object is allowed."""
    primitives, backend, planner, _, _ = motion
    primitives.vla_act()
    if mismatch == "command":
        primitives._last_issued_action[14] = -1
        backend.raw["r1::proprio"][24:26] = 0.0
        backend.previous_proprio[24:26] = 0.0
    else:
        backend.raw["r1::proprio"][24:26] = 0.02
        backend.previous_proprio[24:26] = 0.02
        primitives._last_issued_action[14] = 2.0 * 0.02 / 0.05 - 1.0
    result = primitives.move_to(arm="left", xyz=_motion_target(backend))
    assert result.data.get("code") != "moving_gripper_not_open"
    assert planner.calls, "a closed gripper must reach the planner"


def test_missing_planner_backend_reports_unavailable_without_actions(motion):
    primitives, backend, _, rpc, _ = motion
    primitives.vla_act()
    rpc.facade._motion_planner = None
    result = primitives.move_to(arm="left", xyz=_motion_target(backend))
    assert result.data["code"] == "motion_planner_unavailable"
    assert len(backend.actions) == 1


@pytest.mark.parametrize(
    "bad",
    [
        {"position": []},
        {"position": np.zeros((2, 8))},
        {"position": np.full((2, 7), np.nan)},
        {"position": np.zeros((601, 7))},
        {"joint_names": [f"right_arm_joint{i}" for i in range(1, 8)]},
        {"quaternion_order": "wxyz"},
        {"frame": "world"},
        {"dt_s": 1 / 120},
        {"control_dt_s": None},
        {"dt_s": True},
        {"start_arm_qpos": np.zeros(7)},
        {"held_trunk_qpos": np.zeros(4)},
        {"held_other_arm_qpos": np.zeros(7)},
        {"motion_state_contract": {"stationary": False}},
        {"held_finger_qpos": {"left": [0, 0], "right": [0.05, 0.05]}},
        {"joint_position_limits_rad": np.tile([-0.1, 0.1], (7, 1))},
        {"joint_velocity_limits_rad_s": np.full(7, 0.001)},
        {"joint_velocity_limits_rad_s": np.full(7, -1)},
        {
            "fk_comparison": {
                "left": {"position_error_m": 0, "orientation_error_rad": 0}
            }
        },
    ],
)
def test_malformed_successful_plan_never_reaches_execution(motion, bad):
    primitives, backend, planner, _, _ = motion
    primitives.vla_act()
    planner.mutate = lambda result, observation: result.update(bad)
    result = primitives.move_to(arm="left", xyz=_motion_target(backend))
    assert result.is_error and result.data["code"] == "invalid_arm_plan"
    assert len(backend.actions) == 1


@pytest.mark.parametrize(
    "status,code", [("Failed", "unreachable"), ("Unavailable", "depth_missing")]
)
def test_native_planning_failure_preserves_error_and_does_not_fallback_to_vla(
    motion, status, code
):
    primitives, backend, planner, _, model = motion
    primitives.vla_act()
    planner.mutate = lambda result, observation: result.update(status=status, code=code)
    result = primitives.move_to(arm="left", xyz=_motion_target(backend))
    assert result.is_error and result.data["code"] == code
    assert model.calls == 1 and len(backend.actions) == 1


def test_plan_without_a_collision_scope_is_rejected(motion):
    """IK-only plans report no path check, but they must still describe their scope."""
    primitives, backend, planner, _, _ = motion
    primitives.vla_act()
    planner.mutate = lambda result, observation: result.update(collision_scope=None)
    result = primitives.move_to(arm="left", xyz=_motion_target(backend))
    assert result.data["code"] == "collision_scope_missing"
    assert len(backend.actions) == 1


def test_changed_public_observation_invalidates_planned_path(motion):
    primitives, backend, planner, _, _ = motion
    primitives.vla_act()

    def change_frame(result, observation):
        backend.raw["r1::proprio"][53] = 0.2
        backend.observation_index += 1
        backend.issued += 1
        backend.confirmed += 1

    planner.mutate = change_frame
    result = primitives.move_to(arm="left", xyz=_motion_target(backend))
    assert result.data["code"] == "stale_arm_plan"
    assert len(backend.actions) == 1


def test_new_generation_invalidates_plan_even_with_identical_joint_positions(motion):
    primitives, backend, planner, _, _ = motion
    primitives.vla_act()
    planner.mutate = lambda result, observation: setattr(backend, "generation", 1)
    result = primitives.move_to(arm="left", xyz=_motion_target(backend))
    assert result.is_error and result.data["code"] == "stale_arm_plan"
    assert len(backend.actions) == 1


def test_missing_confirmed_history_cannot_bootstrap_after_prior_actions(motion):
    primitives, backend, planner, _, _ = motion
    primitives.vla_act()
    backend.previous_proprio = None
    result = primitives.move_to(arm="left", xyz=_motion_target(backend))
    assert result.is_error and result.data["code"] == "motion_history_unavailable"
    assert not planner.calls and len(backend.actions) == 1


def test_shared_environment_cannot_invent_an_initial_gripper_command(motion):
    primitives, backend, planner, _, model = motion
    backend.chunk_step(model.predict(backend.get_observation()))
    result = primitives.move_to(arm="left", xyz=_motion_target(backend))
    assert result.is_error and result.data["code"] == "invalid_hold_state"
    assert "initial session snapshot" in result.data["error"]
    assert not planner.calls and len(backend.actions) == 1


def test_confirmed_position_history_replaces_raw_joint_speed_for_planning(motion):
    primitives, backend, planner, _, _ = motion
    primitives.vla_act()
    backend.raw["r1::proprio"][10] = 0.04
    result = primitives.move_to(arm="left", xyz=_motion_target(backend))
    assert result.data["completed"] is True
    assert planner.calls[0][2]["r1::proprio"][10] == 0.04
    assert backend.raw["r1::proprio"][10] == 0.04
    assert [len(actions) for actions in backend.actions] == [2, 4]


def test_real_joint_displacement_is_settled_then_planned(motion):
    """Residual arm motion is damped by counted holds instead of failing the call."""
    primitives, backend, planner, _, _ = motion
    primitives.vla_act()
    backend.raw["r1::proprio"][3] += 0.001
    result = primitives.move_to(arm="left", xyz=_motion_target(backend))
    settle = result.data["auto_settle"]
    assert settle["status"] == "stationary"
    assert settle["attempted"] >= 1 and settle["confirmed"] == settle["attempted"]
    assert planner.calls, "planning proceeds once the start is settled"
    assert result.data["completed"] is True


def test_move_to_can_return_control_to_vla_without_implicit_stationary_holds(motion):
    primitives, backend, _, _, model = motion
    _set_feedback_velocity_sequence(backend, [0.04])
    result = primitives.move_to(arm="left", xyz=_motion_target(backend))
    assert result.data["completed"] is True
    follow_up = primitives.vla_act()
    assert not follow_up.is_error and model.calls == 1
    assert [len(actions) for actions in backend.actions] == [1, 4, 2]


@pytest.mark.parametrize(
    "fault", ["generation", "observation_index", "proprio", "missing"]
)
def test_unbound_motion_feedback_keeps_counts_but_never_completes_or_continues(
    motion, fault
):
    _establish_prior_hold(motion)
    primitives, backend, planner, _, _ = motion
    original = backend.chunk_step

    def unbound_feedback(actions, **kwargs):
        result = original(actions, **kwargs)
        info = result[4]
        if fault == "missing":
            info.pop("observation_token")
        elif fault == "proprio":
            info["motion_context"]["current_proprio"][3] += 0.01
        else:
            info["observation_token"][fault] += 1
        return result

    backend.chunk_step = unbound_feedback
    result = primitives.move_to(arm="left", xyz=_motion_target(backend), settle_steps=3)
    assert result.is_error
    assert result.data["stop_reason"] == "motion_feedback_mismatch"
    assert result.data["issued_actions"] == result.data["confirmed_actions"] == 4
    assert result.data["completed"] is False
    assert result.data["target_reached"] is None
    assert len(backend.actions) == len(planner.calls) == 1
    follow_up = primitives.move_to(arm="left", xyz=_motion_target(backend))
    assert follow_up.data["code"] == "command_state_unknown"
    assert len(backend.actions) == 1


def test_session_closure_during_planning_does_not_send_path(motion):
    primitives, backend, planner, _, _ = motion
    primitives.vla_act()
    planner.mutate = lambda result, observation: setattr(backend, "ended", True)
    result = primitives.move_to(arm="left", xyz=_motion_target(backend))
    assert result.data["session_ended"] is True
    assert result.data["requested_actions"] == 0 and len(backend.actions) == 1


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


def test_unknown_execution_feedback_invalidates_gripper_command_state(motion):
    primitives, backend, _, _, _ = motion
    primitives.vla_act()
    backend.feedback = {"observation_after_last_issued_action": False}
    result = primitives.move_to(arm="left", xyz=_motion_target(backend))
    assert result.data["issued_actions"] is None
    assert result.data["target_reached"] is None
    assert (
        primitives.move_to(arm="left", xyz=_motion_target(backend)).data["code"]
        == "command_state_unknown"
    )


def _set_feedback_velocity_sequence(backend, values, *, index=57):
    original = backend.chunk_step
    calls = 0

    def measured_feedback(actions, **kwargs):
        nonlocal calls
        observation, reward, terminated, truncated, info = original(actions, **kwargs)
        observation["r1::proprio"][index] = values[min(calls, len(values) - 1)]
        info["motion_context"]["current_proprio"] = observation["r1::proprio"].copy()
        calls += 1
        return observation, reward, terminated, truncated, info

    backend.chunk_step = measured_feedback


def _set_feedback_pose_sequence(backend, offsets, *, arm="left"):
    original = backend.chunk_step
    calls = 0

    def measured_feedback(actions, **kwargs):
        nonlocal calls
        result = original(actions, **kwargs)
        result[0]["r1::proprio"][17 if arm == "left" else 42] += offsets[
            min(calls, len(offsets) - 1)
        ]
        result[4]["motion_context"]["current_proprio"] = result[0]["r1::proprio"].copy()
        calls += 1
        return result

    backend.chunk_step = measured_feedback


@pytest.mark.parametrize("arm", ["left", "right"])
def test_optional_holds_stop_on_first_confirmed_pose_without_waiting_for_low_speed(
    motion, arm
):
    _establish_prior_hold(motion)
    primitives, backend, planner, _, model = motion
    _set_feedback_velocity_sequence(backend, [0.025])
    _set_feedback_pose_sequence(backend, [0.006, 0.006, 0.0], arm=arm)
    result = primitives.move_to(
        arm=arm, xyz=_motion_target(backend, arm), settle_steps=10
    )
    assert not result.is_error and model.calls == 0 and len(planner.calls) == 1
    assert [len(actions) for actions in backend.actions] == [4, 1, 1]
    for hold in backend.actions[1:]:
        np.testing.assert_array_equal(hold, backend.actions[0][-1:])
    data = result.data
    assert data["actual_settle_steps"] == 2
    assert data["requested_actions"] == data["issued_actions"] == 6
    assert data["confirmed_actions"] == data["executed_actions"] == 6
    assert data["unconfirmed_actions"] == 0
    assert data["target_reached"] is data["completed"] is True
    assert data["stop_reason"] == "target_reached"
    assert backend.received_tokens == [
        {"generation": 0, "observation_index": index} for index in (1, 5, 6)
    ]


@pytest.mark.parametrize(("index", "budget"), [(57, 0), (10, 4)])
def test_residual_speed_is_diagnostic_without_delaying_pose_completion(
    motion, index, budget
):
    _establish_prior_hold(motion)
    primitives, backend, planner, _, _ = motion
    _set_feedback_velocity_sequence(backend, [0.04], index=index)
    result = primitives.move_to(
        arm="left", xyz=_motion_target(backend), settle_steps=budget
    )
    assert not result.is_error and len(planner.calls) == 1
    assert [len(actions) for actions in backend.actions] == [4]
    assert result.data["actual_settle_steps"] == 0
    assert result.data["requested_actions"] == result.data["confirmed_actions"] == 4
    assert result.data["target_reached"] is result.data["completed"] is True
    assert result.data["stop_reason"] == "target_reached"


@pytest.mark.parametrize(
    "budget,stop", [(0, "target_not_reached"), (2, "target_hold_budget_exhausted")]
)
def test_missed_pose_returns_error_even_with_stationary_feedback(motion, budget, stop):
    _establish_prior_hold(motion)
    primitives, backend, _, _, _ = motion
    _set_feedback_pose_sequence(backend, [0.006])
    result = primitives.move_to(
        arm="left", xyz=_motion_target(backend), settle_steps=budget
    )
    assert result.is_error
    assert result.data["target_reached"] is result.data["completed"] is False
    assert result.data["stop_reason"] == stop
    assert result.data["actual_settle_steps"] == budget
    assert [len(actions) for actions in backend.actions] == [4] + [1] * budget


def test_all_final_native_channels_are_held_not_rederived_from_observations(motion):
    primitives, backend, _, _, _ = motion
    backend.raw["r1::proprio"][49:51] = 0.03
    _establish_prior_hold(motion)
    original = backend.chunk_step

    def perturbed_hold_state(actions, **kwargs):
        returned, reward, terminated, truncated, info = original(actions, **kwargs)
        returned["r1::proprio"][53:57] += 0.001
        returned["r1::proprio"][28:35] += 0.001
        info["motion_context"]["current_proprio"] = returned["r1::proprio"].copy()
        return returned, reward, terminated, truncated, info

    backend.chunk_step = perturbed_hold_state
    _set_feedback_pose_sequence(backend, [0.006, 0.006, 0.0])
    result = primitives.move_to(arm="left", xyz=_motion_target(backend), settle_steps=3)
    assert result.data["completed"] is True
    assert result.data["actual_settle_steps"] == 2
    final_target = backend.actions[0][-1:]
    np.testing.assert_allclose(final_target[:, [14, 22]], [[1, 0.2]])
    np.testing.assert_array_equal(final_target[:, :3], 0)
    for hold in backend.actions[1:]:
        np.testing.assert_array_equal(hold, final_target)


def _feedback_on_hold(backend, replacement, *, hold_number=1, flags=(None, None)):
    # Exercise the explicitly requested correction path, not an implicit wait
    # after an already completed motion.
    _set_feedback_pose_sequence(backend, [0.006])
    original = backend.chunk_step
    calls = 0

    def changed_feedback(actions, **kwargs):
        nonlocal calls
        if calls == hold_number:
            backend.feedback = replacement
        result = original(actions, **kwargs)
        calls += 1
        if calls == hold_number + 1:
            result = (result[0], result[1], *flags, result[4])
        return result

    backend.chunk_step = changed_feedback


def test_partial_hold_preserves_known_totals_and_never_sends_another_action(motion):
    _establish_prior_hold(motion)
    primitives, backend, planner, _, _ = motion
    _feedback_on_hold(
        backend,
        {
            "issued_actions": 1,
            "confirmed_actions": 0,
            "unconfirmed_actions": 1,
            "executed_actions": 0,
            "observation_after_last_issued_action": False,
        },
    )
    result = primitives.move_to(arm="left", xyz=_motion_target(backend), settle_steps=3)
    data = result.data
    assert data["requested_actions"] == data["issued_actions"] == 5
    assert data["confirmed_actions"] == data["executed_actions"] == 4
    assert data["unconfirmed_actions"] == 1
    assert data["actual_settle_steps"] == 0
    assert data["stop_reason"] == "partial_execution"
    assert data["target_reached"] is None
    assert len(backend.actions) == 2 and len(planner.calls) == 1
    assert (
        primitives.move_to(arm="left", xyz=_motion_target(backend)).data["code"]
        == "command_state_unknown"
    )
    assert len(backend.actions) == 2


@pytest.mark.parametrize("flag,stop", [(True, "terminated"), (False, "truncated")])
def test_terminal_flag_during_hold_stops_even_with_complete_feedback(
    motion, flag, stop
):
    _establish_prior_hold(motion)
    primitives, backend, _, _, _ = motion
    _feedback_on_hold(backend, None, flags=(flag, not flag))
    result = primitives.move_to(arm="left", xyz=_motion_target(backend), settle_steps=3)
    assert result.data["requested_actions"] == result.data["confirmed_actions"] == 5
    assert result.data["stop_reason"] == stop
    assert result.data["completed"] is False
    assert len(backend.actions) == 2
    assert primitives.move_to(arm="left", xyz=_motion_target(backend)).is_error
    assert len(backend.actions) == 2


def test_confirmed_partial_path_never_starts_end_target_holds(motion):
    _establish_prior_hold(motion)
    primitives, backend, _, _, _ = motion
    backend.feedback = {
        "issued_actions": 2,
        "confirmed_actions": 2,
        "unconfirmed_actions": 0,
        "executed_actions": 2,
        "observation_after_last_issued_action": True,
    }
    result = primitives.move_to(arm="left", xyz=_motion_target(backend))
    assert len(backend.actions) == 1
    assert result.data["requested_actions"] == 4
    assert result.data["confirmed_actions"] == 2
    assert result.data["stop_reason"] == "partial_execution"


def test_unknown_hold_feedback_preserves_prefix_without_inventing_zero_counts(motion):
    _establish_prior_hold(motion)
    primitives, backend, _, _, _ = motion
    _feedback_on_hold(backend, {})
    result = primitives.move_to(arm="left", xyz=_motion_target(backend), settle_steps=3)
    assert result.data["requested_actions"] == 5 and len(backend.actions) == 2
    for key in (
        "issued_actions",
        "confirmed_actions",
        "unconfirmed_actions",
        "executed_actions",
    ):
        assert result.data[key] is None
    assert result.data["actual_settle_steps"] is None
    assert result.data["stop_reason"] == "execution_feedback_unknown"


def test_stale_hold_observation_does_not_reuse_previous_pose_or_streak(motion):
    _establish_prior_hold(motion)
    primitives, backend, _, _, _ = motion
    _feedback_on_hold(
        backend,
        {
            "issued_actions": 1,
            "confirmed_actions": 1,
            "unconfirmed_actions": 0,
            "executed_actions": 1,
            "observation_after_last_issued_action": False,
        },
        hold_number=2,
    )
    result = primitives.move_to(arm="left", xyz=_motion_target(backend), settle_steps=3)
    assert len(backend.actions) == 3
    assert result.data["confirmed_actions"] == 6
    assert result.data["stop_reason"] == "stale_observation"
    assert result.data["target_reached"] is None


@pytest.mark.parametrize(
    "error",
    [
        TimeoutError("late timeout"),
        RpcError("env.chunk_step", "lost response"),
        TypeError("BEHAVIOR execution must return a Gym 5-tuple"),
        ValueError("BEHAVIOR execution returned no observation frames"),
    ],
)
def test_later_transport_failure_keeps_known_prefix_but_latest_outcome_unknown(
    motion, error
):
    _establish_prior_hold(motion)
    primitives, backend, planner, _, _ = motion
    _set_feedback_pose_sequence(backend, [0.006])
    original = backend.chunk_step

    def fail_on_second_hold(actions, **kwargs):
        if len(backend.actions) == 2:
            backend.actions.append(actions.copy())
            raise error
        return original(actions, **kwargs)

    backend.chunk_step = fail_on_second_hold
    result = primitives.move_to(arm="left", xyz=_motion_target(backend), settle_steps=3)
    data = result.data
    assert len(backend.actions) == 3 and len(planner.calls) == 1
    assert data["requested_actions"] == 6
    for key in (
        "issued_actions",
        "confirmed_actions",
        "unconfirmed_actions",
        "executed_actions",
    ):
        assert data[key] is None
    assert data["actual_settle_steps"] is None
    assert data["target_reached"] is None
    assert data["stop_reason"] == "execution_outcome_unknown"
    assert (
        primitives.move_to(arm="left", xyz=_motion_target(backend)).data["code"]
        == "command_state_unknown"
    )
    assert len(backend.actions) == 3


@pytest.mark.parametrize("bad", [True, -1, 2])
def test_invalid_hold_counts_stop_and_preserve_previous_confirmed_path(motion, bad):
    _establish_prior_hold(motion)
    primitives, backend, _, _, _ = motion
    _feedback_on_hold(
        backend,
        {
            "issued_actions": 1,
            "confirmed_actions": bad,
            "unconfirmed_actions": 0,
            "executed_actions": 1,
            "observation_after_last_issued_action": True,
        },
    )
    result = primitives.move_to(arm="left", xyz=_motion_target(backend), settle_steps=3)
    assert len(backend.actions) == 2
    assert result.data["confirmed_actions"] is None
    assert result.data["stop_reason"] == "invalid_execution_feedback"


def test_session_end_during_hold_preserves_partial_terminal_totals(motion):
    _establish_prior_hold(motion)
    primitives, backend, _, _, _ = motion
    _feedback_on_hold(
        backend,
        {
            "issued_actions": 1,
            "confirmed_actions": 0,
            "unconfirmed_actions": 1,
            "executed_actions": 0,
            "observation_after_last_issued_action": False,
            "stop_reason": "session_ended",
        },
        hold_number=2,
    )
    result = primitives.move_to(arm="left", xyz=_motion_target(backend), settle_steps=3)
    assert len(backend.actions) == 3
    assert result.data["requested_actions"] == result.data["issued_actions"] == 6
    assert result.data["confirmed_actions"] == result.data["executed_actions"] == 5
    assert result.data["unconfirmed_actions"] == 1
    assert result.data["actual_settle_steps"] == 1
    assert result.data["session_ended"] is True
    assert result.data["stop_reason"] == "session_ended"
    assert primitives.move_to(arm="left", xyz=_motion_target(backend)).is_error
    assert len(backend.actions) == 3


def test_inconsistent_hold_counters_do_not_form_a_stable_sample(motion):
    _establish_prior_hold(motion)
    primitives, backend, _, _, _ = motion
    _feedback_on_hold(
        backend,
        {
            "issued_actions": 1,
            "confirmed_actions": 0,
            "unconfirmed_actions": 0,
            "executed_actions": 1,
            "observation_after_last_issued_action": True,
        },
    )
    result = primitives.move_to(arm="left", xyz=_motion_target(backend), settle_steps=3)
    assert len(backend.actions) == 2
    assert result.data["stop_reason"] == "invalid_execution_feedback"
    assert "counters disagree" in result.data["error"]
    for key in (
        "issued_actions",
        "confirmed_actions",
        "unconfirmed_actions",
        "executed_actions",
    ):
        assert result.data[key] is None


def test_cancellation_raised_during_hold_is_not_caught_as_transport_error(motion):
    _establish_prior_hold(motion)
    primitives, backend, _, _, _ = motion
    _set_feedback_pose_sequence(backend, [0.006])
    original = backend.chunk_step

    def cancelled_on_hold(actions, **kwargs):
        if backend.actions:
            raise ToolCancelled("cancel transport")
        return original(actions, **kwargs)

    backend.chunk_step = cancelled_on_hold
    result = primitives.move_to(arm="left", xyz=_motion_target(backend), settle_steps=3)
    assert (
        result.data["code"] == "tool_cancelled" and result.data["interrupted"] is True
    )
    assert result.data["confirmed_actions"] is None
    assert len(backend.actions) == 1
    assert primitives._command_history_unknown is True


@pytest.mark.parametrize("budget", [-1, 181, True, 3.0])
def test_invalid_settling_budget_never_plans_or_executes(motion, budget):
    primitives, backend, planner, _, _ = motion
    with pytest.raises(ValueError, match="0 to 180"):
        primitives.move_to(arm="left", xyz=_motion_target(backend), settle_steps=budget)
    assert not backend.actions and not planner.calls


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
    "arm,target",
    [
        ("both", [0, 0, 0, 0, 0, 0, 1]),
        ("left", [0, 0, 0, 1, 0, 0, 1]),
        ("left", [0, 0, np.nan, 0, 0, 0, 1]),
    ],
)
def test_invalid_rpc_pose_is_rejected_at_facade_boundary(motion, arm, target):
    _, backend, planner, rpc, _ = motion
    with pytest.raises(ValueError):
        rpc.call("env.plan_arm_path", kwargs={"arm": arm, "target_pose": target})
    assert planner.calls == [] and backend.actions == []


def totals(result, requested, issued, confirmed, unconfirmed):
    assert result.data["requested_actions"] == requested
    assert result.data["issued_actions"] == issued
    assert result.data["confirmed_actions"] == confirmed
    assert result.data["executed_actions"] == confirmed
    assert result.data["unconfirmed_actions"] == unconfirmed


@pytest.mark.parametrize(("arm", "raw_velocity"), [("left", 0.0), ("right", 0.33)])
def test_initial_always_samples_exactly_one_hold_before_native_plan(
    motion, arm, raw_velocity
):
    primitive, backend, planner, _, model = motion
    before = backend.raw["r1::proprio"]
    before[57:61] = raw_velocity
    before[10:17] = -raw_velocity
    before[35:42] = raw_velocity
    frozen = before.copy()
    result = primitive.move_to(arm=arm, xyz=_motion_target(backend, arm))
    assert result.data["completed"] is True
    assert [len(actions) for actions in backend.actions] == [1, 4]
    assert len(planner.calls) == 1 and model.calls == 0
    hold = backend.actions[0][0]
    np.testing.assert_array_equal(hold[:3], 0)
    np.testing.assert_allclose(hold[3:7], frozen[53:57])
    np.testing.assert_allclose(hold[7:14], frozen[3:10])
    np.testing.assert_allclose(hold[15:22], frozen[28:35])
    np.testing.assert_array_equal(hold[[14, 22]], 1)
    totals(result, 5, 5, 5, 0)
    assert result.data["planned_motion_actions"] == 4
    assert result.data["actual_settle_steps"] == 0
    # The original raw signal is neither rewritten nor used as the end gate.
    assert backend.raw["r1::proprio"][57] == raw_velocity


@pytest.mark.parametrize("index, value", [(49, 0.047), (51, 0.003), (0, 0.011)])
def test_unsafe_initial_geometry_never_sends_a_sampling_action(motion, index, value):
    primitive, backend, planner, _, _ = motion
    backend.raw["r1::proprio"][index] = value
    result = primitive.move_to(arm="left", xyz=_motion_target(backend))
    assert result.is_error
    totals(result, 0, 0, 0, 0)
    assert not backend.actions and not planner.calls


@pytest.mark.parametrize(
    "kwargs",
    [
        {"arm": "middle", "xyz": [0, 0, 0]},
        {"arm": "left", "xyz": [float("nan"), 0, 0]},
        {"arm": "left", "xyz": [0, 0]},
        {"arm": "left", "xyz": [0, 0, 0], "quat": [0, 0, 0, 2]},
        {"arm": "left", "xyz": [0, 0, 0], "settle_steps": True},
        {"arm": "left", "xyz": [0, 0, 0], "settle_steps": 181},
    ],
)
def test_invalid_arguments_raise_without_spending_actions(motion, kwargs):
    primitive, backend, planner, _, _ = motion
    with pytest.raises((TypeError, ValueError)):
        primitive.move_to(**kwargs)
    assert not backend.actions and not planner.calls


def test_requested_xyz_and_implicit_entry_quaternion_survive_sampling(motion):
    primitive, backend, planner, _, _ = motion
    xyz = _motion_target(backend)
    xyz[0] += 0.05
    initial_quat = backend.raw["r1::proprio"][20:24].copy()
    step = backend.chunk_step

    def sampling_changes_eef(actions, **kwargs):
        result = step(actions, **kwargs)
        if len(backend.actions) == 1:
            backend.raw["r1::proprio"][17] += 0.01
            backend.raw["r1::proprio"][20:24] = [0, 0, np.sin(0.01), np.cos(0.01)]
        return _refresh_reply(backend, result)

    backend.chunk_step = sampling_changes_eef
    result = primitive.move_to(arm="left", xyz=xyz)
    assert len(planner.calls) == 1
    np.testing.assert_array_equal(planner.calls[0][1], [*xyz, *initial_quat])
    np.testing.assert_array_equal(result.data["target_pose"], [*xyz, *initial_quat])


def test_actual_post_hold_displacement_cannot_bypass_existing_static_gate(motion):
    primitive, backend, planner, _, _ = motion
    step = backend.chunk_step

    def moving_hold(actions, **kwargs):
        result = step(actions, **kwargs)
        backend.raw["r1::proprio"][53] += 0.001
        return _refresh_reply(backend, result)

    backend.chunk_step = moving_hold
    result = primitive.move_to(arm="left", xyz=_motion_target(backend))
    # Continuously injected trunk motion never damps out, so the settle budget is
    # spent and the call must refuse to plan. What matters is that the planner is
    # never reached with an unsettled start, not how many holds were spent trying.
    assert result.is_error
    assert result.data["code"] in {"motion_state_unsettled", "base_not_stationary"}
    rejection = result.data["rejection"]
    assert rejection["code"] == result.data["code"]
    assert rejection["reason"] and rejection["suggested_next"]
    assert rejection["auto_settle"] == result.data["auto_settle"]
    assert result.data["auto_settle"]["status"] in {
        "still_unsettled",
        "hold_unavailable",
    }
    assert not planner.calls
    assert len(backend.actions) == 1 + result.data["auto_settle"]["attempted"]


@pytest.mark.parametrize(
    "case",
    [
        "partial",
        "missing_counts",
        "contradictory_counts",
        "stale",
        "generation",
        "previous",
        "pending",
    ],
)
def test_probe_invalid_or_unconfirmed_feedback_never_reaches_planner(motion, case):
    primitive, backend, planner, _, _ = motion
    step = backend.chunk_step
    if case == "partial":
        backend.feedback = {
            "issued_actions": 1,
            "confirmed_actions": 0,
            "unconfirmed_actions": 1,
            "executed_actions": 0,
            "observation_after_last_issued_action": False,
        }

    def broken_feedback(actions, **kwargs):
        obs, reward, terminated, truncated, info = step(actions, **kwargs)
        if case == "missing_counts":
            info.pop("executed_actions")
        elif case == "contradictory_counts":
            info["issued_actions"] = 0
        elif case == "stale":
            info["observation_token"]["observation_index"] = 0
        elif case == "generation":
            info["observation_token"]["generation"] = 1
        elif case == "previous":
            info["motion_context"]["previous_proprio"][53] += 1
        elif case == "pending":
            info["motion_context"]["actions_pending"] = True
        return obs, reward, terminated, truncated, info

    backend.chunk_step = broken_feedback
    result = primitive.move_to(arm="left", xyz=_motion_target(backend))
    assert result.is_error and result.data["observation_fresh"] is False
    assert result.data["requested_actions"] == 1
    assert primitive._command_history_unknown is True
    assert not planner.calls and len(backend.actions) == 1
    if case == "partial":
        totals(result, 1, 1, 0, 1)
    elif case == "missing_counts":
        assert result.data["issued_actions"] == result.data["confirmed_actions"] == 1
        assert result.data["unconfirmed_actions"] == 0
        assert result.data["executed_actions"] is None
    elif case == "contradictory_counts":
        totals(result, 1, None, None, None)
    else:
        totals(result, 1, 1, 1, 0)


def test_probe_transport_failure_never_reports_zero_spent_actions(motion):
    primitive, backend, planner, _, _ = motion
    backend.fail = True
    result = primitive.move_to(arm="left", xyz=_motion_target(backend))
    totals(result, 1, None, None, None)
    assert result.data["code"] == "execution_outcome_unknown"
    assert result.data["observation_fresh"] is False
    assert primitive._command_history_unknown is True
    assert not planner.calls and len(backend.actions) == 1


@pytest.mark.parametrize(
    "stale", [False, True], ids=["solver-rejection", "stale-frame"]
)
def test_native_plan_failure_after_sampling_reports_the_spent_hold(motion, stale):
    primitive, backend, planner, _, _ = motion

    def fail_plan(plan, _):
        if stale:
            backend.observation_index += 1
            backend.raw["r1::proprio"][17] += 0.01
        else:
            plan.update(status="Failure", code="native_plan_failed")

    planner.mutate = fail_plan
    result = primitive.move_to(arm="left", xyz=_motion_target(backend))
    assert result.is_error
    assert result.data["code"] == ("stale_arm_plan" if stale else "native_plan_failed")
    totals(result, 1, 1, 1, 0)
    assert result.data["observation_fresh"] is not stale
    if stale:
        assert primitive.last_observation["r1::proprio"][17] == pytest.approx(0.41)
    assert len(backend.actions) == len(planner.calls) == 1


def test_cancelled_immediately_after_confirmed_probe_preserves_that_frame(motion):
    primitive, backend, planner, _, _ = motion
    step = backend.chunk_step

    def cancel_after_probe(actions, **kwargs):
        result = step(actions, **kwargs)
        primitive._check_cancelled = lambda: (_ for _ in ()).throw(ToolCancelled())
        return result

    backend.chunk_step = cancel_after_probe
    result = primitive.move_to(arm="left", xyz=_motion_target(backend))
    totals(result, 1, 1, 1, 0)
    assert result.data["interrupted"] is True
    assert result.data["observation_fresh"] is True
    assert not planner.calls


def test_terminal_confirmed_probe_does_not_start_native_plan(motion):
    primitive, backend, planner, _, _ = motion
    step = backend.chunk_step

    def terminal_probe(actions, **kwargs):
        obs, reward, _, truncated, info = step(actions, **kwargs)
        return obs, reward, True, truncated, info

    backend.chunk_step = terminal_probe
    result = primitive.move_to(arm="left", xyz=_motion_target(backend))
    totals(result, 1, 1, 1, 0)
    assert result.data["terminated"] is True
    assert result.data["observation_fresh"] is True
    assert not planner.calls and len(backend.actions) == 1


@pytest.mark.parametrize("failed_submission", [2, 3])
def test_unknown_path_or_target_hold_preserves_confirmed_prefix(
    motion, failed_submission
):
    primitive, backend, planner, _, _ = motion
    step = backend.chunk_step
    xyz = _motion_target(backend)
    xyz[0] += 0.05

    def fail_after_prefix(actions, **kwargs):
        backend.fail = len(backend.actions) + 1 == failed_submission
        return step(actions, **kwargs)

    backend.chunk_step = fail_after_prefix
    result = primitive.move_to(arm="left", xyz=xyz, settle_steps=1)
    totals(result, 5 if failed_submission == 2 else 6, None, None, None)
    assert result.data["observation_fresh"] is False
    assert len(backend.actions) == failed_submission and len(planner.calls) == 1
    assert primitive._command_history_unknown is True


def test_partial_path_aggregates_confirmed_preflight_without_replay(motion):
    primitive, backend, planner, _, _ = motion
    step = backend.chunk_step

    def partial_path(actions, **kwargs):
        if len(backend.actions) == 1:
            backend.feedback = {
                "issued_actions": 3,
                "confirmed_actions": 2,
                "unconfirmed_actions": 1,
                "executed_actions": 2,
                "observation_after_last_issued_action": False,
            }
        return step(actions, **kwargs)

    backend.chunk_step = partial_path
    result = primitive.move_to(arm="left", xyz=_motion_target(backend), settle_steps=3)
    totals(result, 5, 4, 3, 1)
    assert [len(actions) for actions in backend.actions] == [1, 4]
    assert len(planner.calls) == 1


def test_cancelled_hold_keeps_every_already_confirmed_submission(motion):
    primitive, backend, planner, _, _ = motion
    step = backend.chunk_step
    xyz = _motion_target(backend)
    xyz[0] += 0.05

    def cancel_after_path(actions, **kwargs):
        result = step(actions, **kwargs)
        if len(backend.actions) == 2:
            primitive._check_cancelled = lambda: (_ for _ in ()).throw(ToolCancelled())
        return result

    backend.chunk_step = cancel_after_path
    result = primitive.move_to(arm="left", xyz=xyz, settle_steps=1)
    totals(result, 5, 5, 5, 0)
    assert result.data["code"] == "tool_cancelled"
    assert len(planner.calls) == 1 and len(backend.actions) == 2


def test_maximum_hold_budget_keeps_complete_parseable_tool_evidence(motion):
    primitive, backend, _, _, _ = motion
    step = backend.chunk_step

    def long_measurements(actions, **kwargs):
        result = step(actions, **kwargs)
        if len(backend.actions) > 1:
            for index in (0, 10, 35, 57, 26, 51):
                result[0]["r1::proprio"][index] = 0.024755092337727547
            result[0]["r1::proprio"][17] += 0.0062345678901234567
            result[0]["r1::proprio"][20] = 0.0012345678901234567
            result[0]["r1::proprio"][23] = np.sqrt(1 - 0.0012345678901234567**2)
            result[4]["motion_context"]["current_proprio"] = result[0][
                "r1::proprio"
            ].copy()
        return result

    backend.chunk_step = long_measurements
    result = primitive.move_to(
        arm="left", xyz=_motion_target(backend), settle_steps=180
    )
    rendered = result.to_text()
    assert len(rendered) < 60000
    assert json.loads(rendered)["actual_settle_steps"] == 180
    totals(result, 185, 185, 185, 0)


@pytest.mark.parametrize("arm", ["left", "right"])
def test_delta_target_is_current_pose_plus_delta_with_orientation_held(motion, arm):
    primitive, backend, planner, _, model = motion
    start = 17 if arm == "left" else 42
    original = Rotation.from_euler("xyz", [0.4, -0.3, 0.2])
    backend.raw["r1::proprio"][start + 3 : start + 7] = original.as_quat()
    _establish_prior_hold(motion)
    before = backend.raw["r1::proprio"].copy()
    # A stale cache must not anchor a relative target.
    primitive.last_observation = deepcopy(backend.raw)
    primitive.last_observation["r1::proprio"][start : start + 3] += 0.1
    _follow_planned_pose(backend, planner)
    delta = [0.01, -0.012, 0.005]

    result = primitive.move_eef_delta(arm=arm, delta_xyz=delta)

    goal = planner.calls[0][1]
    np.testing.assert_allclose(goal[:3], before[start : start + 3] + delta)
    np.testing.assert_allclose(
        Rotation.from_quat(goal[3:]).as_matrix(), original.as_matrix()
    )
    assert [actions.shape for actions in backend.actions] == [(4, 23)]
    assert result.data["completed"] is True and result.data["target_reached"] is True
    assert result.data["collision_checked"] is False
    assert result.data["requested_delta_xyz"] == pytest.approx(delta)
    assert result.data["frame"] == "robot_base"
    assert len(planner.calls) == 1 and model.calls == 0


@pytest.mark.parametrize(
    ("tool", "options", "offset", "yaw"),
    [
        ("move_eef_delta", {"delta_xyz": [0.01, 0, 0]}, [0.01, 0, 0], 0),
        ("rotate_wrist", {"delta_yaw_deg": 15}, [0, 0, 0], 15),
    ],
)
def test_relative_target_is_frozen_across_the_initial_sampling_hold(
    motion, tool, options, offset, yaw
):
    primitive, backend, planner, _, model = motion
    original = backend.raw["r1::proprio"][17:24].copy()
    step = backend.chunk_step

    def initial_pose_drift(actions, **kwargs):
        reply = step(actions, **kwargs)
        if not planner.calls:
            backend.raw["r1::proprio"][17] += 0.001
            backend.raw["r1::proprio"][20:24] = Rotation.from_rotvec(
                [0, 0, 0.01]
            ).as_quat()
        return _refresh_reply(backend, reply)

    backend.chunk_step = initial_pose_drift
    _follow_planned_pose(backend, planner)
    result = getattr(primitive, tool)(arm="left", **options)
    goal = planner.calls[0][1]
    np.testing.assert_allclose(goal[:3], original[:3] + offset)
    expected = Rotation.from_euler("z", yaw, degrees=True) * Rotation.from_quat(
        original[3:]
    )
    np.testing.assert_allclose(
        Rotation.from_quat(goal[3:]).as_matrix(), expected.as_matrix()
    )
    assert [len(actions) for actions in backend.actions] == [1, 4]
    totals(result, 5, 5, 5, 0)
    assert result.data["completed"] is True and model.calls == 0


@pytest.mark.parametrize(
    "delta", [[0, 0, 0], [0.31, 0, 0], [0.25, 0.25, 0], [np.nan, 0, 0], [0.01, 0]]
)
def test_delta_limit_is_euclidean_and_finite(motion, delta):
    primitive, backend, planner, _, _ = motion
    with pytest.raises(ValueError, match="delta_xyz"):
        primitive.move_eef_delta(arm="left", delta_xyz=delta)
    assert not backend.actions and not planner.calls


@pytest.mark.parametrize(
    ("tool", "options"),
    [
        ("move_eef_delta", {"delta_xyz": [0.01, 0, 0]}),
        ("rotate_wrist", {"delta_yaw_deg": 15}),
    ],
)
def test_relative_arm_tools_reject_a_moving_reference_frame(motion, tool, options):
    primitive, backend, planner, _, _ = motion
    backend.raw["r1::proprio"][0] = 0.1
    result = getattr(primitive, tool)(arm="left", **options)
    assert result.is_error and result.data["issued_actions"] == 0
    assert not backend.actions and not planner.calls


def test_unreached_target_is_not_reported_as_completed(motion):
    primitive, backend, _, _, _ = motion
    _establish_prior_hold(motion)
    result = primitive.move_eef_delta(arm="left", delta_xyz=[0.01, 0, 0])
    assert result.data["target_reached"] is False
    assert result.data["completed"] is False
    assert result.data["position_error_m"] == pytest.approx(0.01, abs=0.002)
    assert len(backend.actions) == 1


@pytest.mark.parametrize("delta", [[0.01, 0], [0.01, 0, 0, 0], "move", None])
def test_schema_rejects_out_of_scope_values(motion, delta):
    declaration = {entry.name: entry for entry in iter_tools(motion[0])}[
        "move_eef_delta"
    ]
    with pytest.raises(ValidationError):
        declaration.args_schema.model_validate({"arm": "left", "delta_xyz": delta})


@pytest.mark.parametrize(("arm", "angle"), [("left", -15.0), ("right", 15.0)])
def test_rotation_uses_current_base_axis_pose_and_existing_native23_path(
    motion, arm, angle
):
    primitive, backend, planner, _, model = motion
    start = 17 if arm == "left" else 42
    original = Rotation.from_euler("xyz", [0.4, -0.3, 0.2])
    backend.raw["r1::proprio"][start + 3 : start + 7] = original.as_quat()
    _establish_prior_hold(motion)
    before = backend.raw["r1::proprio"].copy()
    # The primitive's old cache must not anchor a relative target.
    primitive.last_observation = deepcopy(backend.raw)
    primitive.last_observation["r1::proprio"][start : start + 3] += 0.1
    _follow_planned_pose(backend, planner)

    result = primitive.rotate_wrist(arm=arm, delta_yaw_deg=angle)

    expected = Rotation.from_rotvec([0, 0, np.deg2rad(angle)]) * original
    goal = planner.calls[0][1]
    np.testing.assert_allclose(goal[:3], before[start : start + 3])
    np.testing.assert_allclose(
        Rotation.from_quat(goal[3:]).as_matrix(), expected.as_matrix()
    )
    assert [actions.shape for actions in backend.actions] == [(4, 23)]
    assert result.data["completed"] is True and result.data["target_reached"] is True
    assert result.data["confirmed_actions"] == result.data["issued_actions"] == 4
    assert result.data["unconfirmed_actions"] == result.data["actual_settle_steps"] == 0
    assert len(planner.calls) == 1 and model.calls == 0


@pytest.mark.parametrize(
    "angle", [-180.01, 180.01, float("nan"), float("inf"), True, "10", None]
)
def test_invalid_rotation_angles_do_not_request_or_execute_a_path(motion, angle):
    primitive, backend, planner, _, _ = motion
    with pytest.raises(ValueError, match="delta_yaw_deg"):
        primitive.rotate_wrist(arm="left", delta_yaw_deg=angle)
    assert not backend.actions and not planner.calls


def test_unreached_orientation_is_not_reported_as_completed(motion):
    primitive, backend, _, _, _ = motion
    _establish_prior_hold(motion)
    result = primitive.rotate_wrist(arm="left", delta_yaw_deg=15)
    assert result.data["position_error_m"] == 0
    assert result.data["orientation_error_rad"] == pytest.approx(np.deg2rad(15))
    assert result.data["target_reached"] is False and result.data["completed"] is False
    assert result.data["confirmed_actions"] == 4
    assert len(backend.actions) == 1


@pytest.mark.parametrize("angle", [-181, 181, float("nan"), True, "10"])
def test_rotation_schema_rejects_out_of_scope_values(motion, angle):
    declaration = {entry.name: entry for entry in iter_tools(motion[0])}["rotate_wrist"]
    with pytest.raises(ValidationError):
        declaration.args_schema.model_validate({"arm": "left", "delta_yaw_deg": angle})


class Environment:
    def __init__(self):
        proprio = np.zeros(61, dtype=np.float64)
        proprio[3:10] = np.arange(7) * 0.1
        proprio[28:35] = np.arange(7) * -0.1
        proprio[[23, 48]] = 1
        proprio[24:26] = [0.04, 0.04]
        proprio[49:51] = [0.01, 0.01]
        proprio[53:57] = [0.1, -0.2, 0.3, 0.4]
        self.raw = {"r1::proprio": proprio}
        for camera in ("zed_link", "left_realsense_link", "right_realsense_link"):
            self.raw[f"r1::{camera}:Camera:0::rgb"] = np.zeros((4, 5, 3), np.uint8)
        self.initial = deepcopy(self.raw)
        self.actions = []
        self.tokens = []
        self.token = {"generation": 0, "observation_index": 1}
        self.context_patch = {}
        self.snapshot_patch = None
        self.reply_patch = None
        self.fail_at = None
        self.ended = False
        self.residual_stop_velocity = np.zeros(3)
        self.drift_after = None
        self.terminated = None

    def get_observation(self):
        return deepcopy(self.raw)

    def get_session_status(self):
        return {"ended": self.ended}

    def get_motion_snapshot(self):
        value = {
            "observation": self.get_observation(),
            "observation_token": dict(self.token),
            "motion_context": {
                "source": "confirmed_public_control_observations",
                "control_dt_s": CONTROL_DT_S,
                "current_token": dict(self.token),
                "current_proprio": self.raw["r1::proprio"].copy(),
                "initial_bootstrap": True,
                "unconfirmed_actions": 0,
                "actions_pending": False,
                **self.context_patch,
            },
        }
        if self.snapshot_patch:
            self.snapshot_patch(value)
        return value

    def chunk_step(self, actions, *, expected_observation_token=None):
        np.testing.assert_equal(actions.shape, (1, 23))
        self.actions.append(actions.copy())
        self.tokens.append(deepcopy(expected_observation_token))
        assert expected_observation_token == self.token
        if self.fail_at == len(self.actions):
            raise TimeoutError("cannot know whether the action ran")
        previous_token = dict(self.token)
        previous_proprio = self.raw["r1::proprio"].copy()
        self.token["observation_index"] += 1
        self.raw["r1::proprio"][:3] = actions[0, :3] * BASE_OUTPUT_SCALE
        if np.all(actions[0, :3] == 0):
            self.raw["r1::proprio"][:3] = self.residual_stop_velocity
        if self.drift_after:
            self.drift_after(self.raw)
        info = {
            "bridge_generation": self.token["generation"],
            "issued_actions": 1,
            "confirmed_actions": 1,
            "unconfirmed_actions": 0,
            "executed_actions": 1,
            "observation_after_last_issued_action": True,
            "stop_reason": "chunk_completed",
            "observation_token": dict(self.token),
            "motion_context": {
                "source": "confirmed_public_control_observations",
                "control_dt_s": CONTROL_DT_S,
                "current_token": dict(self.token),
                "current_proprio": self.raw["r1::proprio"].copy(),
                "previous_token": previous_token,
                "previous_proprio": previous_proprio,
                "pair_confirmed": True,
                "actions_pending": False,
                "unconfirmed_actions": 0,
            },
        }
        observation = self.get_observation()
        if self.reply_patch:
            self.reply_patch(len(self.actions), info, observation)
        return observation, None, self.terminated, None, info


class Model:
    def predict(self, *args, **kwargs):
        raise AssertionError("base motion must not call VLA")


def make_primitive(env=None, cancelled=lambda: None):
    env = env or Environment()
    primitive = BehaviorPrimitives(
        env=env,
        model=Model(),
        task_id=40,
        task_language="make popcorn",
        check_cancelled=cancelled,
        execution_horizon=32,
        instruction_mode="main",
    )
    return primitive, env


@pytest.mark.parametrize("axis", range(3))
@pytest.mark.parametrize("sign", [-1, 1])
def test_signed_commands_use_exact_native_channel_and_confirm_stop(axis, sign):
    primitive, env = make_primitive()
    velocity = np.zeros(3)
    velocity[axis] = sign * 0.1
    result = primitive.move_base(
        **dict(zip(("forward", "lateral", "turn"), velocity, strict=True)), steps=3
    ).data
    assert result["completed"] and result["stop_command_confirmed"]
    assert result["requested_actions"] == result["confirmed_actions"] == 4
    assert result["issued_actions"] == result["executed_actions"] == 4
    assert result["unconfirmed_actions"] == 0
    assert result["confirmed_move_steps"] == 3
    stack = np.concatenate(env.actions)
    np.testing.assert_allclose(stack[:3, :3], np.tile(velocity, (3, 1)))
    np.testing.assert_equal(stack[-1, :3], 0)
    np.testing.assert_allclose(
        result["target_base_velocity_m_s_rad_s"], velocity * BASE_OUTPUT_SCALE
    )
    assert [item["observation_index"] for item in env.tokens] == [1, 2, 3, 4]
    assert "target_reached" not in result


def test_all_other_twenty_channels_are_fixed_to_initial_targets():
    primitive, env = make_primitive()
    primitive._last_issued_action = np.zeros(23, dtype=np.float32)
    primitive._last_issued_action[[14, 22]] = [-0.7, 0.4]
    env.drift_after = lambda raw: raw["r1::proprio"].__setitem__(
        53, raw["r1::proprio"][53] + 0.003
    )
    result = primitive.move_base(forward=0.1, lateral=0.1, turn=0.1, steps=5).data
    stack = np.concatenate(env.actions)
    np.testing.assert_allclose(stack[:, 3:], np.tile(stack[0, 3:], (6, 1)))
    for action_slice, state_slice in (
        (slice(3, 7), slice(53, 57)),
        (slice(7, 14), slice(3, 10)),
        (slice(15, 22), slice(28, 35)),
    ):
        np.testing.assert_allclose(
            stack[0, action_slice], env.initial["r1::proprio"][state_slice]
        )
    np.testing.assert_allclose(stack[:, [14, 22]], np.tile([-0.7, 0.4], (6, 1)))
    assert result["completed"]


def test_residual_velocity_is_reported_without_an_all_robot_zero_gate():
    primitive, env = make_primitive()
    env.residual_stop_velocity = [0.018, -0.003, 0.005]
    result = primitive.move_base(forward=0.1, steps=2).data
    assert result["completed"] and result["stop_command_confirmed"]
    assert len(env.actions) == 3


@pytest.mark.parametrize(
    "kwargs",
    [
        {"forward": True},
        {"lateral": float("nan")},
        {"turn": float("inf")},
        {"forward": 1.0001},
        {"forward": 1.0, "lateral": 0.01},
        {"turn": -1.0001},
        {"steps": True},
        {"steps": 2.0},
        {"steps": 0},
        {"steps": 5001},
    ],
)
def test_invalid_request_is_rejected_before_sending(kwargs):
    primitive, env = make_primitive()
    with pytest.raises(ValueError):
        primitive.move_base(**kwargs)
    assert env.actions == []


@pytest.mark.parametrize(
    "kwargs",
    [
        {"forward": True},
        {"steps": True},
        {"steps": 2.0},
        {"turn": float("nan")},
        {"steps": 5001},
        {"forward": 1.3},
    ],
)
def test_planner_schema_rejects_invalid_inputs(kwargs):
    definition = BehaviorPrimitives.move_base
    with pytest.raises(ValidationError):
        definition.args_schema.model_validate(kwargs, strict=True)


def test_schema_describes_velocity_units_and_completion():
    primitive, _ = make_primitive()
    tools = {item.name: item for item in iter_tools(primitive)}
    assert not tools["move_base"].readonly
    schema = tools["move_base"].input_schema
    assert set(schema["properties"]) == {"forward", "lateral", "turn", "steps"}
    assert schema["properties"]["steps"]["maximum"] == 5000
    assert "0.075 m/s" in schema["properties"]["forward"]["description"]
    assert "not world-coordinate navigation" in tools["move_base"].description


@pytest.mark.parametrize(
    "patch",
    [
        {"actions_pending": True},
        {"unconfirmed_actions": 1},
        {"unconfirmed_actions": None},
        {"current_token": {"generation": 3, "observation_index": 1}},
    ],
)
def test_pending_unknown_or_mismatched_initial_state_sends_nothing(patch):
    primitive, env = make_primitive()
    env.context_patch = patch
    result = primitive.move_base(forward=0.1).data
    assert result["stop_reason"] == "base_start_unavailable"
    assert result["requested_actions"] == result["confirmed_actions"] == 0
    assert env.actions == []


@pytest.mark.parametrize("state", ["unknown-command", "missing-prior-command"])
def test_base_motion_requires_a_known_gripper_hold(state):
    primitive, env = make_primitive()
    if state == "unknown-command":
        primitive._command_history_unknown = True
    else:
        env.context_patch = {"initial_bootstrap": False}
    result = primitive.move_base(forward=0.1).data
    assert result["stop_reason"] == "base_start_unavailable"
    assert not env.actions


@pytest.mark.parametrize(
    "mutate",
    [
        lambda context: context.update(control_dt_s=1 / 120),
        lambda context: context.update(source="simulator_privileged_state"),
        lambda context: context["current_proprio"].__setitem__(0, 0.123),
    ],
)
def test_wrong_public_start_binding_or_timestep_sends_nothing(mutate):
    primitive, env = make_primitive()
    env.snapshot_patch = lambda snapshot: mutate(snapshot["motion_context"])
    result = primitive.move_base(forward=0.1).data
    assert result["stop_reason"] == "base_start_unavailable" and env.actions == []


@pytest.mark.parametrize("flag", ["_terminated", "_truncated"])
def test_ended_episode_does_not_submit_new_motion(flag):
    primitive, env = make_primitive()
    setattr(primitive, flag, True)
    result = primitive.move_base(forward=0.1).data
    assert result["stop_reason"] == "episode_ended" and env.actions == []


@pytest.mark.parametrize("fail_at", [1, 3, 4])
def test_transport_failure_preserves_prefix_and_never_replays_or_adds_stop(fail_at):
    primitive, env = make_primitive()
    env.fail_at = fail_at
    result = primitive.move_base(forward=0.1, steps=3).data
    assert len(env.actions) == fail_at
    assert result["issued_actions"] is None and result["confirmed_actions"] is None
    assert result["known_execution_before_failure"]["confirmed_actions"] == fail_at - 1
    assert result["requested_actions"] == fail_at
    assert not result["completed"] and not result["stop_command_confirmed"]
    assert primitive._command_history_unknown and primitive._last_issued_action is None


@pytest.mark.parametrize(
    "mutate",
    [
        lambda info: info.update(confirmed_actions=True),
        lambda info: info.update(executed_actions=0),
    ],
)
def test_bad_counts_stop_without_replaying(mutate):
    primitive, env = make_primitive()
    env.reply_patch = lambda index, info, observation: (
        mutate(info) if index == 2 else None
    )
    result = primitive.move_base(forward=0.1, steps=3).data
    assert len(env.actions) == 2
    assert result["known_execution_before_failure"]["confirmed_actions"] == 1
    assert result["confirmed_actions"] is None and not result["completed"]


@pytest.mark.parametrize(
    "mutate",
    [
        lambda info: info["observation_token"].update(generation=1),
        lambda info: info["observation_token"].update(observation_index=1),
        lambda info: info.update(motion_context={}),
        lambda info: info.update(motion_feedback_error="mismatched frame"),
    ],
)
def test_bad_frame_binding_retains_reported_counts_but_stops(mutate):
    primitive, env = make_primitive()
    env.reply_patch = lambda index, info, observation: (
        mutate(info) if index == 2 else None
    )
    result = primitive.move_base(forward=0.1, steps=3).data
    assert len(env.actions) == 2
    assert result["confirmed_actions"] == 2
    assert result["stop_reason"] == "invalid_execution_feedback"
    assert not result["completed"] and primitive._command_history_unknown


def test_invalid_public_observation_does_not_turn_confirmed_count_into_success():
    primitive, env = make_primitive()
    env.reply_patch = lambda index, info, observation: (
        observation["r1::proprio"].__setitem__(0, np.nan) if index == 2 else None
    )
    result = primitive.move_base(forward=0.1, steps=3).data
    assert result["confirmed_actions"] == 2
    assert result["stop_reason"] == "invalid_public_observation"
    assert not result["completed"] and not result["observation_fresh"]
    assert len(env.actions) == 2


@pytest.mark.parametrize(
    "mutate",
    [
        lambda context: context.update(pair_confirmed=False),
        lambda context: context["previous_token"].update(observation_index=99),
        lambda context: context["previous_proprio"].__setitem__(0, 0.5),
        lambda context: context["current_proprio"].__setitem__(0, 0.5),
        lambda context: context.update(actions_pending=True),
        lambda context: context.update(control_dt_s=1 / 120),
    ],
)
@pytest.mark.parametrize("bad_index", [2, 4])
def test_nonadjacent_or_unbound_public_feedback_cannot_integrate_or_continue(
    mutate, bad_index
):
    primitive, env = make_primitive()
    env.reply_patch = lambda index, info, observation: (
        mutate(info["motion_context"]) if index == bad_index else None
    )
    result = primitive.move_base(forward=0.1, steps=3).data
    assert result["confirmed_actions"] == bad_index
    assert result["stop_reason"] == "invalid_public_observation"
    assert not result["completed"] and primitive._command_history_unknown
    assert len(env.actions) == bad_index


def test_terminated_during_drive_preserves_prefix_and_never_claims_stop():
    primitive, env = make_primitive()
    env.terminated = True
    result = primitive.move_base(forward=0.1, steps=3).data
    assert result["confirmed_actions"] == 1 and result["terminated"]
    assert not result["completed"] and not result["stop_command_confirmed"]
    assert len(env.actions) == 1


def test_cancel_between_actions_keeps_confirmed_count_and_does_not_claim_stop():
    env = Environment()

    def cancelled():
        if env.actions:
            raise ToolCancelled("cancelled by caller")

    primitive, env = make_primitive(env, cancelled)
    result = primitive.move_base(forward=0.1, steps=3).data
    assert result["confirmed_actions"] == 1 and result["interrupted"]
    assert result["stop_reason"] == "tool_cancelled"
    assert len(env.actions) == 1 and not result["stop_command_confirmed"]


@pytest.mark.parametrize(
    "reported_velocity", [[0, 0, 0], [0.08, 0.01, -0.03], [0.15, -0.02, 0.04]]
)
def test_complete_raw_sample_chain_never_becomes_trusted_odometry(reported_velocity):
    primitive, env = make_primitive()
    env.drift_after = lambda raw: raw["r1::proprio"].__setitem__(
        slice(0, 3), reported_velocity
    )
    result = primitive.move_base(forward=0.1, steps=2).data
    # The mock has no physical base pose; neither plausible nor contradictory
    # raw velocities can establish a displacement or stillness guarantee.
    assert result["completed"]
    np.testing.assert_allclose(
        np.concatenate(env.actions)[:, :3], [[0.1, 0, 0], [0.1, 0, 0], [0, 0, 0]]
    )


def test_existing_toolkit_dispatch_saves_state_and_returns_three_images(
    tmp_path, monkeypatch
):
    monkeypatch.setattr("rpent.utils.templates.get_output_dir", lambda: tmp_path)
    env = Environment()
    toolkit = BehaviorToolkit(
        runtime_kwargs={
            "env": env,
            "model": Model(),
            "task_id": 40,
            "task_language": "make popcorn",
            "execution_horizon": 32,
            "tool_allowlist": ["view_env_state", "move_base", "finish"],
        },
        dashboard_events=NullDashboardEventSink(),
        memory=MemoryManager(tmp_path / "memory"),
        state_output_dir=tmp_path / "state",
    )
    try:
        result = toolkit.execute_tool("move_base", {"forward": 0.1, "steps": 2})
        assert not result.is_error and len(result.images) == 3
        assert toolkit.state.latest_step == 1
        assert result.data["log"]["result"]["completed"]
        assert result.data["log"]["result"]["confirmed_actions"] == 3
        assert toolkit.state.load("proprio.npy", step=1).shape == (61,)
        assert len(env.actions) == 3
    finally:
        toolkit.close()


@pytest.mark.parametrize("terminal_unconfirmed", [False, True])
def test_real_bridge_and_client_handshake_confirms_each_base_step(terminal_unconfirmed):
    """Use actual bridge/client threads; synthetic public state is not physics."""
    bridge = EvaluatorStepBridge(timeout_s=2)
    facade = BehaviorEnvFacade(bridge, task_id=40)

    class DirectRPC:
        def call(self, method, args=(), kwargs=None, **ignored):
            return facade._dispatch(method, args, kwargs or {})

    client = BehaviorEnvClient(DirectRPC(), task_id=40)
    raw = Environment().get_observation()
    pool = ThreadPoolExecutor(max_workers=2)
    try:
        policy = pool.submit(bridge.observe_and_get_action, raw)
        primitive, _ = make_primitive(client)
        tool_call = pool.submit(primitive.move_base, forward=0.1, steps=2)
        actual = []
        for index in range(3):
            action = policy.result(timeout=3)
            actual.append(action)
            assert action.shape == (23,)
            if terminal_unconfirmed and index == 2:
                bridge.end_session()
                break
            raw = deepcopy(raw)
            raw["r1::proprio"][:3] = action[:3] * BASE_OUTPUT_SCALE
            policy = pool.submit(bridge.observe_and_get_action, raw)
        result = tool_call.result(timeout=3).data
        np.testing.assert_allclose(np.array(actual)[:, 0], [0.1, 0.1, 0])
        assert result["issued_actions"] == 3
        assert result["confirmed_actions"] == (2 if terminal_unconfirmed else 3)
        assert result["unconfirmed_actions"] == int(terminal_unconfirmed)
        assert result["completed"] is not terminal_unconfirmed
        assert result["stop_command_confirmed"] is not terminal_unconfirmed
        assert bridge.stats()["confirmed"] == result["confirmed_actions"]
        assert result["session_ended"] is terminal_unconfirmed
        assert primitive._command_history_unknown is terminal_unconfirmed
        bridge.close()
        if not terminal_unconfirmed:
            with pytest.raises(BridgeCancelled):
                policy.result(timeout=3)
    finally:
        bridge.close()
        facade.close()
        pool.shutdown(wait=True)
