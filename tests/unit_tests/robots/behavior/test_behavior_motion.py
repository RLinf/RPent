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
import unittest
from copy import deepcopy
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest
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
    LocalRpc,
    _GripperEnvironment,
    _GripperModel,
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
    relaxed = planner.plan_arm_path(
        arm="left",
        target_pose=target,
        observation=raw,
        motion_context=context,
        allow_orientation_relax=True,
    )
    assert relaxed["status"] == "Success"
    assert relaxed["orientation_relaxed"] is True
    assert relaxed["ik_orientation_error_rad"] == pytest.approx(np.pi / 2, abs=1e-6)
    assert relaxed["target_pose"] == target.tolist()


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
    for name in (
        "issued_actions",
        "confirmed_actions",
        "unconfirmed_actions",
        "executed_actions",
    ):
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
    for key in (
        "issued_actions",
        "confirmed_actions",
        "unconfirmed_actions",
        "executed_actions",
    ):
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
    for name in (
        "issued_actions",
        "confirmed_actions",
        "unconfirmed_actions",
        "executed_actions",
    ):
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
    names = (
        "issued_actions",
        "confirmed_actions",
        "unconfirmed_actions",
        "executed_actions",
    )
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
    arguments = {"arm": "left", "steps": steps}
    if tool == "set_gripper":
        arguments["val"] = 1
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
    arguments = {"arm": "left", "steps": steps, "settle_steps": settle_steps}
    if tool == "set_gripper":
        arguments["val"] = 1
    with pytest.raises(ValueError, match="5000|budget"):
        getattr(primitives, tool)(**arguments)
    assert env.actions == []


@pytest.mark.parametrize("tool", ["set_gripper", "release"])
def test_gripper_exact_budget_is_accepted(gripper, tool):
    primitives, env, _ = gripper
    arguments = {"arm": "left", "steps": 4990, "settle_steps": 10}
    if tool == "set_gripper":
        arguments["val"] = 1
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
