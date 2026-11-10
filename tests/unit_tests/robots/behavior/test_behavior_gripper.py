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

"""Gripper and single-joint wrist tools on the native23 path."""

from __future__ import annotations

import math

import numpy as np
import pytest

from robots.behavior.primitives import WRIST_JOINT_OFFSET


@pytest.mark.parametrize(
    "arm,val,expected",
    [
        ("left", 1, [-0.2, 0.2, 0.6, 1.0]),
        ("left", 0, [-0.7, -0.8, -0.9, -1.0]),
        ("right", 1, [0.7, 0.8, 0.9, 1.0]),
        ("right", 0, [0.2, -0.2, -0.6, -1.0]),
    ],
)
def test_native_sequence_preserves_other_body_parts_without_calling_vla(
    gripper, arm, val, expected
):
    primitives, env, model = gripper
    primitives.vla_act()
    before = env.raw["r1::proprio"].copy()
    result = primitives.set_gripper(arm=arm, val=val, steps=4, settle_steps=0)
    assert model.calls == 1
    assert len(env.actions) == 2
    actions = env.actions[-1]
    assert actions.shape == (4, 23)
    np.testing.assert_allclose(actions[:, 14 if arm == "left" else 22], expected)
    np.testing.assert_allclose(
        actions[:, 22 if arm == "left" else 14], 0.6 if arm == "left" else -0.6
    )
    np.testing.assert_array_equal(actions[:, :3], 0)
    for action_slice, state_slice in (
        (slice(3, 7), slice(53, 57)),
        (slice(7, 14), slice(3, 10)),
        (slice(15, 22), slice(28, 35)),
    ):
        np.testing.assert_allclose(
            actions[:, action_slice], np.tile(before[state_slice], (4, 1))
        )
    assert result.data["start_source"] == "last_issued_native_action"
    assert result.data["start_val"] == pytest.approx(0.2 if arm == "left" else 0.8)
    assert result.data["issued_actions"] == result.data["confirmed_actions"] == 4
    assert result.data["unconfirmed_actions"] == 0
    assert result.data["grasp_success"] is None
    assert "success" not in result.data
    assert result.data["finger_qpos_after_m"] == result.data["finger_qpos_before_m"]


def test_release_default_opens_then_holds_for_ten_additional_steps(gripper):
    primitives, env, model = gripper
    primitives.vla_act()
    result = primitives.release(arm="left")
    assert result.data["val"] == 1
    assert result.data["native_target_command"] == 1
    assert result.data["ramp_steps"] == result.data["settle_steps"] == 10
    assert result.data["requested_actions"] == 20
    assert env.actions[-1].shape == (20, 23)
    np.testing.assert_array_equal(env.actions[-1][10:, 14], 1)
    assert env.actions[-1][-1, 14] == 1
    assert model.calls == 1


@pytest.mark.parametrize("arm", ["left", "right"])
def test_first_action_establishes_a_new_measured_hold_without_calling_vla(gripper, arm):
    primitives, env, model = gripper
    result = primitives.set_gripper(arm=arm, val=1, steps=2, settle_steps=0)
    assert not result.is_error
    assert model.calls == 0
    assert len(env.actions) == 1
    assert result.data["start_source"] == "new_measured_position_hold"
    assert result.data["other_gripper_hold_source"] == "new_measured_position_hold"
    assert result.data["start_val"] == pytest.approx(0.8 if arm == "left" else 0.2)
    native = env.actions[0]
    np.testing.assert_allclose(
        native[:, 14 if arm == "left" else 22],
        [0.8, 1] if arm == "left" else [0.2, 1],
    )
    np.testing.assert_allclose(
        native[:, 22 if arm == "left" else 14], -0.6 if arm == "left" else 0.6
    )
    np.testing.assert_array_equal(native[:, :3], 0)
    for action_slice, state_slice in (
        (slice(3, 7), slice(53, 57)),
        (slice(7, 14), slice(3, 10)),
        (slice(15, 22), slice(28, 35)),
    ):
        np.testing.assert_allclose(
            native[:, action_slice],
            np.tile(env.raw["r1::proprio"][state_slice], (2, 1)),
        )
    assert result.data["confirmed_actions"] == 2
    assert result.data["grasp_success"] is None
    second = primitives.set_gripper(arm=arm, val=0, steps=1)
    assert second.data["start_source"] == "last_issued_native_action"
    assert second.data["start_val"] == 1


def _assert_not_issued(result):
    for name in (
        "issued_actions",
        "confirmed_actions",
        "unconfirmed_actions",
        "executed_actions",
    ):
        assert result.data[name] == 0


@pytest.mark.parametrize(
    "state_slice,value",
    [
        (slice(24, 26), [0.03, 0.04]),
        (slice(49, 51), [0.03, 0.04]),
        (slice(24, 26), [0.052, 0.052]),
        (slice(49, 51), [-0.002, -0.002]),
        (slice(26, 28), [0.003, 0]),
        (slice(51, 53), [0, -0.003]),
    ],
)
def test_first_hold_refuses_unstable_asymmetric_or_out_of_range_fingers(
    gripper, state_slice, value
):
    primitives, env, model = gripper
    env.raw["r1::proprio"][state_slice] = value
    result = primitives.set_gripper(arm="left", val=1)
    assert result.data["code"] == "invalid_hold_state"
    _assert_not_issued(result)
    assert env.actions == [] and model.calls == 0


def test_small_sensor_tolerances_establish_bounded_symmetric_new_hold(gripper):
    primitives, env, _ = gripper
    env.raw["r1::proprio"][24:26] = [0.05, 0.0502]
    env.raw["r1::proprio"][49:51] = [0.0098, 0.0102]
    result = primitives.set_gripper(arm="right", val=0.5, steps=1)
    assert result.data["start_source"] == "new_measured_position_hold"
    assert result.data["start_val"] == pytest.approx(0.2)
    assert env.actions[0][0, 14] == 1


@pytest.mark.parametrize("val", [0, 0.5, -1, 2, True, "1", float("nan")])
def test_deprecated_release_alias_cannot_close_or_accept_a_non_open_target(
    gripper, val
):
    primitives, env, _ = gripper
    with pytest.raises(ValueError, match="only opens"):
        primitives.release(arm="left", val=val)
    assert env.actions == []


@pytest.mark.parametrize(
    "kwargs",
    [
        {"arm": "both", "val": 1},
        {"arm": "left", "val": -0.1},
        {"arm": "left", "val": 1.1},
        {"arm": "left", "val": float("nan")},
        {"arm": "left", "val": True},
        {"arm": "left", "val": "open"},
        {"arm": "left", "val": 1, "steps": 0},
        {"arm": "left", "val": 1, "steps": True},
        {"arm": "left", "val": 1, "steps": 1.5},
        {"arm": "left", "val": 1, "settle_steps": -1},
        {"arm": "left", "val": 1, "settle_steps": 61},
        {"arm": "left", "val": 1, "settle_steps": True},
        {"arm": "left", "val": 1, "settle_steps": 1.5},
    ],
)
def test_bad_tool_arguments_do_not_send_actions(gripper, kwargs):
    primitives, env, _ = gripper
    with pytest.raises(ValueError):
        primitives.set_gripper(**kwargs)
    assert env.actions == []


@pytest.mark.parametrize("termination", ["session", "flag"])
def test_terminal_state_prevents_further_gripper_commands(gripper, termination):
    primitives, env, _ = gripper
    if termination == "flag":
        env.terminated = True
        primitives.vla_act()
    else:
        env.ended = True
    before_count = len(env.actions)
    result = primitives.release(arm="left")
    assert result.is_error
    assert result.data["stop_reason"] == (
        "terminated" if termination == "flag" else "session_ended"
    )
    _assert_not_issued(result)
    assert result.data["requested_actions"] == 0
    assert len(env.actions) == before_count


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


def test_failed_rpc_is_not_replayed_and_clears_known_commands(gripper):
    primitives, env, _ = gripper
    primitives.vla_act()
    env.fail = True
    with pytest.raises(TimeoutError):
        primitives.release(arm="left")
    count = len(env.actions)
    env.fail = False
    result = primitives.release(arm="left")
    assert result.data["code"] == "command_state_unknown"
    _assert_not_issued(result)
    assert len(env.actions) == count


def test_invalid_final_measurement_preserves_real_execution_feedback(gripper):
    primitives, env, _ = gripper
    primitives.vla_act()
    env.invalid_after = True
    result = primitives.release(arm="left")
    assert result.is_error
    assert result.data["confirmed_actions"] == 20
    assert result.data["finger_qpos_after_m"] is None
    assert "post-gripper state" in result.data["error"]


@pytest.mark.parametrize("failure", ["timeout", "missing_counts", "invalid_counts"])
def test_unknown_first_rpc_cannot_be_bootstrapped_from_measurement(gripper, failure):
    primitives, env, model = gripper
    if failure == "timeout":
        env.fail = True
        with pytest.raises(TimeoutError):
            primitives.set_gripper(arm="left", val=0)
    else:
        env.feedback = {} if failure == "missing_counts" else {"issued_actions": True}
        primitives.set_gripper(arm="left", val=0)
    assert len(env.actions) == 1
    env.feedback = None
    env.fail = False
    result = primitives.set_gripper(arm="right", val=1)
    assert result.data["code"] == "command_state_unknown"
    _assert_not_issued(result)
    assert len(env.actions) == 1 and model.calls == 0


@pytest.mark.parametrize("has_prior_action", [False, True])
def test_confirmed_zero_issued_preserves_the_original_command_state(
    gripper, has_prior_action
):
    primitives, env, _ = gripper
    if has_prior_action:
        primitives.vla_act()
    env.feedback = {
        "issued_actions": 0,
        "confirmed_actions": 0,
        "unconfirmed_actions": 0,
        "executed_actions": 0,
        "observation_after_last_issued_action": False,
    }
    result = primitives.set_gripper(arm="left", val=1)
    _assert_not_issued(result)
    env.feedback = None
    retry = primitives.set_gripper(arm="left", val=0)
    assert retry.data["start_source"] == (
        "last_issued_native_action"
        if has_prior_action
        else "new_measured_position_hold"
    )
    assert retry.data["start_val"] == pytest.approx(0.2 if has_prior_action else 0.8)


def test_zero_issued_vla_does_not_clear_an_unknown_gripper_history(gripper):
    primitives, env, _ = gripper
    env.fail = True
    with pytest.raises(TimeoutError):
        primitives.set_gripper(arm="left", val=1)
    env.fail = False
    env.feedback = {"issued_actions": 0}
    primitives.vla_act()
    result = primitives.set_gripper(arm="left", val=1)
    assert result.data["code"] == "command_state_unknown"
    _assert_not_issued(result)
    assert len(env.actions) == 2


def test_historical_closing_command_is_preserved_under_contact(gripper):
    primitives, env, _ = gripper
    primitives.vla_act()
    # The left command targets 10 mm, but contact leaves the fingers at 40 mm.
    # A right-hand command must not infer 40 mm as a replacement left command.
    result = primitives.set_gripper(arm="right", val=0, steps=2)
    np.testing.assert_allclose(env.actions[-1][:, 14], -0.6)
    assert result.data["other_gripper_hold_source"] == "last_issued_native_action"
    assert result.data["other_gripper_native_hold_command"] == pytest.approx(-0.6)


@pytest.mark.parametrize(
    "val,measured,reached",
    [(1, [0.0499, 0.05], True), (0, [0.02, 0.02], False)],
)
def test_fresh_target_error_is_measured_and_does_not_establish_grasp_success(
    gripper, val, measured, reached
):
    primitives, env, _ = gripper
    primitives.vla_act()
    env.post_finger_qpos = measured
    result = primitives.set_gripper(arm="left", val=val)
    assert result.data["finger_qpos_after_m"] == measured
    np.testing.assert_allclose(
        result.data["finger_target_position_error_m"], np.asarray(measured) - 0.05 * val
    )
    assert result.data["finger_target_position_reached"] is reached
    assert result.data["grasp_success"] is None
    assert "success" not in result.data


def test_invalid_initial_observation_returns_zero_counts(gripper):
    primitives, env, model = gripper
    env.raw["r1::proprio"] = np.zeros(256)
    result = primitives.set_gripper(arm="left", val=1)
    assert result.data["code"] == "invalid_hold_state"
    _assert_not_issued(result)
    assert env.actions == [] and model.calls == 0


@pytest.mark.parametrize("arm,val", [("left", 1), ("right", 0)])
def test_settle_tail_repeats_target_and_preserves_all_other_commands(gripper, arm, val):
    primitives, env, model = gripper
    primitives.vla_act()
    result = primitives.set_gripper(arm=arm, val=val, steps=4)
    actions = env.actions[-1]
    selected = 14 if arm == "left" else 22
    assert actions.shape == (14, 23)
    np.testing.assert_array_equal(actions[4:], np.tile(actions[3], (10, 1)))
    np.testing.assert_array_equal(actions[3:, selected], 2 * val - 1)
    np.testing.assert_allclose(
        actions[:, 22 if arm == "left" else 14], 0.6 if arm == "left" else -0.6
    )
    assert result.data["ramp_steps"] == 4
    assert result.data["settle_steps"] == 10
    assert result.data["requested_actions"] == result.data["confirmed_actions"] == 14
    assert model.calls == 1


@pytest.mark.parametrize(
    "position,velocity,position_reached,settled",
    [
        ([0.0499, 0.05], [0.001, -0.001], True, True),
        ([0.0499, 0.05], [0.003, 0], True, False),
        ([0.048, 0.048], [0, 0], False, False),
    ],
)
def test_settled_requires_measured_position_and_velocity_not_just_a_hold_tail(
    gripper, position, velocity, position_reached, settled
):
    primitives, env, _ = gripper
    primitives.vla_act()
    env.post_finger_qpos = position
    env.post_finger_qvel = velocity
    result = primitives.set_gripper(arm="left", val=1)
    assert result.data["requested_actions"] == 20
    assert result.data["finger_qpos_after_m"] == position
    assert result.data["finger_qvel_after_m_s"] == velocity
    assert result.data["finger_target_position_reached"] is position_reached
    assert result.data["settled"] is settled
    assert result.data["grasp_success"] is None


def test_terminal_during_hold_does_not_invent_fresh_settled_feedback(gripper):
    primitives, env, _ = gripper
    primitives.vla_act()
    env.feedback = {
        "issued_actions": 12,
        "confirmed_actions": 11,
        "unconfirmed_actions": 1,
        "executed_actions": 11,
        "observation_after_last_issued_action": False,
        "stop_reason": "session_ended",
    }
    result = primitives.set_gripper(arm="left", val=1)
    assert result.data["requested_actions"] == 20
    assert result.data["issued_actions"] == 12
    assert result.data["confirmed_actions"] == 11
    assert result.data["unconfirmed_actions"] == 1
    assert result.data["settled"] is None
    assert result.data["finger_qpos_after_m"] is None
    assert result.data["finger_qvel_after_m_s"] is None
    refused = primitives.release(arm="right")
    _assert_not_issued(refused)
    assert refused.data["requested_actions"] == 0
    assert len(env.actions) == 2


def test_release_compatibility_accepts_explicit_settle_steps(gripper):
    primitives, env, _ = gripper
    result = primitives.release(arm="left", steps=2, settle_steps=3)
    assert result.data["ramp_steps"] == 2
    assert result.data["settle_steps"] == 3
    assert result.data["requested_actions"] == 5
    np.testing.assert_array_equal(env.actions[-1][2:, 14], 1)


@pytest.mark.parametrize("arm", ["left", "right"])
@pytest.mark.parametrize("joint", ["roll", "pitch", "yaw"])
def test_wrist_joint_ramps_one_joint_and_holds_everything_else(gripper, arm, joint):
    primitives, env, model = gripper
    primitives.vla_act()
    before = env.raw["r1::proprio"].copy()
    result = primitives.rotate_wrist_joint(
        arm=arm, joint=joint, delta_deg=30, steps=5, settle_steps=2
    )
    assert model.calls == 1 and len(env.actions) == 2
    actions = env.actions[-1]
    assert actions.shape == (7, 23)
    arm_slice = slice(7, 14) if arm == "left" else slice(15, 22)
    state_slice = slice(3, 10) if arm == "left" else slice(28, 35)
    index = arm_slice.start + WRIST_JOINT_OFFSET[joint]
    start = before[state_slice][WRIST_JOINT_OFFSET[joint]]
    ramp = start + math.radians(30) * np.arange(1, 6) / 5
    np.testing.assert_allclose(actions[:5, index], ramp, atol=1e-6)
    np.testing.assert_allclose(actions[5:, index], ramp[-1], atol=1e-6)
    # Every other channel is the hold: zero base, measured joints, last gripper commands.
    np.testing.assert_array_equal(actions[:, :3], 0)
    held = np.ones(23, bool)
    held[[0, 1, 2, index]] = False
    np.testing.assert_allclose(actions[:, held], np.tile(actions[0, held], (7, 1)))
    for action_slice, src in (
        (slice(3, 7), slice(53, 57)),
        (slice(7, 14), slice(3, 10)),
        (slice(15, 22), slice(28, 35)),
    ):
        expected = before[src].copy()
        if action_slice == arm_slice:
            expected[WRIST_JOINT_OFFSET[joint]] = ramp[-1]
        np.testing.assert_allclose(actions[-1, action_slice], expected, atol=1e-6)
    assert (
        result.data["joint_name"] == f"{arm}_arm_joint{WRIST_JOINT_OFFSET[joint] + 1}"
    )
    assert result.data["requested_delta_deg"] == 30
    assert result.data["commanded_delta_deg"] == pytest.approx(30)
    assert result.data["clipped_to_joint_limit"] is False
    assert result.data["issued_actions"] == result.data["confirmed_actions"] == 7
    assert "success" not in result.data


def test_wrist_joint_clips_to_the_urdf_limit_and_says_so(gripper):
    primitives, env, _ = gripper
    primitives.vla_act()
    # Right wrist pitch (joint6) is limited to about +-60 degrees; start is -0.5 rad.
    result = primitives.rotate_wrist_joint(
        arm="right", joint="pitch", delta_deg=90, steps=3, settle_steps=0
    )
    assert result.data["clipped_to_joint_limit"] is True
    low, high = result.data["joint_limit_rad"]
    assert env.actions[-1][-1, 15 + WRIST_JOINT_OFFSET["pitch"]] == pytest.approx(
        high, abs=1e-6
    )
    assert result.data["commanded_delta_deg"] < 90


def test_wrist_joint_refuses_bad_arguments_without_sending_actions(gripper):
    primitives, env, _ = gripper
    primitives.vla_act()
    sent = len(env.actions)
    for kwargs in (
        {"arm": "left", "joint": "elbow", "delta_deg": 10.0},
        {"arm": "left", "joint": "roll", "delta_deg": 91.0},
        {"arm": "left", "joint": "roll", "delta_deg": float("nan")},
        {"arm": "left", "joint": "roll", "delta_deg": 10.0, "steps": 0},
        {"arm": "left", "joint": "roll", "delta_deg": 10.0, "settle_steps": 61},
    ):
        with pytest.raises(ValueError):
            primitives.rotate_wrist_joint(**kwargs)
    assert len(env.actions) == sent


def test_wrist_joint_reports_measured_change_from_fresh_observation(gripper):
    primitives, env, _ = gripper
    primitives.vla_act()
    index = 28 + WRIST_JOINT_OFFSET["roll"]
    env.raw["r1::proprio"][index] = 0.2

    original = env.chunk_step

    def moved(actions):
        returned, a, b, c, info = original(actions)
        returned["r1::proprio"][index] = 0.2 + math.radians(20)
        return returned, a, b, c, info

    env.chunk_step = moved
    result = primitives.rotate_wrist_joint(
        arm="right", joint="roll", delta_deg=20, steps=2, settle_steps=0
    )
    assert result.data["measured_delta_deg"] == pytest.approx(20, abs=1e-6)
    assert result.data["observation_fresh"] is True
