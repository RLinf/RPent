# Copyright 2026 The RPent Authors.
# Licensed under the Apache License, Version 2.0.
"""TCP inverse control and stopping after unacknowledged native actions."""

from types import SimpleNamespace

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from robots.rl.coordinates import Coordinates, grasp_axes
from robots.rl.robodojo import RoboDojoPrimitives
from robots.rl.toolkit import RLToolkit
from rpent.tools import ToolResult


@pytest.fixture
def primitive():
    obj = object.__new__(RoboDojoPrimitives)
    obj.coordinates = Coordinates(
        np.array([2.0, -1.0, 0.5]), Rotation.from_euler("z", 0.7).as_matrix()
    )
    obj.calibration = {
        "tools": {
            arm: {
                "offset_in_native_m": [0.15, 0.003, -0.002],
                "rotation_to_native": grasp_axes([0, 1, 0], [1, 0, 0]).tolist(),
            }
            for arm in ("left", "right")
        }
    }
    motion = {
        f"{arm}_eef_pose": [2.2, -0.9, 0.8, 1, 0, 0, 0] for arm in ("left", "right")
    }
    motion.update(left_gripper=0.5, right_gripper=0.5)
    obj.env_steps = 0
    obj._success = False
    obj.infrastructure_error = None
    obj._last_robot_state = None
    obj.env = SimpleNamespace(
        get_motion_state=lambda: motion,
        get_status=lambda: {"step": obj.env_steps, "success": False},
    )
    obj.saved = {}
    obj._state = SimpleNamespace(
        save=lambda name, data, **kw: obj.saved.update({name: data})
    )
    targets = []

    def move_to(*, arm, xyz, quat):
        targets.append((arm, xyz, quat))
        motion[f"{arm}_eef_pose"] = [*xyz, *quat]
        return ToolResult(
            data={
                "episode_status": {"step": 25, "success": False},
                "stop_reason": "completed",
            }
        )

    obj.native = SimpleNamespace(move_to=move_to)
    obj.robot_state()
    return obj, targets


@pytest.mark.parametrize("arm", ["left", "right"])
def test_absolute_grasp_pose_roundtrips_and_keeps_other_arm(primitive, arm):
    obj, targets = primitive
    other = "right" if arm == "left" else "left"
    initial_other = obj.env.get_motion_state()[f"{other}_eef_pose"].copy()
    goal = [0.2, -0.1, 0.4]
    angles = [2.8, -0.3, 0.2]
    result = obj.move_to(arm=arm, xyz=goal, orientation_rpy=angles)
    assert result.data["status"] == "reached"
    assert obj.env_steps == 25
    xyz, rotation = obj._pose(arm, obj.env.get_motion_state())
    np.testing.assert_allclose(xyz, goal, atol=1e-12)
    np.testing.assert_allclose(
        rotation, Rotation.from_euler("xyz", angles).as_matrix(), atol=1e-12
    )
    assert obj.env.get_motion_state()[f"{other}_eef_pose"] == initial_other
    # Changing grasp orientation also changes the wrist position: the offset
    # must be compensated in the target frame, not the old wrist orientation.
    obj.move_to(arm=arm, orientation_rpy=[2.0, 0.4, -0.3])
    np.testing.assert_allclose(
        obj._pose(arm, obj.env.get_motion_state())[0], goal, atol=1e-12
    )
    assert not np.allclose(targets[0][1], targets[1][1])


def test_uncertain_action_is_preserved_and_ends_before_another_request(primitive):
    from robots.rl.rl_episode import episode_outcome

    obj, _ = primitive
    feedback = {
        "stop_reason": "execution_unknown",
        "error": "ack lost",
        "execution_uncertain": True,
    }
    obj.native.set_gripper = lambda **kw: ToolResult(data=feedback.copy())
    obj.env.get_motion_state = lambda: pytest.fail(
        "must not query an untrusted execution state"
    )
    result = obj.set_gripper(arm="left", opening=0)
    assert result.is_error
    assert obj.saved["infrastructure_error.json"]["execution_uncertain"]
    assert obj.robot_state()["state_stale"]
    tk = object.__new__(RLToolkit)
    tk.environment, tk._primitives = "robodojo", obj
    assert tk.completion_result()["reason"] == "infrastructure_error"
    outcome = episode_outcome(SimpleNamespace(), tk, 50)
    assert outcome["termination_reason"] == "infrastructure_error"
    assert outcome["terminal"] is False


def test_native_planner_exception_latches_and_prevents_a_second_motion(primitive):
    obj, _ = primitive
    calls = []

    def failed_move(**kwargs):
        calls.append(kwargs)
        raise RuntimeError("CUDA illegal memory access")

    obj.native.move_to = failed_move
    with pytest.raises(RuntimeError, match="CUDA illegal"):
        obj.move_to(arm="left", xyz=[0.2, 0, 0.3])
    assert obj.saved["infrastructure_error.json"]["execution_uncertain"]
    with pytest.raises(RuntimeError, match="attempt has ended"):
        obj.move_to(arm="left", xyz=[0.2, 0, 0.3])
    assert len(calls) == 1
