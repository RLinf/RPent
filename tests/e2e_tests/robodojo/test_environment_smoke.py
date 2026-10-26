# Copyright 2026 The RPent Authors.
"""Opt-in real RoboDojo primitive checks; no VLA or SAM3 is needed."""

import os
from pathlib import Path

import numpy as np
import pytest

from robots.robodojo.robot_spec import get_robot_spec
from robots.robodojo.toolkit import RoboDojoToolkit
from rpent.dashboard.events import NullDashboardEventSink
from rpent.memory import MemoryManager
from tests.e2e_tests.common import parse_runtime_args, require_rgb, runtime_phase

pytestmark = [
    pytest.mark.timeout(900),
    pytest.mark.skipif(
        os.environ.get("RPENT_RUN_ROBODOJO_INTEGRATION") != "1",
        reason="requires an explicitly reserved GPU and RoboDojo assets",
    ),
]


def check_pose_transition(before, after, *, z_delta=0.0, yaw_deg=0.0):
    """Check measured motion; a no-op must not pass the small-move check."""
    before, after = np.asarray(before), np.asarray(after)
    np.testing.assert_allclose(after[:3], before[:3] + [0, 0, z_delta], atol=0.005)
    half = np.deg2rad(yaw_deg) / 2
    c, s = np.cos(half), np.sin(half)
    w, x, y, z = before[3:]
    expected = np.array([c * w - s * z, c * x - s * y, c * y + s * x, c * z + s * w])
    expected /= np.linalg.norm(expected)
    actual = after[3:] / np.linalg.norm(after[3:])
    angle_error = 2 * np.arccos(np.clip(abs(np.dot(expected, actual)), 0, 1))
    assert angle_error < np.deg2rad(0.5), angle_error


def test_environment_primitives(monkeypatch):
    from rpent.utils import logging

    output = Path(os.environ["RPENT_E2E_OUTPUT_DIR"])
    monkeypatch.setattr(logging, "_output_dir", output)
    spec = get_robot_spec()
    args = parse_runtime_args(
        spec, ["--task", "put_bottles_into_dustbin", "--seed", "0", "--layout", "0"]
    )
    with runtime_phase(spec, args, output / "runtime", {"env"}) as kwargs:
        env = kwargs["env"]
        toolkit = RoboDojoToolkit(
            primitives_kwargs=kwargs,
            dashboard_events=NullDashboardEventSink(),
            memory=MemoryManager(output / "memory"),
            state_output_dir=output / "states",
        )
        try:
            initial = env.get_motion_state()
            assert env.policy_observation()["inference_index"] == 0
            record = toolkit._state.latest_record()
            for view in ("head", "left_wrist", "right_wrist"):
                rgb = require_rgb(toolkit._state.load(f"cam_{view}.png"), view)
                world = toolkit._state.load(f"{view}_world_xyz.npy")
                assert world.shape == (*rgb.shape[:2], 3)
                valid = np.argwhere(np.isfinite(world).all(axis=-1))
                assert len(valid)
                pixel = valid[len(valid) // 2].tolist()
                result = toolkit.execute_tool(
                    "sample_world_xyz",
                    {"view": view, "pixels": [pixel], "neighborhood": 0},
                ).data
                np.testing.assert_allclose(
                    result["samples"][0]["xyz"], world[tuple(pixel)]
                )
            for arm in ("left", "right"):
                measurements = []
                for value in (0.0, 0.5, 1.0):
                    before = env.get_motion_state()["targets"]
                    result = toolkit.execute_tool(
                        "set_gripper", {"arm": arm, "val": value}
                    )
                    assert not result.is_error, result.data
                    execution = result.data["log"]["result"]
                    assert execution["completed"] and execution["executed_steps"] == 10
                    measurements.append(env.get_motion_state()[f"{arm}_gripper"])
                    after = env.get_motion_state()["targets"]
                    for key in before:
                        if key != f"{arm}_ee_joint_state":
                            np.testing.assert_allclose(
                                after[key], before[key], atol=1e-6
                            )
                assert measurements[0] < measurements[1] < measurements[2], measurements
                assert measurements[2] - measurements[0] > 0.5, measurements
                pose = env.get_motion_state()[f"{arm}_eef_pose"]
                target = np.asarray(pose[:3]) + [0, 0, 0.015]
                result = toolkit.execute_tool(
                    "move_to", {"arm": arm, "xyz": target.tolist(), "substeps": 25}
                )
                assert not result.is_error, result.data
                execution = result.data["log"]["result"]
                assert execution["completed"] and execution["executed_steps"] > 0, (
                    execution
                )
                moved = env.get_motion_state()[f"{arm}_eef_pose"]
                check_pose_transition(pose, moved, z_delta=0.015)
                assert execution["final_dist_m"] < 0.005, execution
                result = toolkit.execute_tool(
                    "rotate_wrist", {"arm": arm, "delta_yaw_deg": 3}
                )
                assert not result.is_error, result.data
                assert result.data["log"]["result"]["completed"], result.data
                check_pose_transition(
                    moved, env.get_motion_state()[f"{arm}_eef_pose"], yaw_deg=3
                )
            # Exercise VPP2's native EE route with a measured hold pose, without
            # loading a policy. Analytic execution must already be in its history.
            policy_obs = env.policy_observation()
            assert policy_obs["inference_index"] == 1
            assert env._history_step == env.get_status()["step"]
            state = policy_obs["state"]
            obs, _, _, info = env.step(
                {
                    "left_ee_pose": state[:7],
                    "left_ee_joint_state": state[7:8],
                    "right_ee_pose": state[8:15],
                    "right_ee_joint_state": state[15:16],
                }
            )
            assert info["executed_actions"] == 1
            latest = env.policy_observation()
            for rgb, camera in zip(
                latest["history"][-1], ("cam_head", "cam_left_wrist", "cam_right_wrist")
            ):
                np.testing.assert_array_equal(rgb, obs["vision"][camera]["color"])
            assert env._history_step == env.get_status()["step"]
            env.reset()
            reset_input = env.policy_observation()
            assert reset_input["inference_index"] == 0 and env._history_step == 0
            assert toolkit._state.get(record.step_idx).step_idx == record.step_idx
            final = env.get_motion_state()
            for arm in ("left", "right"):
                np.testing.assert_allclose(
                    final[f"{arm}_eef_pose"][:3],
                    initial[f"{arm}_eef_pose"][:3],
                    atol=0.02,
                )
            finish = toolkit.execute_tool(
                "finish", {"status": "stuck", "summary": "bounded primitive checks"}
            )
            assert finish.data["_finish"]
        finally:
            toolkit.close()
