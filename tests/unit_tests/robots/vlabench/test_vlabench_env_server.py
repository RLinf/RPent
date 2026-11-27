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

"""Offline contracts for the VLABench environment server."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from robots.vlabench.sim.contracts import (
    VLABENCH_NATIVE_CAMERAS,
    check_env_meta,
    env_identity,
    validate_motion_receipt,
)
from robots.vlabench.sim.env_server import VLABenchEnvFacade

IDENTITY = {
    "task_name": "select_painting",
    "track": "track_1_in_distribution",
    "episode_index": 0,
    "seed": 42,
}


class _Timestep:
    def __init__(self, last: bool = False) -> None:
        self._last = last
        self.observation: dict = {}

    def last(self) -> bool:
        return self._last


class _FakeEnv:
    """Stand-in for a VLABench MuJoCo task with scripted termination."""

    def __init__(self, *, success_after: int | None = None) -> None:
        self.steps: list[np.ndarray] = []
        self.resets = 0
        self.success_after = success_after
        self.physics = object()
        self.robot = SimpleNamespace(get_qpos=lambda physics: np.zeros(7))
        self.task = SimpleNamespace(
            get_instruction=lambda: "pick the painting",
            should_terminate_episode=self._success,
        )

    def _success(self, physics: object) -> bool:
        return self.success_after is not None and len(self.steps) >= self.success_after

    def reset(self) -> None:
        self.resets += 1
        self.steps.clear()

    def step(self, action: np.ndarray) -> _Timestep:
        self.steps.append(np.asarray(action))
        return _Timestep()

    def close(self) -> None:
        pass


def _facade(
    env: _FakeEnv | None = None, *, limit: int = 10
) -> tuple[VLABenchEnvFacade, _FakeEnv]:
    env = env or _FakeEnv()
    facade = VLABenchEnvFacade(
        env, metadata=env_identity(**IDENTITY), max_episode_steps=limit
    )
    # These two read MuJoCo model data and need scipy; the contracts under
    # test do not depend on their values.
    facade._gravity_compensated_action = lambda action: np.asarray(action, dtype=float)
    facade.robot_model_state = lambda: {"gripper_qpos": np.full(2, 0.04)}
    return facade, env


def test_reset_starts_a_new_episode_with_a_fresh_budget() -> None:
    facade, env = _facade()
    first = facade.reset()
    assert env.resets == 1
    assert first["task_name"] == "select_painting"
    assert facade.get_task_language() == "pick the painting"
    assert facade.episode_budget() == {
        "step": 0,
        "step_limit": 10,
        "remaining_steps": 10,
    }
    facade.step(np.zeros(9))
    assert facade.episode_budget()["remaining_steps"] == 9
    second = facade.reset()
    assert second["episode_id"] != first["episode_id"]
    assert facade.episode_budget()["step"] == 0


@pytest.mark.parametrize("action", [np.zeros(8), np.full(9, np.nan)])
def test_step_rejects_malformed_joint_actions(action: np.ndarray) -> None:
    facade, env = _facade()
    with pytest.raises(ValueError):
        facade.step(action)
    assert env.steps == []


def test_budget_exhaustion_truncates_and_blocks_further_steps() -> None:
    facade, _ = _facade(limit=2)
    facade.step(np.zeros(9))
    facade.step(np.zeros(9))
    status = facade.task_status()
    assert status["truncated"] is True
    assert status["terminated"] is False
    assert status["episode_status"]["remaining_steps"] == 0
    with pytest.raises(RuntimeError, match="terminal"):
        facade.step(np.zeros(9))


def test_native_success_terminates_the_episode() -> None:
    facade, _ = _facade(_FakeEnv(success_after=1))
    facade.step(np.zeros(9))
    status = facade.task_status()
    assert status["task_success"] is True
    assert status["terminated"] is True
    assert status["truncated"] is False
    with pytest.raises(RuntimeError, match="terminal"):
        facade.step(np.zeros(9))


def test_gripper_runs_the_requested_steps_and_closes_the_fingers() -> None:
    facade, env = _facade()
    result = facade.execute_gripper(gripper=1, steps=3)
    assert result["ok"] is True
    assert result["steps_used"] == 3
    assert result["stop_reason"] == "completed"
    assert len(env.steps) == 3
    np.testing.assert_allclose(env.steps[0][7:], [0.0, 0.0])


def test_gripper_stops_at_native_success() -> None:
    facade, _ = _facade(_FakeEnv(success_after=2))
    result = facade.execute_gripper(gripper=1, steps=5)
    assert result["ok"] is False
    assert result["steps_used"] == 2
    assert result["stop_reason"] == "native_success"
    assert result["terminated"] is True


def test_cancelled_motion_executes_nothing_and_token_is_single_use() -> None:
    facade, env = _facade()
    token = facade.begin_motion()
    other = facade.cancel_motion(motion_id="other")
    assert other == {"motion_id": "other", "cancel_requested": False}
    assert facade.cancel_motion(motion_id=token)["cancel_requested"] is True
    result = facade.execute_gripper(gripper=-1, steps=3, motion_id=token)
    assert result["interrupted"] is True
    assert result["ok"] is False
    assert result["stop_reason"] == "cancelled"
    assert env.steps == []
    with pytest.raises(ValueError, match="stale"):
        facade.execute_gripper(gripper=-1, steps=3, motion_id=token)


@pytest.mark.parametrize(
    ("gripper", "width"), [(-1, 0.04), (1, 0.0), (0, 0.03), ("hold", 0.03)]
)
def test_finger_target_maps_the_command_to_a_width(gripper, width: float) -> None:
    target = VLABenchEnvFacade._finger_target(gripper, np.full(2, 0.03))
    np.testing.assert_allclose(target, [width, width])


@pytest.mark.parametrize("gripper", [True, 2, float("nan"), "close"])
def test_finger_target_rejects_invalid_commands(gripper) -> None:
    with pytest.raises(ValueError):
        VLABenchEnvFacade._finger_target(gripper, np.zeros(2))


def test_rpc_surface_separates_readonly_queries_from_motion() -> None:
    facade, _ = _facade()
    assert "env.cancel_motion" in facade._rpc
    assert "env.task_status" in facade._readonly_methods
    assert "env.sensor_snapshot" in facade._readonly_methods
    assert "env.execute_pose" not in facade._readonly_methods
    assert "env.execute_gripper" not in facade._readonly_methods


def test_check_env_meta_accepts_extra_provenance() -> None:
    meta = {**env_identity(**IDENTITY), "vlabench_commit": "abc"}
    assert check_env_meta(meta, **IDENTITY) == meta


def test_check_env_meta_rejects_a_different_episode() -> None:
    meta = env_identity(**{**IDENTITY, "episode_index": 1})
    with pytest.raises(ValueError, match="episode_index"):
        check_env_meta(meta, **IDENTITY)


def _gripper_receipt(**changes) -> dict:
    receipt = {
        "ok": True,
        "terminated": False,
        "truncated": False,
        "interrupted": False,
        "motion_id": "m",
        "steps_used": 3,
        "requested_steps": 3,
        "commands_completed": True,
        "stop_reason": "completed",
        "episode_status": {"step": 3, "step_limit": 10, "remaining_steps": 7},
    }
    return {**receipt, **changes}


def _pose_receipt(**changes) -> dict:
    receipt = {
        **_gripper_receipt(requested_steps=150, stop_reason="reached"),
        "ik_failed": False,
        "final_dist_m": 0.005,
        "final_orientation_error_rad": 0.01,
        "final_eef_pos": [0.1, 0.2, 0.3],
    }
    del receipt["commands_completed"]
    return {**receipt, **changes}


POSE_ARGS = {"max_steps": 150, "tol": 0.012, "ori_tol": 0.05}


def test_motion_receipts_from_the_server_are_accepted() -> None:
    validate_motion_receipt(
        _gripper_receipt(), "env.execute_gripper", {"steps": 3}, "m"
    )
    validate_motion_receipt(_pose_receipt(), "env.execute_pose", POSE_ARGS, "m")


@pytest.mark.parametrize(
    ("receipt", "match"),
    [
        (_gripper_receipt(motion_id="other"), "motion identity"),
        (_gripper_receipt(steps_used=4), "steps_used"),
        (_gripper_receipt(stop_reason="unknown"), "stop_reason"),
        (
            _gripper_receipt(
                episode_status={"step": 3, "step_limit": 10, "remaining_steps": 5}
            ),
            "inconsistent episode counters",
        ),
        (_gripper_receipt(interrupted=True), "interrupted motion"),
        (_gripper_receipt(terminated=True), "success termination"),
    ],
)
def test_inconsistent_gripper_receipts_are_rejected(receipt, match: str) -> None:
    with pytest.raises(RuntimeError, match=match):
        validate_motion_receipt(receipt, "env.execute_gripper", {"steps": 3}, "m")


def test_pose_success_must_agree_with_the_measured_error() -> None:
    receipt = _pose_receipt(final_dist_m=0.5)
    with pytest.raises(RuntimeError, match="disagrees with measured error"):
        validate_motion_receipt(receipt, "env.execute_pose", POSE_ARGS, "m")


class _FullFakeEnv:
    """Fake VLABench task exposing MuJoCo physics, camera, and Franka IK."""

    def __init__(
        self,
        *,
        advance_on_step: bool = True,
        fail_ik: bool = False,
        success_after: int | None = None,
        render_bad_depth: bool = False,
    ) -> None:
        self.steps: list[np.ndarray] = []
        self.advance_on_step = advance_on_step
        self.fail_ik = fail_ik
        self.success_after = success_after
        self.render_bad_depth = render_bad_depth
        self.eef_pos = np.array([0.0, 0.0, 0.0], dtype=float)
        self.eef_quat_wxyz = np.array([1.0, 0.0, 0.0, 0.0], dtype=float)
        self._target_pos = self.eef_pos.copy()
        self._target_quat = self.eef_quat_wxyz.copy()

        camera_ids = {
            name: idx for idx, name in enumerate(VLABENCH_NATIVE_CAMERAS.values())
        }
        joint_names = [f"arm_joint{i}" for i in range(7)]
        joint_ids = {name: i for i, name in enumerate(joint_names)}

        def name2id(name: str, kind: str) -> int:
            table = camera_ids if kind == "camera" else joint_ids
            return table.get(name, -1)

        model = SimpleNamespace(
            name2id=name2id,
            actuator_trnid=np.column_stack([np.arange(7), np.zeros(7, dtype=int)]),
            actuator_gainprm=np.full((7, 3), 100.0, dtype=float),
            actuator_biasprm=np.tile([0.0, -100.0, 0.0], (7, 1)),
            actuator_gear=np.ones((7, 6), dtype=float),
            actuator_forcelimited=np.array([True] + [False] * 6),
            actuator_forcerange=np.tile([-5.0, 5.0], (7, 1)),
            jnt_dofadr=np.arange(7, dtype=int),
        )
        data = SimpleNamespace(
            cam_xmat=np.tile(np.eye(3).reshape(9), (4, 1)),
            cam_xpos=np.array(
                [[0.5, 0.0, 0.8], [0.0, 0.5, 0.8], [0.0, -0.5, 0.8], [0.1, 0.0, 0.4]],
                dtype=float,
            ),
            qfrc_bias=np.array([10.0, 2.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=float),
        )

        def render(height: int, width: int, camera_id: int, depth: bool = False):
            if not depth:
                return np.full((height, width, 3), camera_id, dtype=np.uint8)
            values = np.full((height, width), 1.5, dtype=np.float32)
            if self.render_bad_depth:
                values[0, 0] = np.nan
                values[0, 1] = -0.5
                values[0, 2] = np.inf
            return values

        self.physics = SimpleNamespace(
            model=model,
            data=data,
            render=render,
            bind=lambda joint: SimpleNamespace(qpos=np.array([0.04])),
        )
        self.robot = SimpleNamespace(
            n_dof=7,
            joints=[SimpleNamespace(full_identifier=name) for name in joint_names],
            mjcf_model=SimpleNamespace(find=lambda kind, name: name),
            get_qpos=lambda physics: np.zeros(7),
            get_end_effector_pos=lambda physics: self.eef_pos.copy(),
            get_end_effector_quat=lambda physics: self.eef_quat_wxyz.copy(),
            get_qpos_from_ee_pos=self._ik,
        )
        self.task = SimpleNamespace(
            get_instruction=lambda: "pick the painting",
            should_terminate_episode=self._success,
        )

    def _ik(self, physics, pos, quat, inplace: bool, joint_names):
        del physics, inplace, joint_names
        self._target_pos = np.asarray(pos, dtype=float)
        self._target_quat = np.asarray(quat, dtype=float)
        if self.fail_ik:
            return False, np.zeros(7)
        return True, np.zeros(7)

    def _success(self, physics) -> bool:
        del physics
        return self.success_after is not None and len(self.steps) >= self.success_after

    def get_camera_matrix(self, camera_id: int, width: int, height: int):
        del camera_id
        k = np.array(
            [[width, 0.0, width / 2], [0.0, height, height / 2], [0.0, 0.0, 1.0]]
        )
        return k, np.eye(4)

    def reset(self) -> None:
        self.steps.clear()

    def step(self, action: np.ndarray) -> _Timestep:
        self.steps.append(np.asarray(action, dtype=float))
        if self.advance_on_step:
            self.eef_pos = self._target_pos.copy()
            self.eef_quat_wxyz = self._target_quat.copy()
        return _Timestep()

    def close(self) -> None:
        pass


def _full_facade(
    env: _FullFakeEnv | None = None, *, limit: int = 20
) -> tuple[VLABenchEnvFacade, _FullFakeEnv]:
    env = env or _FullFakeEnv()
    facade = VLABenchEnvFacade(
        env, metadata=env_identity(**IDENTITY), max_episode_steps=limit
    )
    return facade, env


def test_execute_pose_reaches_goal_and_validates_receipt() -> None:
    facade, _ = _full_facade()
    token = facade.begin_motion()
    args = {
        "xyz": [0.04, 0.0, 0.0],
        "quaternion_xyzw": [0.0, 0.0, 0.0, 1.0],
        "gripper": "hold",
        "max_steps": 10,
        "tol": 0.01,
        "step_clip": 0.02,
        "ori_tol": 0.05,
    }
    receipt = facade.execute_pose(**args, motion_id=token)
    validate_motion_receipt(receipt, "env.execute_pose", args, token)
    assert receipt["ok"] is True
    assert receipt["stop_reason"] == "reached"
    assert receipt["steps_used"] == 2
    assert receipt["final_dist_m"] <= 0.01


def test_execute_pose_reports_ik_failed_max_steps_and_terminal_states() -> None:
    ik_facade, _ = _full_facade(_FullFakeEnv(fail_ik=True))
    token = ik_facade.begin_motion()
    ik_args = {
        "xyz": [0.2, 0.0, 0.0],
        "quaternion_xyzw": [0.0, 0.0, 0.0, 1.0],
        "max_steps": 5,
        "tol": 0.01,
        "ori_tol": 0.05,
    }
    ik_res = ik_facade.execute_pose(**ik_args, motion_id=token)
    validate_motion_receipt(ik_res, "env.execute_pose", ik_args, token)
    assert ik_res["ok"] is False
    assert ik_res["ik_failed"] is True
    assert ik_res["stop_reason"] == "ik_failed"

    stalled_facade, _ = _full_facade(_FullFakeEnv(advance_on_step=False))
    token2 = stalled_facade.begin_motion()
    stalled_args = {**ik_args, "max_steps": 2}
    stalled_res = stalled_facade.execute_pose(**stalled_args, motion_id=token2)
    validate_motion_receipt(stalled_res, "env.execute_pose", stalled_args, token2)
    assert stalled_res["stop_reason"] == "max_steps"
    assert stalled_res["steps_used"] == 2

    success_facade, _ = _full_facade(_FullFakeEnv(success_after=1))
    token3 = success_facade.begin_motion()
    suc_res = success_facade.execute_pose(**ik_args, motion_id=token3)
    validate_motion_receipt(suc_res, "env.execute_pose", ik_args, token3)
    assert suc_res["stop_reason"] == "native_success"
    assert suc_res["terminated"] is True

    budget_facade, _ = _full_facade(_FullFakeEnv(), limit=1)
    token4 = budget_facade.begin_motion()
    bud_res = budget_facade.execute_pose(**ik_args, motion_id=token4)
    validate_motion_receipt(bud_res, "env.execute_pose", ik_args, token4)
    assert bud_res["stop_reason"] == "budget_exhausted"
    assert bud_res["truncated"] is True


def test_execute_pose_cancelled_and_invalid_inputs_rejected() -> None:
    facade, env = _full_facade()
    token = facade.begin_motion()
    facade.cancel_motion(motion_id=token)
    args = {
        "xyz": [0.1, 0.0, 0.0],
        "quaternion_xyzw": [0.0, 0.0, 0.0, 1.0],
        "max_steps": 5,
        "tol": 0.01,
        "ori_tol": 0.05,
    }
    cancelled = facade.execute_pose(**args, motion_id=token)
    validate_motion_receipt(cancelled, "env.execute_pose", args, token)
    assert cancelled["interrupted"] is True
    assert cancelled["stop_reason"] == "cancelled"
    assert env.steps == []

    with pytest.raises(ValueError, match="unit length"):
        facade.execute_pose(xyz=[0.0, 0.0, 0.0], quaternion_xyzw=[0.0, 0.0, 0.0, 2.0])
    with pytest.raises(ValueError, match="servo limits"):
        facade.execute_pose(
            xyz=[0.0, 0.0, 0.0], quaternion_xyzw=[0.0, 0.0, 0.0, 1.0], max_steps=0
        )


def test_sensor_snapshot_calibration_and_validation() -> None:
    facade, _ = _full_facade(_FullFakeEnv(render_bad_depth=True))
    snap = facade.sensor_snapshot(cameras=("agentview", "wrist"), height=32, width=32)
    assert len(snap["cameras"]) == 2
    first = snap["cameras"][0]
    assert first["camera"] == "agentview"
    assert first["rgb_top_down"].shape == (32, 32, 3)
    # Invalid / non-positive depth pixels are zeroed.
    assert first["depth_meters"][0, 0] == 0.0
    assert first["depth_meters"][0, 1] == 0.0
    assert first["depth_meters"][0, 2] == 0.0
    assert first["depth_meters"][1, 0] == pytest.approx(1.5)
    # MuJoCo +Y/+Z axes are flipped to metric +Y down/+Z forward.
    np.testing.assert_allclose(
        first["calibration"]["extrinsic_cam2world"][:3, :3], np.diag([1.0, -1.0, -1.0])
    )
    with pytest.raises(ValueError, match="unsupported VLABench camera"):
        facade.sensor_snapshot(cameras=("unknown",), height=32, width=32)
    with pytest.raises(ValueError, match="unique"):
        facade.sensor_snapshot(cameras=("agentview", "agentview"), height=32, width=32)
    with pytest.raises(ValueError, match=">= 32"):
        facade.sensor_snapshot(cameras=("agentview",), height=16, width=32)


def test_gravity_compensation_clips_torques_and_tracks_stats() -> None:
    facade, _ = _full_facade()
    compensated = facade._gravity_compensated_action(np.zeros(9, dtype=float))
    # Joint 0 bias is 10 Nm clipped to [-5, 5], divided by gain 100 -> 0.05 rad.
    assert compensated[0] == pytest.approx(0.05)
    # Joint 1 bias is 2 Nm (within limits), divided by gain 100 -> 0.02 rad.
    assert compensated[1] == pytest.approx(0.02)
    assert facade._gravity_stats["steps"] == 1
    assert facade._gravity_stats["force_limited_joint_steps"] == 1
    assert facade._gravity_stats["max_abs_compensation_torque_nm"] == pytest.approx(5.0)
    with pytest.raises(ValueError, match="9-vector"):
        facade._gravity_compensated_action(np.zeros(8))
