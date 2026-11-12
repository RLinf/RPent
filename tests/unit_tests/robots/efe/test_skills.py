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

"""move_to_xyz closes the loop on public proprioception alone."""

from __future__ import annotations

import numpy as np
import pytest

from robots.efe.skills import EfeSkills, joint_layout
from tests.unit_tests.robots.efe.efe_support import FakeEfeClient, proprio


class MovingClient(FakeEfeClient):
    """Ideal robot: each submitted joint target moves the tool to the IK target."""

    def __init__(self) -> None:
        super().__init__()
        self.position = np.array([0.0, 0.0, 0.3])
        self.orientation = (1.0, 0.0, 0.0, 0.0)
        self.pending: np.ndarray | None = None
        self.ik_calls = 0
        self.ik_response: dict | None = None
        self.last_meta = {"proprio": proprio()}

    def observe(self, cameras=None, *, depth=False, video="frames"):
        result = super().observe(cameras=[], depth=False, video=video)
        result.meta["proprio"] = proprio(
            position=tuple(self.position),
            quaternion=self.orientation,
        )
        self.last_meta = result.meta
        return result

    def ik(self, targets, *, seed=None):
        self.ik_calls += 1
        if self.ik_response is not None:
            return self.ik_response
        self.pending = np.asarray(targets["tcp"][:3], dtype=np.float64)
        return {
            "ok": True,
            "q": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            "error": {"tcp": {"pos_m": 0.0, "rot_rad": 0.0}},
        }

    def submit(self, action_type, actions, *, dt=None, **kwargs):
        result = super().submit(action_type, actions, dt=dt, **kwargs)
        if self.pending is not None:
            self.position = self.pending
            self.pending = None
        return result


@pytest.fixture
def skills():
    client = MovingClient()
    return EfeSkills(client=client, check_cancelled=lambda: None), client


def test_joint_layout_rejects_a_mismatched_task():
    client = FakeEfeClient()
    client.spec["action_types"]["joint_position"]["layout"] = ["gripper", "waist"]

    with pytest.raises(ValueError, match="kinematics order"):
        joint_layout(client.spec)


def test_joint_layout_ignores_the_q_prefix_on_either_side():
    client = FakeEfeClient()
    joints = client.spec["kinematics"]["joint_names"]
    layout = client.spec["action_types"]["joint_position"]["layout"]
    client.spec["kinematics"]["joint_names"] = [
        name.removeprefix("q_") for name in joints
    ]
    client.spec["action_types"]["joint_position"]["layout"] = [
        f"q_{name}" for name in layout[:-1]
    ] + ["gripper"]

    joint_names, _ = joint_layout(client.spec)

    assert joint_names == [name.removeprefix("q_") for name in joints]


def test_move_to_xyz_reaches_a_target_in_bounded_steps(skills):
    service, client = skills

    result = service.move_to_xyz(0.0, 0.035, 0.3)

    assert not result.is_error, result.data
    assert result.data["ok"] is True
    assert result.data["position_error_m"] <= service.tolerance_m
    # 3.5 cm at a 2 cm step needs two corrections.
    assert client.ik_calls == 2
    assert client.submit_calls == 2
    assert all(row[1][0][-1] == 0.037 for row in client.submitted)
    assert all(request["video"] == "none" for request in client.observe_requests)


def test_move_to_xyz_does_nothing_when_already_at_the_target(skills):
    service, client = skills

    result = service.move_to_xyz(0.0, 0.0, 0.3)

    assert result.data["ok"] is True
    assert client.submit_calls == 0


def test_move_to_xyz_applies_a_gripper_target_when_already_at_the_position(skills):
    service, client = skills

    result = service.move_to_xyz(0.0, 0.0, 0.3, gripper=0.005)

    assert result.data["ok"] is True
    assert result.data["submit_calls"] == 1
    assert client.submitted[0][1][0][-1] == pytest.approx(0.005)


def test_move_to_xyz_skips_a_gripper_target_that_matches_the_command(skills):
    service, client = skills

    result = service.move_to_xyz(0.0, 0.0, 0.3, gripper=0.037)

    assert result.data["ok"] is True
    assert client.submit_calls == 0


def test_move_to_xyz_keeps_an_explicit_gripper_command(skills):
    service, client = skills

    service.move_to_xyz(0.0, 0.02, 0.3, gripper=0.005)

    assert client.submitted[0][1][0][-1] == pytest.approx(0.005)


def test_unreachable_target_is_reported_without_submitting():
    client = MovingClient()
    client.spec["kinematics"]["ik_targets"] = ["tcp"]
    client.ik_response = {
        "ok": True,
        "q": [0.0] * 6,
        "error": {"tcp": {"pos_m": 0.5, "rot_rad": 0.0}},
    }
    service = EfeSkills(client=client, check_cancelled=lambda: None)

    result = service.move_to_xyz(10.0, 0.0, 0.0)

    assert result.is_error
    assert "reachable workspace" in result.data["error"]
    assert client.submit_calls == 0


def test_failed_ik_is_reported_without_submitting():
    client = MovingClient()
    client.ik_response = {"ok": False, "error": "no solution"}
    service = EfeSkills(client=client, check_cancelled=lambda: None)

    result = service.move_to_xyz(0.0, 0.1, 0.0)

    assert result.is_error
    assert "IK did not converge" in result.data["error"]
    assert client.submit_calls == 0


def test_move_stops_when_the_episode_ends(skills):
    service, client = skills
    client.done = True
    client.done_reason = "success"

    result = service.move_to_xyz(0.0, 0.1, 0.0)

    assert result.data["done"] is True
    assert result.data["submit_calls"] == 0


def test_chunk_duration_follows_a_control_rate_the_default_does_not_fit():
    client = MovingClient()
    client.spec["control_hz"] = 24.0
    service = EfeSkills(client=client, check_cancelled=lambda: None)

    service.move_to_xyz(0.0, 0.0, 0.32)

    assert client.submitted
    for _, _, dt in client.submitted:
        assert dt is not None
        periods = dt * 24.0
        assert abs(periods - round(periods)) < 1e-6
        assert dt <= client.spec["max_chunk_s"]


def test_explicit_chunk_duration_must_fit_the_control_period():
    client = MovingClient()
    client.spec["control_hz"] = 24.0

    with pytest.raises(ValueError, match="whole multiple"):
        EfeSkills(client=client, check_cancelled=lambda: None, chunk_s=0.2)


def test_a_slow_robot_uses_one_control_step_per_chunk():
    client = MovingClient()
    client.spec["control_hz"] = 3.0

    service = EfeSkills(client=client, check_cancelled=lambda: None)

    assert service.chunk_s == pytest.approx(1 / 3)


def test_a_chunk_shorter_than_one_control_period_disables_the_skills():
    client = MovingClient()
    client.spec["control_hz"] = 3.0
    client.spec["max_chunk_s"] = 0.1

    with pytest.raises(ValueError, match="does not fit"):
        EfeSkills(client=client, check_cancelled=lambda: None)


def test_gripper_tolerance_follows_the_declared_travel(skills):
    service, client = skills

    limit = client.spec["action_types"]["joint_position"]["limits"]["gripper"]
    expected = 0.1 * (limit[1] - limit[0])

    assert service.gripper_tolerance_m == pytest.approx(expected)
    assert service.gripper_tolerance_m < service.tolerance_m


def test_a_blocked_gripper_is_reported_as_an_error():
    class BlockedGripperClient(MovingClient):
        def submit(self, action_type, actions, *, dt=None, **kwargs):
            result = super().submit(action_type, actions, dt=dt, **kwargs)
            result.meta["proprio"] = proprio(gripper=0.002)
            return result

    service = EfeSkills(client=BlockedGripperClient(), check_cancelled=lambda: None)

    result = service.set_gripper(0.037)

    assert result.data["ok"] is False
    assert result.is_error is True
    assert "Gripper reached" in result.data["error"]


def test_a_task_with_no_declared_gripper_travel_keeps_the_skills():
    """A missing gripper limit must not unregister the skills."""
    client = MovingClient()
    del client.spec["action_types"]["joint_position"]["limits"]["gripper"]

    service = EfeSkills(client=client, check_cancelled=lambda: None)

    assert service.gripper_tolerance_m > 0
    assert service.move_to_xyz(0.0, 0.0, 0.3).data["ok"] is True


def test_a_null_gripper_limit_keeps_the_skills():
    client = MovingClient()
    client.spec["action_types"]["joint_position"]["limits"]["gripper"] = [None, None]

    service = EfeSkills(client=client, check_cancelled=lambda: None)

    assert service.gripper_tolerance_m > 0


def test_the_tool_frame_must_be_an_ik_target():
    """``ik`` only accepts declared targets, so an untargeted frame has no skill."""
    client = MovingClient()
    client.spec["kinematics"]["tcp_frames"] = ["unlisted"]

    reason = EfeSkills.unavailable_reason(client.spec)

    assert reason is not None
    assert "ik_targets" in reason


def test_the_tool_pose_must_appear_in_proprioception():
    client = MovingClient()
    client.spec["proprio_layout"] = [
        name for name in client.spec["proprio_layout"] if not name.startswith("tcp_")
    ]

    reason = EfeSkills.unavailable_reason(client.spec)

    assert reason is not None
    assert "proprioception columns" in reason


def test_q_prefixed_joints_are_read_from_bare_proprioception():
    """The kinematics may strip ``q_`` while proprioception keeps it, or reverse."""
    client = MovingClient()
    client.spec["kinematics"]["joint_names"] = [
        name.removeprefix("q_") for name in client.spec["kinematics"]["joint_names"]
    ]

    service = EfeSkills(client=client, check_cancelled=lambda: None)

    result = service.move_to_xyz(0.0, 0.035, 0.3)

    assert not result.is_error, result.data
    assert result.data["ok"] is True


def test_an_episode_that_succeeds_mid_move_is_not_an_error():
    """The chunk that crosses the goal ends the episode before the arm arrives."""

    class GoalClient(MovingClient):
        def submit(self, action_type, actions, *, dt=None, **kwargs):
            result = super().submit(action_type, actions, dt=dt, **kwargs)
            # The service reports success while the tool is still short of the
            # requested position.
            self.pending = None
            self.done = True
            self.done_reason = "success"
            result.meta["done"] = True
            result.meta["reason"] = "success"
            return result

    service = EfeSkills(client=GoalClient(), check_cancelled=lambda: None)

    result = service.move_to_xyz(0.0, 0.09, 0.3)

    assert result.data["ok"] is True
    assert result.is_error is False
    assert result.data["done"] is True
    assert "after the episode succeeded" in result.data["note"]


def test_in_place_gripper_move_uses_the_gripper_tolerance(skills):
    """A small gripper delta at the target must not be submitted as a move."""
    service, client = skills
    # Half a tolerance away from the command the proprioception already reports.
    near = 0.037 - service.gripper_tolerance_m / 2

    result = service.move_to_xyz(0.0, 0.0, 0.3, gripper=near)

    assert result.data["ok"] is True
    assert client.submit_calls == 0
