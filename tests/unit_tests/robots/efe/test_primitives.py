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

from robots.efe.primitives import EfePrimitives, joint_layout
from tests.efe_support import FakeEfeClient, proprio


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
def primitives():
    client = MovingClient()
    return EfePrimitives(client=client, check_cancelled=lambda: None), client


def test_joint_layout_rejects_a_mismatched_task():
    client = FakeEfeClient()
    client.spec["action_types"]["joint_position"]["layout"] = ["gripper", "waist"]

    with pytest.raises(ValueError, match="kinematics order"):
        joint_layout(client.spec)


def test_move_to_xyz_reaches_a_target_in_bounded_steps(primitives):
    service, client = primitives

    result = service.move_to_xyz(0.0, 0.035, 0.3)

    assert not result.is_error, result.data
    assert result.data["ok"] is True
    assert result.data["position_error_m"] <= service.tolerance_m
    # 3.5 cm at a 2 cm step needs two corrections.
    assert client.ik_calls == 2
    assert client.submit_calls == 2
    assert all(row[1][0][-1] == 0.037 for row in client.submitted)
    assert all(request["video"] == "none" for request in client.observe_requests)


def test_move_to_xyz_does_nothing_when_already_at_the_target(primitives):
    service, client = primitives

    result = service.move_to_xyz(0.0, 0.0, 0.3)

    assert result.data["ok"] is True
    assert client.submit_calls == 0


def test_move_to_xyz_keeps_an_explicit_gripper_command(primitives):
    service, client = primitives

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
    service = EfePrimitives(client=client, check_cancelled=lambda: None)

    result = service.move_to_xyz(10.0, 0.0, 0.0)

    assert result.is_error
    assert "reachable workspace" in result.data["error"]
    assert client.submit_calls == 0


def test_failed_ik_is_reported_without_submitting():
    client = MovingClient()
    client.ik_response = {"ok": False, "error": "no solution"}
    service = EfePrimitives(client=client, check_cancelled=lambda: None)

    result = service.move_to_xyz(0.0, 0.1, 0.0)

    assert result.is_error
    assert "IK did not converge" in result.data["error"]
    assert client.submit_calls == 0


def test_move_stops_when_the_episode_ends(primitives):
    service, client = primitives
    client.done = True
    client.done_reason = "success"

    result = service.move_to_xyz(0.0, 0.1, 0.0)

    assert result.data["done"] is True
    assert result.data["submit_calls"] == 0
