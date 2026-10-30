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
"""Offline tests for R2 primitives, state capture, perception, and the toolkit."""

from __future__ import annotations

import numpy as np
import pytest

from robots.r2.toolkit import R2Toolkit
from robots.r2.tools import (
    R2Primitives,
    back_project,
    dump_state,
    segment,
    view_env_state,
)
from rpent.dashboard.events import NullDashboardEventSink
from rpent.memory import MemoryManager
from rpent.robots.components.sam3_client import Sam3Result
from rpent.session import EnvState
from rpent.utils.logging import init_output_dir

#: Head camera 1 m up at the origin, looking along +x (optical z forward).
_HEAD_POSE = [[0, 0, 1, 0], [-1, 0, 0, 0], [0, -1, 0, 1], [0, 0, 0, 1]]


class FakeClient:
    def __init__(self, *, base_motion=False, intrinsics=True, distortion=None) -> None:
        self.meta = {"robot": "r2", "protocol": 1, "base_motion": base_motion}
        self.calls: list[tuple] = []
        self.intrinsics = (
            {"fx": 100.0, "fy": 100.0, "cx": 3.0, "cy": 2.0} if intrinsics else None
        )
        self.distortion = distortion

    def observe(self):
        depth = np.full((4, 6), 2000, dtype=np.uint16)
        depth[0, 0] = 0
        return {
            "robot_state": {"frame": "base_link", "arms": {}},
            "views": {
                "head": {"rgb": np.zeros((4, 6, 3), dtype=np.uint8), "depth_mm": depth},
                "wrist_left": {"error": "camera 'wrist_left' has delivered no frame"},
            },
            "camera_meta": {
                "frame": "base_link",
                "cameras": {
                    "head": {
                        "image_shape": [4, 6],
                        "intrinsics": self.intrinsics,
                        "distortion": self.distortion,
                        "T_base_camera": _HEAD_POSE,
                    }
                },
            },
        }

    def __getattr__(self, name):
        def call(*args):
            self.calls.append((name, *args))
            return {"ok": True}

        return call


def test_primitives_forward_bounded_motions_to_the_env_server():
    client = FakeClient()
    primitives = R2Primitives(env=client, check_cancelled=lambda: None)

    primitives.move_to("left", [0.4, 0.2, 0.9])
    primitives.move_delta("right", [0.0, 0.0, 0.05])
    primitives.set_gripper("left", 0.0)
    primitives.look(0.0, -0.5)

    assert client.calls == [
        ("move_to", "left", [0.4, 0.2, 0.9], None),
        ("move_delta", "right", [0.0, 0.0, 0.05]),
        ("set_gripper", "left", 0.0),
        ("look", 0.0, -0.5),
    ]


def test_dump_state_records_images_depth_camera_meta_and_camera_errors(tmp_path):
    state = EnvState(tmp_path)
    record = dump_state(
        R2Primitives(env=FakeClient(), check_cancelled=lambda: None),
        state,
        command={"action": "look"},
        result={"ok": True},
        elapsed_s=0.1,
    )

    assert {"head.png", "head_depth_mm.npy", "camera_meta.json"} <= set(
        record.artifacts
    )
    assert record.state["camera_errors"] == {
        "wrist_left": "camera 'wrist_left' has delivered no frame"
    }
    view = view_env_state(record.step_idx, state=state)
    assert view.data["images"] == ["head"] and len(view.images) == 1


def test_back_project_maps_pixels_through_depth_intrinsics_and_camera_pose(tmp_path):
    state = EnvState(tmp_path)
    dump_state(
        R2Primitives(env=FakeClient(), check_cancelled=lambda: None),
        state,
        command=None,
        result=None,
        elapsed_s=None,
    )

    result = back_project(
        [[2, 3], [3, 5], [0, 0], [9, 9]], neighborhood=0, state=state
    ).data

    centre, right_low, no_depth, outside = result["points"]
    # The principal point lies 2 m straight ahead of the camera.
    assert centre["xyz"] == pytest.approx([2.0, 0.0, 1.0])
    # 2 px right and 1 px down at 2 m and f=100: 4 cm right (-y), 2 cm down.
    assert right_low["xyz"] == pytest.approx([2.0, -0.04, 0.98])
    assert no_depth["error"] == "no valid depth around this pixel"
    assert "outside" in outside["error"]


def test_back_project_inverts_the_lens_distortion(tmp_path):
    k1, k2, p1, p2, k3 = coefficients = [-0.3, 0.1, 0.002, -0.001, -0.02]
    state = EnvState(tmp_path)
    dump_state(
        R2Primitives(
            env=FakeClient(distortion=coefficients), check_cancelled=lambda: None
        ),
        state,
        command=None,
        result=None,
        elapsed_s=None,
    )

    xyz = back_project([[3, 5]], neighborhood=0, state=state).data["points"][0]["xyz"]

    # Project the point back through OpenCV's forward model onto the pixel.
    camera = np.linalg.inv(np.asarray(_HEAD_POSE)) @ np.array([*xyz, 1.0])
    x, y = camera[0] / camera[2], camera[1] / camera[2]
    r2 = x * x + y * y
    radial = 1 + k1 * r2 + k2 * r2**2 + k3 * r2**3
    col = 100.0 * (x * radial + 2 * p1 * x * y + p2 * (r2 + 2 * x * x)) + 3.0
    row = 100.0 * (y * radial + p1 * (r2 + 2 * y * y) + 2 * p2 * x * y) + 2.0
    assert (row, col) == pytest.approx((3.0, 5.0), abs=1e-3)
    assert camera[2] == pytest.approx(2.0, abs=1e-4)


def test_back_project_names_the_missing_calibration(tmp_path):
    state = EnvState(tmp_path)
    dump_state(
        R2Primitives(env=FakeClient(intrinsics=False), check_cancelled=lambda: None),
        state,
        command=None,
        result=None,
        elapsed_s=None,
    )

    result = back_project([[2, 3]], state=state).data

    assert "intrinsics" in result["error"]
    assert "pose" not in result["error"] and "depth" not in result["error"]


class FakeSam3:
    def __init__(self, found=True) -> None:
        self.found = found
        self.calls: list[dict] = []

    def segment(self, image, *, text_prompt=None, point=None, min_score=0.2):
        self.calls.append({"prompt": text_prompt, "point": point, "png": image[:4]})
        if not self.found:
            return Sam3Result(found=False, reason="no match for the prompt")
        mask = np.zeros((4, 6), dtype=bool)
        mask[1:4, 1:6] = True
        return Sam3Result(found=True, score=0.91, mask=mask, mask_shape=(4, 6))


def _recorded(tmp_path, **client):
    state = EnvState(tmp_path)
    dump_state(
        R2Primitives(env=FakeClient(**client), check_cancelled=lambda: None),
        state,
        command=None,
        result=None,
        elapsed_s=None,
    )
    return state


def test_segment_locates_the_mask_and_records_an_overlay(tmp_path):
    state = _recorded(tmp_path)
    sam3 = FakeSam3()

    result = segment(prompt=" red cup ", state=state, sam3_client=sam3)

    assert sam3.calls == [{"prompt": "red cup", "point": None, "png": b"\x89PNG"}]
    data = result.data
    assert (data["found"], data["score"], data["mask_pixels"]) == (True, 0.91, 15)
    assert data["pixel_box"] == [1, 1, 3, 5] and data["pixel_centre"] == [2, 3]
    # Every mask pixel is 2 m ahead; the median pixel is the principal point.
    assert data["xyz"] == pytest.approx([2.0, 0.0, 1.0])
    assert data["extent_low"][1] < 0.0 < data["extent_high"][1]
    assert state.exists(data["overlay"], step=0) and len(result.images) == 1


def test_segment_without_calibration_still_returns_the_pixels(tmp_path):
    state = _recorded(tmp_path, intrinsics=False)

    data = segment(point=[2, 3], state=state, sam3_client=FakeSam3()).data

    assert data["pixel_centre"] == [2, 3]
    assert "xyz" not in data and "intrinsics" in data["xyz_error"]


def test_segment_reports_a_miss_and_needs_exactly_one_query(tmp_path):
    state = _recorded(tmp_path)

    missed = segment(prompt="cup", state=state, sam3_client=FakeSam3(found=False))
    neither = segment(state=state, sam3_client=FakeSam3())
    both = segment(prompt="cup", point=[1, 1], state=state, sam3_client=FakeSam3())

    assert missed.data == {
        "step": 0,
        "camera": "head",
        "found": False,
        "reason": "no match for the prompt",
    }
    assert neither.data["error"] == both.data["error"]
    assert neither.data["error"] == "give exactly one of prompt or point"


@pytest.mark.parametrize(("base_motion", "sam3"), [(False, None), (True, FakeSam3())])
def test_toolkit_offers_optional_tools_only_when_configured(
    tmp_path, base_motion, sam3
):
    init_output_dir(tmp_path / "run")
    toolkit = R2Toolkit(
        runtime_kwargs={
            "env": FakeClient(base_motion=base_motion),
            "sam3_client": sam3,
        },
        dashboard_events=NullDashboardEventSink(),
        memory=MemoryManager(root=tmp_path / "memory"),
    )

    names = {tool.name for tool in toolkit.list_tools()}

    assert {"move_to", "back_project", "view_env_state", "go_home"} <= names
    assert ("move_base" in names) is base_motion
    assert ("segment" in names) is (sam3 is not None)
    assert toolkit.state.get(0).state["robot_info"]["robot"] == "r2"
    moved = toolkit.execute_tool(
        "move_delta", {"arm": "left", "delta_xyz": [0.0, 0.0, 0.01]}
    )
    assert moved.data["step_idx"] == 1 and moved.images
    assert toolkit.execute_tool("set_gripper", {"arm": "left", "opening": 2.0}).is_error
