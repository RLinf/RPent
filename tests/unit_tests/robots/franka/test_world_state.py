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

"""Offline tests for model-free Franka WorldState generation."""

from __future__ import annotations

import numpy as np
from robots.franka.world_state import (
    ClassicalWorldStateBuilder,
    PerceptionConfig,
    WorkspaceBounds,
)

from robots.franka.perception import project_depth_pixels_to_base


def _calibration() -> dict:
    # Camera optical +z points down toward a table at base z=0.2.
    camera_to_base = np.eye(4)
    camera_to_base[:3, :3] = np.diag([1.0, -1.0, -1.0])
    camera_to_base[2, 3] = 1.0
    return {
        "external": {"matrix": camera_to_base},
        "wrist": {"matrix": np.eye(4)},
    }


def _camera_meta() -> dict:
    return {
        "depth_unit": "m",
        "observation_camera_map": {"main": "wrist", "extra_0": "external"},
        "cameras": {
            "wrist": {"intrinsic_K": [[100, 0, 40], [0, 100, 40], [0, 0, 1]]},
            "external": {"intrinsic_K": [[100, 0, 40], [0, 100, 40], [0, 0, 1]]},
        },
    }


def _robot_state() -> dict:
    return {
        "raw_base_state": {
            "tcp_pose": [0.0, 0.0, 0.55, 0.0, 0.0, 0.0, 1.0],
            "gripper_open": True,
        }
    }


def _scene(offset: int = 0) -> tuple[np.ndarray, np.ndarray]:
    image = np.full((80, 80, 3), 120, dtype=np.uint8)
    depth = np.full((80, 80), 0.8, dtype=np.float32)
    image[25:45, 20 + offset : 40 + offset] = [230, 35, 35]
    depth[25:45, 20 + offset : 40 + offset] = 0.68
    return image, depth


def _builder() -> ClassicalWorldStateBuilder:
    return ClassicalWorldStateBuilder(
        workspace=WorkspaceBounds(minimum=(-0.5, -0.5, 0.05), maximum=(0.5, 0.5, 0.7)),
        config=PerceptionConfig(
            pixel_stride=2,
            min_table_points=100,
            min_component_points=20,
        ),
        calibration=_calibration(),
        calibration_id="test-calibration",
    )


def test_bulk_projection_preserves_invalid_entries():
    depth = np.array([[0.8, 0.0], [np.nan, 0.7]], dtype=np.float32)
    points, valid = project_depth_pixels_to_base(
        depth=depth,
        rows=np.array([0, 0, 1, 1]),
        cols=np.array([0, 1, 0, 1]),
        meta={
            "depth_unit": "m",
            "observation_camera_map": {"extra_0": "external"},
            "cameras": {"external": {"intrinsic_K": np.eye(3).tolist()}},
        },
        camera="third_person",
        tcp_pose=[0, 0, 0, 0, 0, 0, 1],
        calibration=_calibration(),
    )

    assert valid.tolist() == [True, False, False, True]
    np.testing.assert_allclose(points[0], [0.0, 0.0, 0.2])
    assert np.isnan(points[1]).all()
    np.testing.assert_allclose(points[3], [0.7, -0.7, 0.3])


def test_world_state_tracks_geometry_and_freezes_parameters():
    builder = _builder()
    image, depth = _scene()
    first = builder.update(
        rgb=image,
        depth=depth,
        camera_meta=_camera_meta(),
        robot_state=_robot_state(),
        captured_at_s=10.0,
    )
    moved_image, moved_depth = _scene(offset=2)
    second = builder.update(
        rgb=moved_image,
        depth=moved_depth,
        camera_meta=_camera_meta(),
        robot_state=_robot_state(),
        captured_at_s=10.5,
    )

    assert first["source"]["simulator_object_state_used"] is False
    assert first["source"]["learned_model_used"] is False
    assert len(first["objects"]) == 1
    assert second["objects"][0]["track_id"] == first["objects"][0]["track_id"]
    assert second["objects"][0]["semantic_label"] is None
    assert second["objects"][0]["appearance"]["coarse_color"] == "red"
    assert second["perception_quality"]["camera_skew_ms"] is None

    approaches = [
        item
        for item in second["parameter_candidates"]
        if item["tool"] == "move_delta" and item.get("subject_track_id")
    ]
    assert len(approaches) == 1
    candidate = approaches[0]
    assert candidate["based_on_state_id"] == second["state_id"]
    assert np.linalg.norm(candidate["args"]["delta_xyz"]) <= 0.02001
    assert candidate["resolver_version"] == "classical-v1"


def test_missing_geometry_suppresses_object_motion_candidates():
    builder = _builder()
    image = np.zeros((80, 80, 3), dtype=np.uint8)
    depth = np.zeros((80, 80), dtype=np.float32)
    world = builder.update(
        rgb=image,
        depth=depth,
        camera_meta=_camera_meta(),
        robot_state=_robot_state(),
        captured_at_s=20.0,
    )

    assert world["objects"] == []
    assert world["perception_quality"]["status"] == "insufficient_geometry"
    assert {
        item["candidate_id"].rsplit("/", 1)[-1]
        for item in world["parameter_candidates"]
    } == {
        "hold",
        "refresh",
        "retreat-up",
    }
