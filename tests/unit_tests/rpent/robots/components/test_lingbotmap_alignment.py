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

"""Offline tests for fitting a reconstruction to pushed robot base poses."""

from __future__ import annotations

import numpy as np
import pytest

from rpent.robots.components.lingbotmap_server import (
    _apply_alignment,
    _estimate_base_alignment,
    _estimate_similarity,
    camera_to_world,
)


def _rotation(yaw: float, pitch: float) -> np.ndarray:
    """Rotation about ``z`` by ``yaw`` followed by ``x`` by ``pitch``."""
    cos_yaw, sin_yaw = np.cos(yaw), np.sin(yaw)
    cos_pitch, sin_pitch = np.cos(pitch), np.sin(pitch)
    about_z = np.array(
        [[cos_yaw, -sin_yaw, 0.0], [sin_yaw, cos_yaw, 0.0], [0.0, 0.0, 1.0]]
    )
    about_x = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, cos_pitch, -sin_pitch],
            [0.0, sin_pitch, cos_pitch],
        ]
    )
    return about_z @ about_x


def _circle_positions(count: int = 24, radius: float = 2.0) -> np.ndarray:
    angles = np.linspace(0.0, 2.0 * np.pi, count, endpoint=False)
    return np.stack(
        [radius * np.cos(angles), radius * np.sin(angles), np.zeros(count)], axis=1
    )


def _extrinsics_from_centres(
    centres: np.ndarray, world_to_camera: np.ndarray | None = None
) -> np.ndarray:
    """World-to-camera extrinsics whose camera centres are ``centres``."""
    rotation = np.eye(3) if world_to_camera is None else world_to_camera
    extrinsic = np.zeros((centres.shape[0], 3, 4))
    extrinsic[:, :3, :3] = rotation
    extrinsic[:, :3, 3] = -(centres @ rotation.T)
    return extrinsic


def _upright_camera_axes(model_to_robot: np.ndarray) -> np.ndarray:
    """Camera axes in the model frame for a camera mounted upright.

    The returned world-to-camera rotation has camera y (its second row) map to
    the robot's ``-z`` under ``model_to_robot``, so the fitted map is not
    upside down.
    """
    axes = np.stack([model_to_robot[0], -model_to_robot[2], model_to_robot[1]])
    assert np.linalg.det(axes) > 0.0
    return axes


def _poses(positions: np.ndarray) -> list[list[float]]:
    return [[float(x), float(y), 0.0, 0.0, 0.0, 0.0] for x, y, _ in positions]


def _model_centres(
    positions: np.ndarray, scale: float, rotation: np.ndarray, translation: np.ndarray
) -> np.ndarray:
    """Invert ``positions = scale * rotation @ centres + translation``."""
    return (positions - translation) @ rotation / scale


class TestEstimateSimilarity:
    def test_recovers_known_transform(self) -> None:
        source = _circle_positions()
        scale, rotation, translation = (
            2.5,
            _rotation(0.4, 1.1),
            np.array([1.0, -2.0, 3.0]),
        )
        target = scale * source @ rotation.T + translation

        fitted_scale, fitted_rotation, fitted_translation = _estimate_similarity(
            source, target
        )

        assert fitted_scale == pytest.approx(scale, rel=1e-9)
        np.testing.assert_allclose(fitted_rotation, rotation, atol=1e-9)
        np.testing.assert_allclose(fitted_translation, translation, atol=1e-9)


class TestEstimateBaseAlignment:
    def test_aligns_camera_trajectory_to_base_poses(self) -> None:
        positions = _circle_positions()
        # Upright cameras look towards the floor (negative pitch), so the
        # fitted map is not upside down.
        scale, rotation, translation = (
            2.5,
            _rotation(0.4, -1.1),
            np.array([1.0, -2.0, 3.0]),
        )
        centres = _model_centres(positions, scale, rotation, translation)

        alignment, reason = _estimate_base_alignment(
            _extrinsics_from_centres(centres, _upright_camera_axes(rotation)),
            _poses(positions),
        )

        assert reason == ""
        assert alignment is not None
        assert alignment.scale == pytest.approx(scale, rel=1e-9)
        assert alignment.frames == len(positions)
        assert alignment.rms_error_m == pytest.approx(0.0, abs=1e-9)
        np.testing.assert_allclose(alignment.rotation, rotation, atol=1e-9)
        np.testing.assert_allclose(
            _apply_alignment(centres, alignment), positions, atol=1e-9
        )

    def test_frame_without_base_pose_is_left_out_of_the_fit(self) -> None:
        positions = _circle_positions()
        poses: list[list[float] | None] = _poses(positions)
        poses[3] = None
        centres = _model_centres(positions, 2.5, _rotation(0.4, -1.1), np.zeros(3))

        alignment, reason = _estimate_base_alignment(
            _extrinsics_from_centres(centres), poses
        )

        assert reason == ""
        assert alignment is not None
        assert alignment.frames == len(positions) - 1
        assert alignment.unposed_frames == 1
        assert alignment.scale == pytest.approx(2.5, rel=1e-9)
        posed = [index for index, pose in enumerate(poses) if pose is not None]
        np.testing.assert_allclose(
            _apply_alignment(centres[posed], alignment),
            positions[posed],
            atol=1e-9,
        )

    def test_stationary_robot_leaves_map_unfitted(self) -> None:
        positions = np.zeros((24, 3))
        centres = _circle_positions()

        alignment, reason = _estimate_base_alignment(
            _extrinsics_from_centres(centres), _poses(positions)
        )

        assert alignment is None
        assert "below the 0.05 m" in reason

    def test_straight_line_leaves_roll_free(self) -> None:
        positions = np.zeros((24, 3))
        positions[:, 0] = np.linspace(0.0, 2.0, 24)
        centres = _circle_positions()

        alignment, reason = _estimate_base_alignment(
            _extrinsics_from_centres(centres), _poses(positions)
        )

        assert alignment is None
        assert "straight line" in reason

    def test_mirrored_trajectory_is_rejected(self) -> None:
        positions = _circle_positions()
        mirrored = positions * np.array([-1.0, 1.0, 1.0])
        centres = _model_centres(mirrored, 1.0, np.eye(3), np.zeros(3))

        alignment, reason = _estimate_base_alignment(
            _extrinsics_from_centres(centres), _poses(positions)
        )

        assert alignment is None
        assert "turns the reconstruction upside down" in reason

    def test_pose_count_mismatch_is_reported(self) -> None:
        centres = _circle_positions()
        alignment, reason = _estimate_base_alignment(
            _extrinsics_from_centres(centres), _poses(centres[:10])
        )

        assert alignment is None
        assert "10 poses were pushed for 24 frames" in reason


class TestCameraPoses:
    def test_camera_to_world_inverts_extrinsics(self) -> None:
        rotation = _rotation(0.3, -0.8)
        translation = np.array([0.5, 1.5, -2.0])
        extrinsic = np.zeros((2, 3, 4))
        extrinsic[:, :3, :3] = rotation
        extrinsic[:, :3, 3] = translation

        c2w = camera_to_world(extrinsic)

        np.testing.assert_allclose(
            c2w[:, :3, :3], np.tile(rotation.T, (2, 1, 1)), atol=1e-12
        )
        centres = -rotation.T @ translation
        np.testing.assert_allclose(c2w[:, :3, 3], np.tile(centres, (2, 1)), atol=1e-12)
        np.testing.assert_allclose(c2w[:, 3, :], np.tile([0, 0, 0, 1], (2, 1)))
