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

"""Geometry tests with no robot or calibration dependency."""

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from rpent.utils.transforms import (
    compose_transforms,
    invert_transform,
    transform_points,
    transform_pose,
    transform_vectors,
)


def example_transform():
    matrix = np.eye(4)
    matrix[:3, :3] = Rotation.from_euler("z", 90, degrees=True).as_matrix()
    matrix[:3, 3] = [1, 2, 3]
    return matrix


def test_points_and_vectors_are_distinct():
    matrix = example_transform()
    np.testing.assert_allclose(transform_points(matrix, [1, 0, 0]), [1, 3, 3])
    np.testing.assert_allclose(
        transform_vectors(matrix, [1, 0, 0]), [0, 1, 0], atol=1e-12
    )
    points = np.array([[1, 0, 0], [0, 1, 0]])
    np.testing.assert_allclose(
        transform_points(matrix, points), [[1, 3, 3], [0, 2, 3]], atol=1e-12
    )
    assert transform_points(matrix, np.empty((0, 3))).shape == (0, 3)


def test_inverse_and_composition():
    matrix = example_transform()
    inverse = invert_transform(matrix)
    np.testing.assert_allclose(
        compose_transforms(matrix, inverse), np.eye(4), atol=1e-12
    )
    points = np.array([[0.2, -0.3, 0.4], [1, 2, 3]])
    np.testing.assert_allclose(
        transform_points(inverse, transform_points(matrix, points)), points
    )
    other = np.eye(4)
    other[:3, 3] = [0.5, 0, 0]
    np.testing.assert_allclose(
        transform_points(compose_transforms(matrix, other), points),
        transform_points(matrix, transform_points(other, points)),
    )


def test_pose_roundtrip_xyzw_and_input_unchanged():
    matrix = example_transform()
    pose = np.r_[[0.5, 0.2, 0.1], Rotation.from_euler("xyz", [0.2, 0.3, 0.4]).as_quat()]
    original = pose.copy()
    result = transform_pose(matrix, pose)
    np.testing.assert_allclose(result[:3], [0.8, 2.5, 3.1])
    expected = matrix[:3, :3] @ Rotation.from_quat(pose[3:]).as_matrix()
    np.testing.assert_allclose(Rotation.from_quat(result[3:]).as_matrix(), expected)
    restored = transform_pose(invert_transform(matrix), result)
    np.testing.assert_allclose(restored[:3], pose[:3])
    assert (
        Rotation.from_quat(restored[3:]) * Rotation.from_quat(pose[3:]).inv()
    ).magnitude() < 1e-12
    np.testing.assert_array_equal(pose, original)


@pytest.mark.parametrize(
    "bad", [np.zeros((4, 4)), np.eye(3), np.diag([2, 1, 1, 1]), np.full((4, 4), np.nan)]
)
def test_invalid_transform_rejected(bad):
    with pytest.raises(ValueError):
        transform_points(bad, [0, 0, 0])


@pytest.mark.parametrize("pose", [[0] * 7, [0] * 6, [np.nan, 0, 0, 0, 0, 0, 1]])
def test_invalid_pose_rejected(pose):
    with pytest.raises(ValueError):
        transform_pose(np.eye(4), pose)
