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

"""Rigid transforms, independent of robot frames and calibration storage.

``T_target_source`` maps source-frame coordinates into the target frame.
Poses use xyz followed by an xyzw quaternion; translations use metres.
"""

import numpy as np
from scipy.spatial.transform import Rotation


def _rigid_matrix(transform: np.ndarray) -> np.ndarray:
    matrix = np.asarray(transform, dtype=np.float64)
    if matrix.shape != (4, 4) or not np.isfinite(matrix).all():
        raise ValueError("expected a finite 4x4 rigid transform")
    rotation = matrix[:3, :3]
    if not (
        np.allclose(matrix[3], [0, 0, 0, 1], atol=1e-7)
        and np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-6)
        and np.isclose(np.linalg.det(rotation), 1, atol=1e-6)
    ):
        raise ValueError("expected an SE(3) transform (proper rotation, no scaling)")
    return matrix


def transform_vectors(transform: np.ndarray, vectors: np.ndarray) -> np.ndarray:
    """Rotate direction/displacement vectors (..., 3), without translation."""
    matrix = _rigid_matrix(transform)
    values = np.asarray(vectors, dtype=np.float64)
    if values.ndim < 1 or values.shape[-1] != 3 or not np.isfinite(values).all():
        raise ValueError("expected finite vectors with shape (..., 3)")
    return values @ matrix[:3, :3].T


def transform_points(transform: np.ndarray, points: np.ndarray) -> np.ndarray:
    """Transform one point (3,) or a batch (..., 3), including translation."""
    matrix = _rigid_matrix(transform)
    return transform_vectors(matrix, points) + matrix[:3, 3]


def transform_pose(transform: np.ndarray, pose: np.ndarray) -> np.ndarray:
    """Transform a single xyz+xyzw pose without modifying the input."""
    matrix = _rigid_matrix(transform)
    values = np.asarray(pose, dtype=np.float64)
    if values.shape != (7,) or not np.isfinite(values).all():
        raise ValueError("expected a finite xyz+xyzw pose with shape (7,)")
    if np.linalg.norm(values[3:]) < 1e-12:
        raise ValueError("pose quaternion must be nonzero")
    orientation = Rotation.from_matrix(matrix[:3, :3]) * Rotation.from_quat(values[3:])
    return np.concatenate([transform_points(matrix, values[:3]), orientation.as_quat()])


def invert_transform(transform: np.ndarray) -> np.ndarray:
    """Return T_source_target from T_target_source."""
    matrix = _rigid_matrix(transform)
    result = np.eye(4)
    result[:3, :3] = matrix[:3, :3].T
    result[:3, 3] = -result[:3, :3] @ matrix[:3, 3]
    return result


def compose_transforms(
    target_from_intermediate: np.ndarray, intermediate_from_source: np.ndarray
) -> np.ndarray:
    """Return T_target_source = T_target_intermediate @ T_intermediate_source."""
    return _rigid_matrix(target_from_intermediate) @ _rigid_matrix(
        intermediate_from_source
    )
