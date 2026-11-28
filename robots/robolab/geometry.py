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

"""Contact geometry and the native DROID absolute-IK action convention."""

from __future__ import annotations

import numpy as np
from scipy.spatial.transform import Rotation

from rpent.utils.transforms import quaternion_wxyz


def root_control_action(
    world_xyz: np.ndarray,
    world_rotation: np.ndarray,
    opening: float,
    root_xyz: np.ndarray,
    root_rotation: np.ndarray,
) -> np.ndarray:
    """Convert a world flange target to [root xyz, root WXYZ, native close]."""
    root_from_world = root_rotation.T
    return np.r_[
        root_from_world @ (world_xyz - root_xyz),
        quaternion_wxyz(Rotation.from_matrix(root_from_world @ world_rotation)),
        float(opening < 0.5),
    ]


def inner_pad_center(
    flange_vertices: np.ndarray,
    polygons: list[np.ndarray] | np.ndarray,
    other_jaw_xyz: np.ndarray,
) -> np.ndarray:
    """Locate the centre of the contact face that points at the opposite jaw."""
    inward = other_jaw_xyz - np.mean(flange_vertices, axis=0)
    inward /= np.linalg.norm(inward)
    faces = []
    for indices in polygons:
        points = flange_vertices[indices]
        normal = np.cross(points[1] - points[0], points[2] - points[0])
        length = np.linalg.norm(normal)
        if length > 1e-12 and abs(float(normal @ inward)) > 0.95 * length:
            faces.append(points)
    if not faces:
        raise ValueError("No opposed Robotiq contact pad faces were found")
    # A pad has two nearly parallel outer faces. Select the inward layer
    # by position, independent of the USD polygon winding on mirrored jaws.
    inner = max(float(points.mean(axis=0) @ inward) for points in faces)
    points = np.concatenate(
        [points for points in faces if inner - points.mean(axis=0) @ inward < 0.002]
    )
    return (points.min(axis=0) + points.max(axis=0)) / 2
