# Copyright 2026 The RPent Authors.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy at https://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""CV depth projection, including active RealSense distortion coefficients.

The no-distortion path needs only NumPy. Distorted cameras use the actual SDK
deprojection implementation and cache unit-depth rays per active profile.
No camera is opened here. See librealsense/src/rs.cpp rs2_deproject_pixel_to_point.
"""

from functools import lru_cache

import numpy as np


@lru_cache(maxsize=6)
def _rays(height: int, width: int, k: tuple, model: str, coeffs: tuple) -> np.ndarray:
    intrinsic = np.array(k).reshape(3, 3)
    if not np.isfinite(intrinsic).all() or intrinsic[0, 0] <= 0 or intrinsic[1, 1] <= 0:
        raise ValueError(
            "active camera intrinsic_K must be finite with positive focal lengths"
        )
    model = model.rsplit(".", 1)[-1].lower()
    if model in {"none", "0"} and not any(coeffs):
        rows, cols = np.mgrid[:height, :width]
        rays = np.stack(
            (
                (cols - intrinsic[0, 2]) / intrinsic[0, 0],
                (rows - intrinsic[1, 2]) / intrinsic[1, 1],
                np.ones((height, width)),
            ),
            axis=-1,
        )
    else:
        try:
            import pyrealsense2 as rs
        except ImportError as exc:
            raise RuntimeError(
                "Install pyrealsense2 on the Agent machine for distorted-camera world projection"
            ) from exc
        if model == "modified_brown_conrady":
            raise ValueError(
                "RealSense cannot deproject modified_brown_conrady color directly; rectify the stream first"
            )
        known_models = {
            "none",
            "inverse_brown_conrady",
            "brown_conrady",
            "ftheta",
            "kannala_brandt4",
        }
        if model not in known_models:
            raise ValueError(f"unsupported RealSense distortion model {model!r}")
        intr = rs.intrinsics()
        intr.width, intr.height = width, height
        intr.fx, intr.fy = float(intrinsic[0, 0]), float(intrinsic[1, 1])
        intr.ppx, intr.ppy = float(intrinsic[0, 2]), float(intrinsic[1, 2])
        intr.model, intr.coeffs = getattr(rs.distortion, model), list(coeffs)
        rays = np.asarray(
            [
                rs.rs2_deproject_pixel_to_point(intr, [col, row], 1.0)
                for row in range(height)
                for col in range(width)
            ]
        ).reshape(height, width, 3)
    rays = rays.astype(np.float32)
    rays.setflags(write=False)
    return rays


def world_from_depth(depth_metric, camera_meta: dict) -> np.ndarray:
    depth = np.asarray(depth_metric, dtype=np.float32)
    height, width = int(camera_meta["height"]), int(camera_meta["width"])
    if depth.shape != (height, width):
        raise ValueError(
            f"depth shape {depth.shape} differs from active profile {(height, width)}"
        )
    transform = np.asarray(camera_meta["cam2world_cv"], dtype=np.float64)
    if transform.shape != (4, 4) or not np.isfinite(transform).all():
        raise ValueError("cam2world_cv must be a finite 4x4 transform")
    if camera_meta.get("projection_valid") is False:
        raise ValueError(
            camera_meta.get("projection_limitation", "frame and FK are not aligned")
        )
    intrinsic = np.asarray(camera_meta["intrinsic_K"], dtype=np.float64)
    if intrinsic.shape != (3, 3):
        raise ValueError("intrinsic_K must be 3x3")
    coeffs = np.asarray(
        camera_meta.get("distortion_coeffs", [0.0] * 5), dtype=np.float64
    )
    if coeffs.shape != (5,) or not np.isfinite(coeffs).all():
        raise ValueError("distortion_coeffs must contain five finite values")
    model = camera_meta.get(
        "distortion_model", "none" if not np.any(coeffs) else "unknown"
    )
    rays = _rays(height, width, tuple(intrinsic.ravel()), str(model), tuple(coeffs))
    world = (rays * depth[..., None]) @ transform[:3, :3].T + transform[:3, 3]
    world[~np.isfinite(depth) | (depth <= 0)] = np.nan
    return world.astype(np.float32)
