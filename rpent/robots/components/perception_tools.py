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

"""Backend-independent recorded observations and calibrated depth tools."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

from rpent.session import EnvState


def view_recorded_state(
    step: int,
    *,
    state: EnvState,
    image_artifacts: Mapping[str, Sequence[str]],
) -> dict[str, Any]:
    """Read a state record and the first registered image for each output slot.

    A missing file is omitted, without falling back to a lower-priority image.
    Backend wrappers retain their public defaults and read-only decoration.
    """
    try:
        record = state.get(step)
    except Exception as error:
        return {"error": f"state step not available: {error}"}
    result = {
        "step": record.step_idx,
        "terminated": record.terminated,
        "truncated": record.truncated,
        "state": record.state,
        "artifacts": sorted(record.artifacts),
        "task_language": record.extras.get("task_language"),
        "log": {
            "command": record.command,
            "result": record.result,
            "elapsed_s": record.elapsed_s,
        },
    }
    for slot, candidates in image_artifacts.items():
        name = next((name for name in candidates if name in record.artifacts), None)
        if name:
            try:
                result[slot] = state.load_bytes(name, step=record.step_idx)
            except FileNotFoundError:
                pass
    return result


def back_project_depth(
    camera_obs: Mapping[str, Any],
    row: int,
    col: int,
    *,
    camera: str,
    camera_z_sign: int,
) -> dict[str, Any]:
    """Project optical-axis depth through intrinsics and a camera-to-world pose.

    The caller supplies the camera's Z convention (+1 or -1); image X/Y
    conventions and calibration are supplied unchanged by the backend.
    """
    if camera_z_sign not in (-1, 1):
        raise ValueError("camera_z_sign must be -1 or 1")
    depth = camera_obs.get("distance_to_image_plane")
    if depth is None:
        depth = camera_obs.get("depth")
    if depth is None:
        return {"error": "depth not available in observation"}
    K = np.asarray(camera_obs.get("intrinsic_matrix"), dtype=np.float64)
    T = np.asarray(camera_obs.get("extrinsic_matrix"), dtype=np.float64)
    if K.shape != (3, 3) or T.shape != (4, 4):
        return {"error": f"calibration missing/invalid: K={K.shape} T={T.shape}"}
    h, w = int(depth.shape[0]), int(depth.shape[1])
    if not (0 <= int(row) < h and 0 <= int(col) < w):
        return {"error": f"pixel out of range: row 0..{h - 1}, col 0..{w - 1}"}
    d = float(depth[int(row), int(col)])
    if not np.isfinite(d) or d <= 0:
        return {"error": f"invalid depth at pixel: {d}"}
    fx, fy = float(K[0, 0]), float(K[1, 1])
    cx, cy = float(K[0, 2]), float(K[1, 2])
    p_cam = np.array(
        [(int(col) - cx) / fx * d, (int(row) - cy) / fy * d, camera_z_sign * d, 1.0],
        dtype=np.float64,
    )
    p_world = T @ p_cam
    return {
        "camera": camera,
        "pixel": [int(row), int(col)],
        "depth_m": round(d, 4),
        "world_xyz": [round(float(v), 4) for v in p_world[:3]],
    }
