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
"""Validation and geometry shared by stationary calibration tools."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.spatial.transform import Rotation

Array = NDArray[np.float64]
Record = dict[str, Any]
METHODS = ("TSAI", "PARK", "HORAUD", "ANDREFF", "DANIILIDIS")


def check_opencv() -> None:
    """Require the OpenCV APIs used by collection and hand-eye fitting."""
    import cv2

    required = (
        "aruco.CharucoBoard",
        "aruco.CharucoParameters",
        "aruco.CharucoDetector",
        "aruco.getPredefinedDictionary",
        "solvePnP",
        "Rodrigues",
        "projectPoints",
        "calibrateHandEye",
    ) + tuple("CALIB_HAND_EYE_" + method for method in METHODS)
    missing = []
    for name in required:
        value = cv2
        for part in name.split("."):
            value = getattr(value, part, None)
            if value is None:
                missing.append(name)
                break
    if missing:
        raise RuntimeError(
            f"OpenCV {cv2.__version__} lacks required APIs: {', '.join(missing)}. "
            "Install calibration_tools/requirements.txt in a separate environment."
        )


def rigid_transform(value: ArrayLike, name: str = "transform") -> Array:
    """Validate and return a finite SE(3) matrix, without projecting rotations."""
    matrix = np.asarray(value, dtype=float)
    if matrix.shape != (4, 4) or not np.isfinite(matrix).all():
        raise ValueError(f"{name} must be a finite 4 x 4 matrix")
    rotation = matrix[:3, :3]
    if not (
        np.allclose(matrix[3], [0, 0, 0, 1], rtol=0, atol=1e-8)
        and np.allclose(rotation.T @ rotation, np.eye(3), rtol=0, atol=1e-5)
        and np.isclose(np.linalg.det(rotation), 1, rtol=0, atol=1e-5)
    ):
        raise ValueError(
            f"{name} must contain a proper rotation and homogeneous last row"
        )
    return matrix


def transform(state: Record) -> Array:
    """Decode libfranka's column-major O_T_EE pose."""
    return rigid_transform(
        np.asarray(state["O_T_EE"]).reshape(4, 4, order="F"), "O_T_EE"
    )


def check_state(state: Record) -> None:
    """Reject moving, non-idle, erroneous, or malformed robot states."""
    if state["robot_mode"] != 1 or state["has_errors"]:
        raise ValueError("Robot must be Idle and free of errors before sampling")
    velocity = np.asarray(state["dq"], dtype=float)
    if velocity.shape != (7,) or not np.isfinite(velocity).all():
        raise ValueError("Robot dq must contain seven finite joint velocities")
    if np.max(np.abs(velocity)) > 0.015:
        raise ValueError("Robot is still moving; release manual guidance and wait")
    transform(state)
    rigid_transform(np.asarray(state["F_T_EE"]).reshape(4, 4, order="F"), "F_T_EE")


def delta(first: Array, second: Array) -> tuple[float, float]:
    """Return relative translation in meters and rotation in radians."""
    return (
        float(np.linalg.norm(first[:3, 3] - second[:3, 3])),
        float(Rotation.from_matrix(first[:3, :3].T @ second[:3, :3]).magnitude()),
    )


def mean_pose(poses: Array) -> Array:
    """Average translations and rotations of a nonempty pose array."""
    if len(poses) == 0:
        raise ValueError("Cannot average an empty pose collection")
    result = np.eye(4)
    result[:3, :3] = Rotation.from_matrix(poses[:, :3, :3]).mean().as_matrix()
    result[:3, 3] = poses[:, :3, 3].mean(axis=0)
    return result


def errors(poses: Array, reference: Array) -> tuple[Array, Array]:
    """Return translation errors in millimeters and rotation errors in degrees."""
    return (
        np.linalg.norm(poses[:, :3, 3] - reference[:3, 3], axis=1) * 1000,
        np.rad2deg(
            Rotation.from_matrix(reference[:3, :3].T @ poses[:, :3, :3]).magnitude()
        ),
    )


def stats(values: ArrayLike) -> Record:
    """Summarize a nonempty finite vector without emitting NaN JSON values."""
    array = np.asarray(values, dtype=float)
    if array.ndim != 1 or array.size == 0 or not np.isfinite(array).all():
        raise ValueError("Statistics require a nonempty finite vector")
    return {
        "rms": float(np.sqrt(np.mean(array**2))),
        "max": float(array.max()),
        "values": array.tolist(),
    }


def write_json(path: Path, value: Record) -> None:
    """Write readable UTF-8 JSON, rejecting non-finite floating point values."""
    path.write_text(
        json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )


def check_camera(camera: Record, reference: Record) -> None:
    """Require the same camera identity, resolution, and intrinsics."""
    for key in ("serial", "width", "height", "distortion_model"):
        if camera[key] != reference[key]:
            raise ValueError(f"Camera {key} changed relative to the reference sample")
    for key in ("K", "distortion"):
        current = np.asarray(camera[key], dtype=float)
        expected = np.asarray(reference[key], dtype=float)
        if (
            current.shape != expected.shape
            or not np.isfinite(current).all()
            or not np.allclose(current, expected, rtol=0, atol=1e-8)
        ):
            raise ValueError(f"Camera {key} changed or contains invalid values")
    intrinsic = np.asarray(camera["K"], dtype=float)
    if intrinsic.shape != (3, 3) or intrinsic[0, 0] <= 0 or intrinsic[1, 1] <= 0:
        raise ValueError("Camera K must be 3 x 3 with positive focal lengths")


def load_candidate(path: Path) -> Record:
    """Load a candidate only when its report and original samples still agree."""
    candidate = json.loads(path.read_text(encoding="utf-8"))
    report = json.loads(
        (path.parent / "quality_report.json").read_text(encoding="utf-8")
    )
    if any(
        data.get("status") != "candidate_requires_independent_validation"
        for data in (candidate, report)
    ):
        raise ValueError("Training report is not a successful candidate fit")
    for key in ("arm", "calibration_mode", "sample_count", "selected_method"):
        if candidate.get(key) != report.get(key):
            raise ValueError(f"Candidate and quality report disagree on {key}")
    if candidate.get("camera_serial") != report.get("serial"):
        raise ValueError("Candidate camera serial differs from the quality report")
    files = sorted(path.parent.glob("sample_*/sample.json"))
    actual_hashes = {
        str(p.relative_to(path.parent)): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in files
    }
    if (
        len(files) < 10
        or len(files) != candidate["sample_count"]
        or actual_hashes != report["source_hashes"]
    ):
        raise ValueError("Training samples changed since the candidate was fitted")
    if candidate["calibration_mode"] != "eye_to_hand" or candidate["arm"] not in (
        "left",
        "right",
    ):
        raise ValueError("Candidate must describe fixed-camera calibration")
    selected = report["methods"][candidate["selected_method"]]
    for key in (f"T_{candidate['arm']}_base_camera", "T_ee_board"):
        if not np.allclose(
            rigid_transform(candidate[key]),
            rigid_transform(selected[key]),
            rtol=0,
            atol=1e-8,
        ):
            raise ValueError(f"Candidate {key} differs from the selected fit")
    return candidate
