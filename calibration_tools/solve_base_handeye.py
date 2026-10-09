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
"""Offline fixed-camera calibration with leave-one-out reports."""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
from pathlib import Path

import cv2
import numpy as np
from common import (
    METHODS,
    Array,
    Record,
    check_camera,
    check_opencv,
    check_state,
    delta,
    errors,
    mean_pose,
    rigid_transform,
    stats,
    transform,
    write_json,
)
from scipy.spatial.transform import Rotation

LOGGER = logging.getLogger(__name__)


def fit(
    robot_poses: Array, board_poses: Array, method: str
) -> tuple[Array, Array, tuple[Array, Array]]:
    """Fit X where inverse(A) X C is the fixed end-effector-to-board pose.

    Args:
        robot_poses: Base-to-end-effector transforms, shape (N, 4, 4).
        board_poses: Camera-to-board transforms, shape (N, 4, 4).
        method: An OpenCV hand-eye method name from METHODS.

    Returns:
        Camera extrinsic, mean fixed-board transform, and residual vectors.
    """
    if method not in METHODS:
        raise ValueError("Unknown hand-eye method")
    if len(robot_poses) != len(board_poses) or len(robot_poses) < 3:
        raise ValueError("Hand-eye fitting requires at least three paired poses")
    robot_poses = np.array([rigid_transform(p) for p in robot_poses])
    board_poses = np.array([rigid_transform(p) for p in board_poses])
    rotations = Rotation.from_matrix(
        robot_poses[0, :3, :3].T @ robot_poses[:, :3, :3]
    ).as_rotvec()
    if np.linalg.matrix_rank(rotations, tol=1e-6) < 2:
        raise ValueError(
            "Insufficient rotation excitation: rotate around at least two axes"
        )
    solver_poses = np.linalg.inv(robot_poses)
    rotation, translation = cv2.calibrateHandEye(
        solver_poses[:, :3, :3],
        solver_poses[:, :3, 3],
        board_poses[:, :3, :3],
        board_poses[:, :3, 3],
        method=getattr(cv2, "CALIB_HAND_EYE_" + method),
    )
    extrinsic = np.eye(4)
    extrinsic[:3, :3], extrinsic[:3, 3] = rotation, translation.ravel()
    rigid_transform(extrinsic, f"{method} solution")
    fixed_poses = solver_poses @ extrinsic @ board_poses
    reference = mean_pose(fixed_poses)
    return extrinsic, reference, errors(fixed_poses, reference)


def load_samples(
    root: Path, arm: str, serial: str | None = None
) -> tuple[list[Path], list[Record], Array, Array]:
    """Load samples and reject inconsistent identities, geometry, or motion."""
    if arm not in ("left", "right"):
        raise ValueError("Arm must be left or right")
    files = sorted(root.glob("sample_*/sample.json"))
    samples = [json.loads(path.read_text(encoding="utf-8")) for path in files]
    if len(samples) < 10:
        raise ValueError(
            "At least ten samples are required for leave-one-out calibration"
        )
    reference = samples[0]
    expected_serial = serial or reference["camera"]["serial"]
    robot_poses, board_poses = [], []
    for path, sample in zip(files, samples):
        if sample.get("calibration_mode") != "eye_to_hand" or sample.get("arm") != arm:
            raise ValueError(
                f"{path}: arm or calibration mode does not match this solver"
            )
        if (
            sample["camera"]["serial"] != expected_serial
            or sample["board"] != reference["board"]
        ):
            raise ValueError(f"{path}: camera serial or board definition changed")
        check_camera(sample["camera"], reference["camera"])
        for key in ("robot_before", "robot_after"):
            check_state(sample[key])
            if not np.allclose(
                sample[key]["F_T_EE"],
                reference["robot_before"]["F_T_EE"],
                rtol=0,
                atol=1e-8,
            ):
                raise ValueError(f"{path}: end-effector frame changed")
        pose = rigid_transform(sample[f"T_{arm}_base_ee"], str(path))
        if not np.allclose(pose, transform(sample["robot_before"]), rtol=0, atol=1e-8):
            raise ValueError(f"{path}: stored pose disagrees with robot_before")
        drift_m, drift_rad = delta(pose, transform(sample["robot_after"]))
        if drift_m > 0.001 or drift_rad > 0.005:
            raise ValueError(f"{path}: robot moved during image acquisition")
        image = cv2.imread(str(path.parent / "color.png"))
        if image is None or image.shape[:2] != (
            sample["camera"]["height"],
            sample["camera"]["width"],
        ):
            raise ValueError(f"{path}: missing image or incorrect image dimensions")
        if (
            sample["corners"] < 15
            or not np.isfinite(sample["reprojection_rms_px"])
            or not 0 <= sample["reprojection_rms_px"] <= 1
        ):
            raise ValueError(
                f"{path}: insufficient corners or excessive reprojection error"
            )
        robot_poses.append(pose)
        board_poses.append(rigid_transform(sample["T_camera_board"], str(path)))
    return files, samples, np.array(robot_poses), np.array(board_poses)


def _solve_session(root: Path, arm: str, serial: str | None = None) -> Record:
    """Write candidate extrinsics and training/leave-one-out consistency reports."""
    files, samples, robot_poses, board_poses = load_samples(root, arm, serial)
    serial = samples[0]["camera"]["serial"]
    extrinsic_key = f"T_{arm}_base_camera"
    board_key = "T_ee_board"
    report = {
        "status": "candidate_requires_independent_validation",
        "arm": arm,
        "serial": serial,
        "calibration_mode": "eye_to_hand",
        "sample_count": len(samples),
        "source_hashes": {
            str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in files
        },
        "corners": [s["corners"] for s in samples],
        "image_reprojection_px": stats([s["reprojection_rms_px"] for s in samples]),
        "max_bracket_drift_mm": max(
            delta(transform(s["robot_before"]), transform(s["robot_after"]))[0]
            for s in samples
        )
        * 1000,
        "position_span_m": np.ptp(robot_poses[:, :3, 3], axis=0).tolist(),
        "rotation_excitation_singular_values": np.linalg.svd(
            Rotation.from_matrix(
                robot_poses[0, :3, :3].T @ robot_poses[:, :3, :3]
            ).as_rotvec(),
            compute_uv=False,
        ).tolist(),
        "methods": {},
    }
    for method in METHODS:
        try:
            extrinsic, board, residual = fit(robot_poses, board_poses, method)
            held_translation, held_rotation, change_translation, change_rotation = (
                [],
                [],
                [],
                [],
            )
            for index in range(len(samples)):
                mask = np.arange(len(samples)) != index
                other, fixed, _ = fit(robot_poses[mask], board_poses[mask], method)
                pose = np.linalg.inv(robot_poses[index])
                translation, rotation = errors(
                    (pose @ other @ board_poses[index])[None], fixed
                )
                held_translation.append(translation[0])
                held_rotation.append(rotation[0])
                translation, rotation = errors(other[None], extrinsic)
                change_translation.append(translation[0])
                change_rotation.append(rotation[0])
            report["methods"][method] = {
                extrinsic_key: extrinsic.tolist(),
                board_key: board.tolist(),
                "training_translation_mm": stats(residual[0]),
                "training_rotation_deg": stats(residual[1]),
                "loo_translation_mm": stats(held_translation),
                "loo_rotation_deg": stats(held_rotation),
                "loo_camera_translation_change_mm": stats(change_translation),
                "loo_camera_rotation_change_deg": stats(change_rotation),
            }
        except (ValueError, cv2.error, np.linalg.LinAlgError) as error:
            report["methods"][method] = {"error": str(error)}
    valid = {
        key: value for key, value in report["methods"].items() if "error" not in value
    }
    if not valid:
        report["status"] = "failed"
        write_json(root / "quality_report.json", report)
        raise ValueError(
            "All hand-eye methods failed; inspect quality_report.json and collect diverse rotations"
        )
    selected = min(valid, key=lambda key: valid[key]["loo_translation_mm"]["rms"])
    report["selected_method"] = selected
    report["selection"] = (
        "Lowest leave-one-out translation RMS; all samples retained; not independent validation"
    )
    best = valid[selected]
    extrinsic = np.array(best[extrinsic_key])
    parameters = {
        "eye_on_hand": False,
        "robot_base_frame": f"{arm}_base",
        "robot_effector_frame": f"{arm}_ee_O_T_EE",
        "tracking_base_frame": f"base_{serial}_color_optical_frame",
        "tracking_marker_frame": "charuco_board",
    }
    transformation = dict(
        zip(
            ("x", "y", "z", "qx", "qy", "qz", "qw"),
            map(
                float,
                np.r_[
                    extrinsic[:3, 3], Rotation.from_matrix(extrinsic[:3, :3]).as_quat()
                ],
            ),
        )
    )
    candidate = {
        "status": report["status"],
        "camera_serial": serial,
        "arm": arm,
        "calibration_mode": "eye_to_hand",
        "sample_count": len(samples),
        "selected_method": selected,
        "method": selected,
        extrinsic_key: extrinsic.tolist(),
        "F_T_EE": samples[0]["robot_before"]["F_T_EE"],
        "camera_intrinsics": {
            k: samples[0]["camera"][k]
            for k in ("K", "distortion", "distortion_model", "width", "height")
        },
        "convention": "T_A_B maps B to A. EE is libfranka O_T_EE; camera is color optical frame.",
        "parameters": parameters,
        "transformation": transformation,
        "source_report": str(root / "quality_report.json"),
    }
    candidate[f"T_camera_{arm}_base"] = np.linalg.inv(extrinsic).tolist()
    candidate["T_ee_board"] = best[board_key]
    write_json(root / "quality_report.json", report)
    write_json(root / "base_camera_extrinsic_candidate.json", candidate)
    LOGGER.info(
        "Selected %s using %d samples; independent validation still required",
        selected,
        len(samples),
    )
    return report


def run(root: Path, arm: str, serial: str | None = None) -> Record:
    """Fit a session and invalidate previous reports if any stage fails."""
    if not root.is_dir():
        raise ValueError(f"Session directory does not exist: {root}")
    report_path = root / "quality_report.json"
    write_json(report_path, {"status": "running"})
    try:
        return _solve_session(root, arm, serial)
    except (
        ValueError,
        KeyError,
        TypeError,
        OSError,
        cv2.error,
        np.linalg.LinAlgError,
    ) as error:
        report = json.loads(report_path.read_text(encoding="utf-8"))
        report.update(status="failed", error=str(error))
        write_json(report_path, report)
        raise


def main() -> None:
    """Parse solver arguments without connecting to robot or camera hardware."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("session", type=Path)
    parser.add_argument("--arm", choices=("left", "right"), required=True)
    parser.add_argument(
        "--camera-serial", help="Optional expected serial; defaults to the first sample"
    )
    args = parser.parse_args()
    check_opencv()
    logging.basicConfig(level=logging.INFO)
    run(args.session, args.arm, args.camera_serial)


if __name__ == "__main__":
    main()
