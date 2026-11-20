# Copyright 2025 The RLinf Authors.
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

"""LeKiwi ZED calibration, rectification and camera geometry."""

import configparser
import math
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from robots.lekiwi.runtime_config import (
    _number,
    get_robot_config_path,
    load_mapping,
)


def resized_k(
    k: np.ndarray, source_size: tuple[int, int], target_size: tuple[int, int]
) -> np.ndarray:
    """OpenCV pixel-center resize: u' = s*(u+0.5)-0.5; no crop."""
    result = np.asarray(k, dtype=np.float64).reshape(3, 3).copy()
    sx, sy = np.asarray(target_size) / np.asarray(source_size)
    result[0, 0] *= sx
    result[1, 1] *= sy
    result[0, 2] = (result[0, 2] + 0.5) * sx - 0.5
    result[1, 2] = (result[1, 2] + 0.5) * sy - 0.5
    return result


class StereoCalibration:
    """Read native-resolution ZED calibration and use OpenCV stereo rectification."""

    def __init__(
        self,
        path: str | Path | None = None,
        *,
        size: tuple[int, int] | None = None,
    ) -> None:
        if path is None:
            from robots.lekiwi.runtime_config import load_device_config

            devices = load_device_config()
            path = devices["calibration"]
            size = size or devices["eye_size"]
        raw = Path(path).read_bytes()
        cfg = configparser.ConfigParser()
        cfg.read_string(raw.decode())
        size = size or (1280, 720)
        modes = {(672, 376): "VGA", (1280, 720): "HD"}
        if size not in modes:
            raise ValueError("Calibration requires native VGA or HD720 eye dimensions")
        self.size = size
        mode = modes[size]
        cameras = []
        for side in ("LEFT", "RIGHT"):
            s = cfg[f"{side}_CAM_{mode}"]
            k = np.array(
                [
                    [float(s["fx"]), 0, float(s["cx"])],
                    [0, float(s["fy"]), float(s["cy"])],
                    [0, 0, 1.0],
                ]
            )
            d = np.array([float(s[x]) for x in ("k1", "k2", "p1", "p2", "k3")])
            cameras.append((k, d))
        s = cfg["STEREO"]
        rotation, _ = cv2.Rodrigues(
            np.array([float(s[f"{k}_{mode}"]) for k in ("RX", "CV", "RZ")])
        )
        translation = (
            np.array([-float(s["Baseline"]), float(s["TY"]), float(s["TZ"])]) / 1000
        )
        k1, d1 = cameras[0]
        k2, d2 = cameras[1]
        r1, r2, p1, p2, _, _, _ = cv2.stereoRectify(
            k1,
            d1,
            k2,
            d2,
            self.size,
            rotation,
            translation.reshape(3, 1),
            flags=cv2.CALIB_ZERO_DISPARITY,
            alpha=0,
        )
        self.rectifications = [(r1, p1), (r2, p2)]
        self.maps = [
            cv2.initUndistortRectifyMap(k, d, r, p, self.size, cv2.CV_32FC1)
            for (k, d), (r, p) in zip(cameras, self.rectifications)
        ]

    def localization_camera(self, size: tuple[int, int] = (672, 376)) -> dict[str, Any]:
        intrinsics = resized_k(self.rectifications[0][1][:3, :3], self.size, size)
        return {
            "model": "PINHOLE",
            "width": size[0],
            "height": size[1],
            "params": [
                float(intrinsics[0, 0]),
                float(intrinsics[1, 1]),
                float(intrinsics[0, 2]),
                float(intrinsics[1, 2]),
            ],
            "calibrated": True,
            "pixel_convention": "opencv_center",
        }


def load_geometry(path=None, *, raw=None) -> dict:
    raw = raw if raw is not None else load_mapping(path or get_robot_config_path())
    geometry = raw["geometry"]
    body, camera = geometry["body"], geometry["camera_mount"]
    body_height = _number(body["height_m"], "body.height_m", positive=True)
    xyz = [
        _number(camera["forward_m"], "camera_mount.forward_m"),
        _number(camera["left_m"], "camera_mount.left_m"),
        _number(
            camera["height_above_ground_m"],
            "camera_mount.height_above_ground_m",
            positive=True,
        ),
    ]
    rpy = camera["mount_rpy_deg"]
    if not isinstance(rpy, list) or len(rpy) != 3:
        raise ValueError("camera_mount.mount_rpy_deg requires roll, pitch, yaw")
    rpy = [_number(value, "camera_mount.mount_rpy_deg") for value in rpy]
    return {
        "mount_xyz_m": xyz,
        "mount_rpy_deg": rpy,
        "collision_height_m": body_height,
        "robot_geometry": geometry,
    }


def camera_mount_transform(calibration, geometry=None) -> np.ndarray:
    """Map the installed rectified optical camera into the chassis ground frame."""
    installed = load_geometry() if geometry is None else geometry
    mount = np.eye(4)
    mount[:3, :3] = (
        optical_to_base_rotation(tuple(installed["mount_rpy_deg"]))
        @ calibration.rectifications[0][0].T
    )
    mount[:3, 3] = installed["mount_xyz_m"]
    return mount


def rigid(value: list) -> np.ndarray:
    """Reject malformed poses before they enter the TF tree."""
    pose = np.asarray(value, dtype=float)
    if (
        pose.shape != (4, 4)
        or not np.isfinite(pose).all()
        or not np.allclose(pose[3], [0, 0, 0, 1], atol=1e-5)
        or not np.allclose(pose[:3, :3].T @ pose[:3, :3], np.eye(3), atol=1e-4)
        or not np.isclose(np.linalg.det(pose[:3, :3]), 1, atol=1e-4)
    ):
        raise ValueError("Expected a finite rigid transform")
    return pose


def optical_to_base_rotation(rpy_degrees: tuple[float, float, float]) -> np.ndarray:
    """Compose mechanical mount Rz*Ry*Rx with ROS optical-to-body axis conversion."""
    if len(rpy_degrees) != 3 or not all(math.isfinite(v) for v in rpy_degrees):
        raise ValueError("Mount RPY must contain three finite angles")
    roll, pitch, yaw = map(math.radians, rpy_degrees)
    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)
    rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]])
    ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
    rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]])
    optical_to_body = np.array([[0, 0, 1], [-1, 0, 0], [0, -1, 0]])
    return (rz @ ry @ rx @ optical_to_body).astype(np.float32)


def optical_to_base_quaternion(rpy_degrees: tuple[float, float, float]) -> np.ndarray:
    """Express the shared optical-to-base rotation as a ROS xyzw quaternion."""
    from scipy.spatial.transform import Rotation

    return Rotation.from_matrix(optical_to_base_rotation(rpy_degrees)).as_quat()
