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

"""R2 site configuration: hardware, motion bounds, and camera calibration."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

ARMS = ("left", "right")
DEFAULT_CONFIG_PATH = Path(__file__).parent / "config" / "example.yaml"


@dataclass(frozen=True)
class Workspace:
    """Axis-aligned TCP box in ``base_link``, metres."""

    low: tuple[float, float, float]
    high: tuple[float, float, float]

    def contains(self, xyz: Any) -> bool:
        """Return whether ``xyz`` lies inside the box, boundary included."""
        return all(lo <= float(v) <= hi for lo, v, hi in zip(self.low, xyz, self.high))


@dataclass(frozen=True)
class MotionLimits:
    """Bounds every planner-commanded motion must satisfy."""

    max_joint_speed: float
    max_move_m: float
    max_delta_m: float
    max_rotation_rad: float
    workspace: dict[str, Workspace]


@dataclass(frozen=True)
class BaseLimits:
    """Whether the planner may drive the chassis, and how fast."""

    allow_motion: bool
    max_linear_speed: float
    max_angular_speed: float
    max_duration: float


@dataclass(frozen=True)
class CameraCalibration:
    """Where one camera sits in the URDF and how it projects.

    ``distortion`` holds OpenCV's ``k1, k2, p1, p2, k3`` for the same image.
    """

    frame: str
    intrinsics: dict[str, float] | None
    distortion: tuple[float, ...] | None = None


@dataclass(frozen=True)
class R2SiteConfig:
    """One R2 site: the RLinf hardware options plus RPent's planner bounds."""

    robot: dict[str, Any]
    rest: dict[str, Any]
    motion: MotionLimits
    base: BaseLimits
    urdf: str | None
    cameras: dict[str, CameraCalibration] = field(default_factory=dict)


_ROBOT_KEYS = {
    "body_dof",
    "end_effector_type",
    "enable_base",
    "cameras",
    "enable_depth",
    "use_compressed_camera",
    "gripper_min_closure",
}
_REST_KEYS = {"joint_qpos", "joint_speed", "duration"}


def _section(data: Mapping[str, Any], name: str, keys: set[str]) -> dict[str, Any]:
    section = data.get(name) or {}
    if not isinstance(section, Mapping):
        raise ValueError(f"R2 config section {name!r} must be a mapping")
    unknown = set(section) - keys
    if unknown:
        raise ValueError(
            f"unknown keys in R2 config section {name!r}: {sorted(unknown)}; "
            f"expected a subset of {sorted(keys)}"
        )
    return dict(section)


def _positive(section: Mapping[str, Any], key: str, name: str) -> float:
    value = float(section[key])
    if not math.isfinite(value) or value <= 0:
        raise ValueError(f"{name}.{key} must be finite and positive, got {value!r}")
    return value


def _vector3(value: Any, name: str) -> tuple[float, float, float]:
    vector = tuple(float(v) for v in value)
    if len(vector) != 3 or not all(math.isfinite(v) for v in vector):
        raise ValueError(f"{name} must be three finite numbers, got {value!r}")
    return vector


def _intrinsics(value: Any, camera: str) -> dict[str, float] | None:
    if value is None:
        return None
    keys = {"fx", "fy", "cx", "cy"}
    if not isinstance(value, Mapping) or set(value) != keys:
        raise ValueError(
            f"perception.cameras.{camera}.intrinsics must have exactly "
            f"{sorted(keys)}, got {value!r}"
        )
    return {key: float(value[key]) for key in sorted(keys)}


def _distortion(value: Any, camera: str) -> tuple[float, ...] | None:
    if value is None:
        return None
    coefficients = tuple(float(v) for v in value)
    if len(coefficients) != 5 or not all(math.isfinite(v) for v in coefficients):
        raise ValueError(
            f"perception.cameras.{camera}.distortion must be five finite numbers "
            f"(k1, k2, p1, p2, k3), got {value!r}"
        )
    return coefficients


def load_site_config(path: str | Path | None = None) -> R2SiteConfig:
    """Load and validate an R2 site YAML, defaulting to the checked-in example."""
    config_path = Path(path or DEFAULT_CONFIG_PATH).expanduser()
    data = yaml.safe_load(config_path.read_text()) or {}
    unknown = set(data) - {"robot", "rest", "motion", "base", "perception"}
    if unknown:
        raise ValueError(f"unknown R2 config sections: {sorted(unknown)}")

    robot = _section(data, "robot", _ROBOT_KEYS)
    rest = _section(data, "rest", _REST_KEYS)

    motion = _section(
        data,
        "motion",
        {
            "max_joint_speed",
            "max_move_m",
            "max_delta_m",
            "max_rotation_rad",
            "workspace",
        },
    )
    workspace: dict[str, Workspace] = {}
    for arm in ARMS:
        box = (motion.get("workspace") or {}).get(arm)
        if box is None:
            raise ValueError(f"motion.workspace.{arm} is required")
        low = _vector3(box["low"], f"motion.workspace.{arm}.low")
        high = _vector3(box["high"], f"motion.workspace.{arm}.high")
        if not all(lo < hi for lo, hi in zip(low, high)):
            raise ValueError(
                f"motion.workspace.{arm} needs low < high on every axis, "
                f"got low={low} high={high}"
            )
        workspace[arm] = Workspace(low=low, high=high)
    limits = MotionLimits(
        max_joint_speed=_positive(motion, "max_joint_speed", "motion"),
        max_move_m=_positive(motion, "max_move_m", "motion"),
        max_delta_m=_positive(motion, "max_delta_m", "motion"),
        max_rotation_rad=_positive(motion, "max_rotation_rad", "motion"),
        workspace=workspace,
    )

    base = _section(
        data,
        "base",
        {"allow_motion", "max_linear_speed", "max_angular_speed", "max_duration"},
    )
    if not isinstance(base.get("allow_motion", False), bool):
        raise ValueError("base.allow_motion must be true or false")
    base_limits = BaseLimits(
        allow_motion=base.get("allow_motion", False),
        max_linear_speed=_positive(base, "max_linear_speed", "base"),
        max_angular_speed=_positive(base, "max_angular_speed", "base"),
        max_duration=_positive(base, "max_duration", "base"),
    )
    if base_limits.allow_motion and not robot.get("enable_base", False):
        raise ValueError("base.allow_motion requires robot.enable_base: true")

    perception = _section(data, "perception", {"urdf", "cameras"})
    cameras = {
        name: CameraCalibration(
            frame=str(entry["frame"]),
            intrinsics=_intrinsics(entry.get("intrinsics"), name),
            distortion=_distortion(entry.get("distortion"), name),
        )
        for name, entry in (perception.get("cameras") or {}).items()
    }
    missing = set(robot.get("cameras", ())) - set(cameras)
    if perception.get("urdf") and missing:
        raise ValueError(
            f"perception.cameras lacks a URDF frame for robot cameras {sorted(missing)}"
        )
    return R2SiteConfig(
        robot=robot,
        rest=rest,
        motion=limits,
        base=base_limits,
        urdf=perception.get("urdf"),
        cameras=cameras,
    )
