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

"""Robot identity and observation/control settings for the LeKiwi service."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from pathlib import Path

from omegaconf import DictConfig, OmegaConf

from robots.franka.runtime_config import load_mapping, strict_mapping

DEFAULT_CONFIG = Path(__file__).with_name("config") / "example.yaml"


@dataclass
class BaseConfig:
    """Serial connection and wheel geometry consumed by the local base."""

    serial_port: str = ""
    wheel_radius: float = 0.05
    base_radius: float = 0.125
    max_raw: int = 3000


@dataclass
class DriveConfig:
    """Service control rate, velocity limits and camera observation settings."""

    step_frequency: float = 10.0
    camera_max_age: float = 0.05
    max_linear_velocity: float = 0.3
    max_angular_velocity: float = 60.0
    frame_size: tuple[int, int] = (128, 128)


def load_runtime_config(path: str | Path | None) -> DictConfig:
    """Validate robot YAML without constructing a Gym environment or hardware."""
    raw = load_mapping(path or DEFAULT_CONFIG)
    robot = raw.get("robot")
    if not isinstance(robot, dict) or not robot.get("serial_port"):
        raise ValueError("robot.serial_port is required")
    hardware = strict_mapping(
        BaseConfig,
        {**asdict(BaseConfig()), **robot},
        where="robot",
    )
    for name in ("wheel_radius", "base_radius"):
        value = float(hardware[name])
        if not math.isfinite(value) or value <= 0:
            raise ValueError(f"robot.{name} must be finite and positive")
    if not 1 <= hardware["max_raw"] <= 32767:
        raise ValueError("robot.max_raw must be between 1 and 32767")
    cameras = raw.get("cameras")
    if not isinstance(cameras, dict) or not isinstance(cameras.get("devices"), dict):
        raise ValueError("cameras.devices must be a mapping")
    devices = cameras["devices"]
    main = []
    serials = set()
    for name, device in devices.items():
        if not isinstance(device, dict) or not device.get("serial"):
            raise ValueError(f"cameras.devices.{name}.serial is required")
        if str(device["serial"]) in serials:
            raise ValueError("camera serials must be unique")
        serials.add(str(device["serial"]))
        if device.get("main", False):
            main.append(name)
    if len(main) != 1:
        raise ValueError("exactly one camera device must set main: true")
    drive = strict_mapping(
        DriveConfig, {**asdict(DriveConfig()), **raw.get("drive", {})}, where="drive"
    )
    for name in ("step_frequency", "max_linear_velocity", "max_angular_velocity"):
        value = float(drive[name])
        if not math.isfinite(value) or value <= 0:
            raise ValueError(f"drive.{name} must be finite and positive")
    if not 5 <= drive["step_frequency"] <= 100:
        raise ValueError("drive.step_frequency must be between 5 and 100 Hz")
    age = float(drive["camera_max_age"])
    if not math.isfinite(age) or age < 0:
        raise ValueError("drive.camera_max_age must be finite and nonnegative")
    size = drive["frame_size"]
    if len(size) != 2 or any(type(v) is not int or v <= 0 for v in size):
        raise ValueError("drive.frame_size must contain two positive integers")
    drive["frame_size"] = list(size)
    return OmegaConf.create(
        {"robot": hardware, "cameras": devices, "main_camera": main[0], "drive": drive}
    )
