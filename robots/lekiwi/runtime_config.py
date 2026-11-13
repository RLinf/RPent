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

"""LeKiwi robot, camera geometry and navigation configuration.

Build RLinf environment settings with ``load_runtime_config`` and Nav2 settings
with ``navigation.config.parameters``. Geometry and calibration validation are shared by mapping,
navigation; machine-specific values stay in user configs.
"""

from __future__ import annotations

import argparse
import dataclasses
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from omegaconf import DictConfig, OmegaConf

DEFAULT_CONFIG = Path(__file__).with_name("config") / "example.yaml"
LEKIWI_ROOT = Path(__file__).resolve().parent

_robot_config_path: Path | None = None


def set_robot_config_path(path: str | Path | None) -> None:
    """Record the ``--robot-config`` CLI value (``None`` when absent)."""
    global _robot_config_path
    _robot_config_path = Path(path).expanduser() if path else None


def get_robot_config_path(default: str | Path = DEFAULT_CONFIG) -> Path:
    """Return the ``--robot-config`` override, or the robot's packaged default."""
    return Path(_robot_config_path or default)


def _camera_config(raw: dict) -> tuple[str, dict]:
    devices = raw["cameras"]["devices"]
    if len(devices) != 1:
        raise ValueError("ZED navigation requires exactly one camera")
    name, camera = next(iter(devices.items()))
    if not camera.get("main", False):
        raise ValueError("The ZED camera must set main: true")
    return name, camera


def load_device_config(path: str | Path | None = None) -> dict[str, Any]:
    """Read device identity and perception paths from the selected robot YAML."""
    path = path or get_robot_config_path()
    raw = load_mapping(path)
    _, camera = _camera_config(raw)
    calibration = Path(
        os.path.expandvars(raw["perception"]["calibration"])
    ).expanduser()
    if not calibration.is_absolute():
        calibration = Path(path).expanduser().resolve().parent / calibration
    device = str(camera["serial"])
    if device.startswith("usb-") and "/" not in device:
        device = "/dev/v4l/by-id/" + device
    elif device.startswith("video") and device[5:].isdigit():
        device = "/dev/" + device
    return {
        "device": device,
        "serial": str(camera["zed_serial"]),
        "calibration": os.path.abspath(calibration),
        "eye_size": tuple(camera.get("eye_size", [1280, 720])),
    }


def localization_service_config(path: str | Path | None = None) -> dict[str, Any]:
    """Read the localization endpoint shared by image and map clients."""
    config = dict(load_mapping(path or get_robot_config_path())["localization"])
    endpoint = config.get("endpoint")
    if not isinstance(endpoint, str) or not endpoint.strip():
        raise ValueError("localization.endpoint must be a nonempty string")
    return config


def runtime_settings_from_args(args: argparse.Namespace) -> dict[str, Any]:
    """Build navigation and recording settings from the parsed Env CLI options."""
    map_yaml = args.map_yaml or Path(args.robot_config).parent / "map.yaml"
    return {
        "mode": args.mode,
        "ros_domain_id": args.ros_domain_id,
        "robot_config": args.robot_config,
        "map_yaml": os.path.abspath(map_yaml.expanduser()),
        # Keep venv interpreter symlinks rather than resolving to the base Python.
        "ros_python": os.path.abspath(args.ros_python.expanduser())
        if args.ros_python is not None
        else None,
        "recording": {
            "root": str(args.recording_dir.expanduser().resolve()),
            "sample_fps": args.recording_fps,
            "maximum_recording_seconds": args.capture_seconds,
        },
    }


# ---------------------------------------------------------------------------
# Developer defaults
# ---------------------------------------------------------------------------

# Primitive-control knobs consumed by the RPent LeKiwi env server. RLinf has
# no equivalent field, so these live here rather than in any RLinf dataclass.
CONTROL = {
    "drive_max_duration_s": 10.0,
    # Longest the server keeps a held velocity after the last refresh: the
    # dead-man that stops the base when the operator's session goes away.
    "hold_max_ttl_s": 2.0,
}

# ``env.eval.override_cfg`` values RPent sets away from RLinf's
# ``LeKiwiEnvConfig`` defaults. Keys are RLinf field names; anything omitted
# here keeps RLinf's default.
ENV_DEFAULTS = {
    # RPent streams frames over RPC; it does not run the in-process player.
    "enable_camera_player": False,
    # Episodes are bounded by the planner, not a step budget.
    "max_num_steps": 2_000_000_000,
}


@dataclass(frozen=True)
class LeKiwiRuntimeConfig:
    """Generated RLinf adapter config and RPent primitive settings."""

    rlinf: DictConfig
    controller: dict[str, Any]


def load_runtime_config(path: str | Path | None) -> LeKiwiRuntimeConfig:
    """Load the user YAML, apply developer defaults, and build the adapter."""
    from rlinf.envs.real.lekiwi.base import LeKiwiEnvConfig

    raw = load_mapping(path or get_robot_config_path())
    robot = raw["robot"]
    camera_name, camera = _camera_config(raw)
    serial = str(camera["serial"])
    drive = raw.get("drive") or {}
    unknown = drive.keys() - {
        field.name for field in dataclasses.fields(LeKiwiEnvConfig)
    }
    if unknown:
        raise ValueError(f"drive: unknown key(s) {sorted(unknown)}")

    max_raw = robot.get("max_raw", 3000)
    if type(max_raw) is not int or max_raw <= 0:
        raise ValueError("robot.max_raw must be a positive integer")
    hardware = {
        "port": str(robot["port"]),
        "wheel_radius": _positive(robot.get("wheel_radius", 0.05), "wheel_radius"),
        "base_radius": _positive(robot.get("base_radius", 0.125), "base_radius"),
        "max_raw": max_raw,
        "camera_serials": [serial],
        "camera_type": camera.get("type", "webcam"),
        "node_rank": 0,
    }
    override_cfg = {
        **ENV_DEFAULTS,
        "camera_names": {serial: camera_name},
        **drive,
    }

    rlinf = OmegaConf.create(
        {
            "cluster": {
                "num_nodes": 1,
                "component_placement": {
                    "env": {"node_group": "lekiwi", "placement": 0}
                },
                "node_groups": [
                    {
                        "label": "lekiwi",
                        "node_ranks": 0,
                        "hardware": {"type": "LeKiwi", "configs": [hardware]},
                    }
                ],
            },
            "env": {
                "eval": {
                    "seed": 0,
                    "group_size": 1,
                    "auto_reset": False,
                    "ignore_terminations": False,
                    "use_fixed_reset_state_ids": False,
                    "max_episode_steps": None,
                    "use_spacemouse": False,
                    "no_gripper": False,
                    "main_image_key": camera_name,
                    "keyboard_reward_wrapper": None,
                    "video_cfg": {},
                    "init_params": {"id": "RPentLeKiwiDriveEnv-v1"},
                    "override_cfg": override_cfg,
                }
            },
        }
    )
    return LeKiwiRuntimeConfig(
        rlinf=rlinf,
        controller=CONTROL.copy(),
    )


def _number(value: object, name: str, *, positive: bool = False) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
        or (positive and value <= 0)
    ):
        raise ValueError(f"{name} must be finite{' and positive' if positive else ''}")
    return float(value)


def load_mapping(path: str | Path) -> dict[str, Any]:
    """Load a YAML configuration as a resolved plain dictionary."""
    return OmegaConf.to_container(OmegaConf.load(Path(path).expanduser()), resolve=True)


def _positive(value: float, name: str) -> float:
    """Validate a finite positive control setting."""
    return _number(float(value), name, positive=True)


def control_settings(args: argparse.Namespace, drive: DictConfig) -> dict[str, float]:
    """Apply CLI limits to the loaded RLinf drive settings."""
    linear = _positive(drive.get("max_linear_velocity", 0.1), "max_linear_velocity")
    angular = _positive(drive.get("max_angular_velocity", 30), "max_angular_velocity")
    if args.max_linear is not None:
        linear = min(linear, _positive(args.max_linear, "max_linear"))
    if args.max_angular_deg is not None:
        angular = min(angular, _positive(args.max_angular_deg, "max_angular_deg"))
    return {
        "max_linear": linear,
        "max_angular": math.radians(angular),
        "command_timeout": _positive(args.command_timeout, "command_timeout"),
    }
