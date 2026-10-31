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

"""User-facing config, developer defaults, and the RLinf adapter for one LeKiwi.

Users edit only ``example.yaml``: machine identity (serial port, camera
serial) and drive limits. The RLinf field values RPent deliberately tunes
live here as developer defaults and are applied as ``override_cfg`` over
RLinf's own dataclass defaults, so they are never restated in the YAML.

``load_runtime_config`` turns that into the RLinf adapter config. The
generic helpers are shared with :mod:`robots.franka`.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from omegaconf import DictConfig, OmegaConf

from robots.franka.runtime_config import load_mapping, strict_mapping

DEFAULT_CONFIG = Path(__file__).with_name("config") / "example.yaml"

_robot_config_path: Path | None = None


def set_robot_config_path(path: str | Path | None) -> None:
    """Record the ``--robot-config`` CLI value (``None`` when absent)."""
    global _robot_config_path
    _robot_config_path = Path(path).expanduser() if path else None


def get_robot_config_path(default: str | Path = DEFAULT_CONFIG) -> Path:
    """Return the ``--robot-config`` override, or the robot's packaged default."""
    return Path(_robot_config_path or default)


# ---------------------------------------------------------------------------
# Developer defaults
# ---------------------------------------------------------------------------

# Primitive-control knobs consumed by the RPent LeKiwi env server. RLinf has
# no equivalent field, so these live here rather than in any RLinf dataclass.
CONTROL = {
    "drive": {"max_duration_s": 10.0},
    # Longest the server keeps a held velocity after the last refresh: the
    # dead-man that stops the base when the operator's session goes away.
    "hold": {"max_ttl_s": 2.0},
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


def flatten_control(control: dict[str, Any]) -> dict[str, Any]:
    """Flatten nested ``CONTROL`` defaults into the flat env-server keys."""
    return {
        "drive_max_duration_s": float(control["drive"]["max_duration_s"]),
        "hold_max_ttl_s": float(control["hold"]["max_ttl_s"]),
    }


def _require_mapping(value: Any, name: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{name} must be a mapping")
    return value


@dataclass(frozen=True)
class LeKiwiRuntimeConfig:
    """Generated RLinf adapter config and RPent primitive settings."""

    rlinf: DictConfig
    controller: dict[str, Any]


def load_runtime_config(path: str | Path | None) -> LeKiwiRuntimeConfig:
    """Load the user YAML, apply developer defaults, and build the adapter."""
    # Lazy RLinf imports: keys are validated against these dataclasses (drift
    # guard), deferred so importing this module stays RLinf-free.
    from rlinf.envs.real.lekiwi.base import LeKiwiEnvConfig
    from rlinf.robotics.robots.lekiwi import LeKiwiConfig

    raw = load_mapping(path or get_robot_config_path())
    robot = _require_mapping(raw.get("robot"), "robot")
    if "port" not in robot:
        raise ValueError("robot.port is required: the wheel bus needs a serial port")
    cameras = _require_mapping(raw.get("cameras"), "cameras")
    devices = _require_mapping(cameras.get("devices"), "cameras.devices")

    camera_serials: list[str] = []
    camera_names: dict[str, str] = {}
    camera_types: set[str] = set()
    main_image_keys: list[str] = []
    for name, value in devices.items():
        device = _require_mapping(value, f"cameras.devices.{name}")
        serial = str(device["serial"])
        camera_serials.append(serial)
        camera_names[serial] = str(name)
        camera_types.add(str(device.get("type", "webcam")))
        if bool(device.get("main", False)):
            main_image_keys.append(str(name))
    if len(main_image_keys) != 1:
        raise ValueError("exactly one camera device must set main: true")
    if len(camera_types) != 1:
        raise ValueError("the LeKiwi currently requires one camera type")

    drive = raw.get("drive") or {}
    if not isinstance(drive, dict):
        raise ValueError("drive must be a mapping")

    hardware = strict_mapping(
        LeKiwiConfig,
        {
            "port": str(robot["port"]),
            "wheel_radius": robot.get("wheel_radius", 0.05),
            "base_radius": robot.get("base_radius", 0.125),
            "max_raw": robot.get("max_raw", 3000),
            "camera_serials": camera_serials,
            "camera_type": camera_types.pop(),
            "node_rank": 0,
        },
        where="cluster.node_groups[].hardware.configs[]",
    )
    override_cfg = strict_mapping(
        LeKiwiEnvConfig,
        {
            **ENV_DEFAULTS,
            "camera_names": camera_names,
            **drive,
        },
        where="env.eval.override_cfg",
    )

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
                    "main_image_key": main_image_keys[0],
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
        controller=flatten_control(CONTROL),
    )
