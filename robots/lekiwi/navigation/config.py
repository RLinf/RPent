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

"""Build Nav2 and local perception parameters from the robot configuration."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import yaml

from robots.lekiwi.runtime_config import (
    LEKIWI_ROOT,
    _number,
    get_robot_config_path,
    load_mapping,
)
from robots.lekiwi.stereo import load_geometry

STEREO_DEFAULTS = {
    "width": 336,
    "height": 188,
    "num_disparities": 96,
    "block_size": 5,
    "minimum_depth_m": 0.2,
    "maximum_depth_m": 3.0,
    "pixel_stride": 2,
    "minimum_obstacle_height_m": 0.04,
    "maximum_marking_depth_m": 1.5,
    "voxel_size_m": 0.04,
    "minimum_depth_coverage": 0.2,
    "maximum_age_seconds": 1.0,
}


def load_navigation_config(path: str | Path | None = None, *, raw=None) -> dict:
    """Use calibrated geometry in the odometry ground frame for local navigation."""
    config = load_geometry(path, raw=raw)
    radius = _number(
        config["robot_geometry"]["navigation"]["robot_radius_m"],
        "navigation.robot_radius_m",
        positive=True,
    )
    config["robot_radius"] = radius
    return config


def load_navigation_limits(geometry=None) -> dict:
    """Read symmetric per-axis velocity/acceleration limits from the shared YAML."""
    geometry = geometry if geometry is not None else load_geometry()
    values = geometry["robot_geometry"]["navigation"]
    result = {}
    for key in ("max_velocity", "max_accel"):
        vector = values[key]
        if not isinstance(vector, list) or len(vector) != 3:
            raise ValueError(f"navigation.{key} requires forward, lateral and yaw")
        result[key] = [_number(v, f"navigation.{key}", positive=True) for v in vector]
    return result


def stereo_config(path=None, *, geometry=None, raw=None) -> dict:
    raw = raw if raw is not None else load_mapping(path or get_robot_config_path())
    perception = raw["perception"]
    overrides = perception.get("stereo", {})
    unknown = overrides.keys() - STEREO_DEFAULTS.keys()
    if unknown:
        raise ValueError(f"Unknown stereo settings: {sorted(unknown)}")
    config = {**STEREO_DEFAULTS, **overrides}
    geometry = geometry if geometry is not None else load_geometry(raw=raw)
    config.update({key: geometry[key] for key in ("mount_xyz_m", "mount_rpy_deg")})
    config["maximum_obstacle_height_m"] = geometry["collision_height_m"]
    return config


def obstacle_parameters(
    path: str | Path | None = None, *, raw=None, stereo=None
) -> dict:
    """Configure upstream RTAB-Map segmentation in the chassis ground frame.

    Grid parameters are strings in rtabmap_util. Geometry-derived bounds cannot
    be overridden separately from the shared robot dimensions.
    """
    raw = raw if raw is not None else load_mapping(path or get_robot_config_path())
    cfg = stereo if stereo is not None else stereo_config(raw=raw)
    perception = raw["perception"]
    overrides = perception.get("obstacles", {})
    types = {
        "Grid/NormalsSegmentation": bool,
        "Grid/MaxGroundAngle": float,
        "Grid/NormalK": int,
        "Grid/ClusterRadius": float,
        "Grid/MinClusterSize": int,
    }
    if unknown := overrides.keys() - types.keys():
        raise ValueError(f"Unknown obstacle settings: {sorted(unknown)}")
    values = {
        "Grid/NormalsSegmentation": True,
        "Grid/CellSize": _number(cfg["voxel_size_m"], "voxel_size_m", positive=True),
        "Grid/MaxGroundHeight": _number(
            cfg["minimum_obstacle_height_m"], "minimum_obstacle_height_m", positive=True
        ),
        "Grid/MaxObstacleHeight": cfg["maximum_obstacle_height_m"],
        # Input already contains finite, range-filtered stereo depth rays.
        "Grid/RangeMin": 0,
        "Grid/RangeMax": 0,
        "Grid/MapFrameProjection": False,
    }
    if values["Grid/MaxGroundHeight"] >= values["Grid/MaxObstacleHeight"]:
        raise ValueError("Ground height must be below the robot collision height")
    for key, value in overrides.items():
        kind = types[key]
        if kind in (bool, int) and type(value) is not kind:
            raise ValueError(f"{key} must be {kind.__name__}")
        if kind is not bool:
            _number(value, key, positive=True)
        values[key] = value
    return {
        "frame_id": "base_link",
        "qos": 2,  # Same best-effort policy as the stereo publisher.
        "wait_for_transform": 0.2,
        **{key: str(value).lower() for key, value in values.items()},
    }


def perception_profile(*, geometry=None, stereo=None):
    if geometry is None or stereo is None:
        raw = load_mapping(get_robot_config_path())
        geometry = geometry if geometry is not None else load_navigation_config(raw=raw)
        stereo = (
            stereo if stereo is not None else stereo_config(raw=raw, geometry=geometry)
        )
    return {
        **geometry,
        "topic_prefix": "/stereo/navigation",
        "camera_frame": "zed_left_optical_frame",
        "maximum_age_seconds": stereo["maximum_age_seconds"],
    }


def localization_settings() -> dict:
    """Read live localization freshness and correction limits."""
    options = load_mapping(get_robot_config_path()).get("localization", {})
    return {
        key: _number(options.get(key, default), "localization." + key, positive=True)
        for key, default in {
            "localization_max_age": 2.0,
            "max_correction_translation": 0.5,
            "max_correction_yaw_deg": 30.0,
        }.items()
    }


def nav2_defaults() -> dict:
    """Use installed Nav2 parameters with LeKiwi's integration overrides."""
    from ament_index_python.packages import get_package_share_directory

    source = (
        Path(get_package_share_directory("nav2_bringup")) / "params/nav2_params.yaml"
    )
    result = yaml.safe_load(source.read_text())
    overrides = yaml.safe_load((LEKIWI_ROOT / "config/nav2_overrides.yaml").read_text())

    def merge(target, values):
        for key, value in values.items():
            if isinstance(value, dict):
                merge(target.setdefault(key, {}), value)
            else:
                target[key] = value

    merge(result, overrides)
    local = result["local_costmap"]["local_costmap"]["ros__parameters"]
    local["voxel_layer"].pop("scan")
    global_map = result["global_costmap"]["global_costmap"]["ros__parameters"]
    global_map.pop("obstacle_layer")
    global_map["voxel_layer"] = deepcopy(local["voxel_layer"])
    result["collision_monitor"]["ros__parameters"].pop("scan")
    return result


def parameters(map_yaml: Path | None = None) -> dict:
    raw = load_mapping(get_robot_config_path())
    geometry = load_navigation_config(raw=raw)
    cfg = stereo_config(raw=raw, geometry=geometry)
    sensor = perception_profile(geometry=geometry, stereo=cfg)

    result = nav2_defaults()
    limits = load_navigation_limits(geometry)
    x, y, yaw = limits["max_velocity"]
    ax, ay, az = limits["max_accel"]
    result["controller_server"]["ros__parameters"]["FollowPath"].update(
        vx_max=x,
        vx_min=-x,
        vy_max=y,
        wz_max=yaw,
        ax_max=ax,
        ax_min=-ax,
        ay_max=ay,
        ay_min=-ay,
        az_max=az,
    )
    result["velocity_smoother"]["ros__parameters"].update(
        max_velocity=limits["max_velocity"],
        min_velocity=[-v for v in limits["max_velocity"]],
        max_accel=limits["max_accel"],
        max_decel=[-v for v in limits["max_accel"]],
    )
    for name in ("local_costmap", "global_costmap"):
        costmap = result[name][name]["ros__parameters"]
        # Use Nav2's native circle; an explicit polygon would override the radius.
        costmap.pop("footprint", None)
        costmap["robot_radius"] = geometry["robot_radius"]
    if map_yaml is not None:
        result["bt_navigator"]["ros__parameters"]["global_frame"] = "map"
        global_map = result["global_costmap"]["global_costmap"]["ros__parameters"]
        global_map.update(global_frame="map", rolling_window=False)
        global_map["plugins"].insert(0, "static_layer")
        global_map["static_layer"] = {
            "plugin": "nav2_costmap_2d::StaticLayer",
            "map_subscribe_transient_local": True,
            "map_topic": "/map",
        }
        result["map_server"] = {
            "ros__parameters": {
                "yaml_filename": str(map_yaml.resolve()),
                "frame_id": "map",
                "topic_name": "map",
            }
        }
        result["lifecycle_manager_navigation"]["ros__parameters"]["node_names"].insert(
            0, "map_server"
        )
    for name in ("local_costmap", "global_costmap"):
        layer = result[name][name]["ros__parameters"]["voxel_layer"]
        for source, suffix in (("depth", "obstacles"), ("clearing_depth", "points")):
            layer[source]["topic"] = sensor["topic_prefix"] + "/" + suffix
            layer[source]["sensor_frame"] = sensor["camera_frame"]
        # Mark segmented obstacles and clear only along valid measured depth rays.
        # Both sources update the same voxel grid, so dynamic marks can disappear.
        layer["max_obstacle_height"] = geometry["collision_height_m"]
        layer["depth"]["max_obstacle_height"] = geometry["collision_height_m"]
        layer["z_resolution"] = max(
            layer["z_resolution"],
            (geometry["collision_height_m"] - layer["origin_z"] + 0.01)
            / layer["z_voxels"],
        )
        layer["depth"]["obstacle_max_range"] = min(
            cfg["maximum_marking_depth_m"], cfg["maximum_depth_m"]
        )
        layer["clearing_depth"]["raytrace_max_range"] = cfg["maximum_depth_m"]
    result["obstacles_detection"] = {
        "ros__parameters": obstacle_parameters(raw=raw, stereo=cfg)
    }
    result["collision_monitor"]["ros__parameters"]["depth"]["source_timeout"] = cfg[
        "maximum_age_seconds"
    ]
    result["collision_monitor"]["ros__parameters"]["depth"]["topic"] = (
        sensor["topic_prefix"] + "/obstacles"
    )
    result["collision_monitor"]["ros__parameters"]["depth"]["max_height"] = geometry[
        "collision_height_m"
    ]
    return result
