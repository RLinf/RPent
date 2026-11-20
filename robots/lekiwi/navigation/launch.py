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

"""Launch the Pi ROS graph with ROS launch process and shutdown management."""

import argparse
import sys
import tempfile
from pathlib import Path

import yaml


def run_navigation(
    endpoint: str,
    hardware: bool,
    map_yaml: Path,
    config: str,
    rpc_port: int,
    parent_watch: bool = False,
) -> int:
    from ament_index_python.packages import get_package_share_directory
    from launch import LaunchDescription, LaunchService
    from launch.actions import EmitEvent, ExecuteProcess, RegisterEventHandler
    from launch.event_handlers import OnProcessExit
    from launch.events import Shutdown
    from launch_ros.actions import Node

    from robots.lekiwi.runtime_config import get_robot_config_path
    from rpent.utils.daemon import watch_parent_death

    folder = Path(__file__).resolve().parents[1] / "config"
    tree = str(
        Path(get_package_share_directory("nav2_bt_navigator"))
        / "behavior_trees/navigate_w_replanning_time.xml"
    )
    with tempfile.TemporaryDirectory(prefix="lekiwi-nav2-") as directory:
        params = Path(directory) / "nav2.yaml"
        params.write_text(config)
        processes = [
            Node(
                package="robot_localization",
                executable="ekf_node",
                name="lekiwi_wheel_ekf",
                parameters=[str(folder / "ekf_wheel.yaml")],
                remappings=[("odometry/filtered", "/lekiwi/odometry_raw")],
            )
        ]
        bridge_args = ["--endpoint", endpoint, "--odometry"]
        if not hardware:
            bridge_args.append("--require-mock")
        workers = [
            ("bridge", ["base", *bridge_args]),
            ("bridge", ["camera", "--endpoint", endpoint]),
            ("perception", []),
            ("localization", ["stream"]),
            ("localization", ["transform"]),
            ("nav2", ["--port", str(rpc_port)]),
        ]
        for module, arguments in workers:
            processes.append(
                ExecuteProcess(
                    cmd=[
                        sys.executable,
                        "-m",
                        f"robots.lekiwi.navigation.{module}",
                        "--robot-config",
                        str(get_robot_config_path()),
                        *arguments,
                    ]
                )
            )
        nodes = [
            (
                "rtabmap_util",
                "obstacles_detection",
                [
                    ("cloud", "/stereo/navigation/points"),
                    ("obstacles", "/stereo/navigation/obstacles"),
                    ("ground", "/stereo/navigation/ground"),
                ],
            ),
            ("nav2_controller", "controller_server", [("cmd_vel", "/lekiwi/nav_raw")]),
            ("nav2_planner", "planner_server", []),
            ("nav2_bt_navigator", "bt_navigator", []),
            (
                "nav2_velocity_smoother",
                "velocity_smoother",
                [
                    ("cmd_vel", "/lekiwi/nav_raw"),
                    ("cmd_vel_smoothed", "/lekiwi/nav_smoothed"),
                ],
            ),
            ("nav2_collision_monitor", "collision_monitor", []),
            ("nav2_lifecycle_manager", "lifecycle_manager", []),
        ]
        if map_yaml:
            nodes.insert(0, ("nav2_map_server", "map_server", []))
        for package, executable, remappings in nodes:
            overrides = (
                {"default_nav_to_pose_bt_xml": tree}
                if executable == "bt_navigator"
                else {}
            )
            processes.append(
                Node(
                    package=package,
                    executable=executable,
                    name="lifecycle_manager_navigation"
                    if executable == "lifecycle_manager"
                    else executable,
                    parameters=[str(params), overrides],
                    remappings=remappings,
                    arguments=["--ros-args", "--disable-external-lib-logs"],
                )
            )
        # ROS launch owns the entire graph; any worker exit stops its peers.
        shutdown = RegisterEventHandler(
            OnProcessExit(
                on_exit=[EmitEvent(event=Shutdown(reason="Navigation worker exited"))]
            )
        )
        service = LaunchService()
        service.include_launch_description(LaunchDescription([shutdown, *processes]))
        watcher = (
            watch_parent_death(
                lambda: service.emit_event(Shutdown(reason="Env process exited"))
            )
            if parent_watch
            else None
        )
        try:
            return service.run()
        finally:
            if watcher is not None:
                watcher.close()


def main(argv=None) -> None:
    from robots.lekiwi.navigation.config import parameters
    from robots.lekiwi.runtime_config import set_robot_config_path

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--robot-config", type=Path)
    parser.add_argument("--endpoint", required=True)
    parser.add_argument("--map", type=Path, required=True)
    parser.add_argument("--rpc-port", type=int, required=True)
    parser.add_argument("--hardware", action="store_true")
    parser.add_argument("--parent-watch", action="store_true")
    args = parser.parse_args(argv)
    set_robot_config_path(args.robot_config)
    config = yaml.safe_dump(parameters(args.map), sort_keys=False)
    raise SystemExit(
        run_navigation(
            args.endpoint,
            args.hardware,
            args.map,
            config,
            args.rpc_port,
            args.parent_watch,
        )
    )


if __name__ == "__main__":
    main()
