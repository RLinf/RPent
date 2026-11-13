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

"""Query local or remote visual localization and publish map-to-odom corrections."""

from __future__ import annotations

import json
import math
import signal
import time
from collections import deque
from itertools import pairwise
from typing import Any

import numpy as np

from robots.lekiwi.navigation.config import localization_settings, stereo_config
from robots.lekiwi.stereo import camera_mount_transform, rigid


def planar(x: float, y: float, yaw: float) -> np.ndarray:
    """Construct an SE(2) transform."""
    c, s = math.cos(yaw), math.sin(yaw)
    return np.array([[c, -s, x], [s, c, y], [0, 0, 1.0]])


def angle(value: float) -> float:
    """Wrap an angle without discontinuities at the interpolation boundary."""
    return math.atan2(math.sin(value), math.cos(value))


class Localization:
    """Correct odom at image capture time; rejected fixes never renew freshness."""

    def __init__(self, settings: dict, *, clock_domain: str = "ros") -> None:
        self.settings = settings
        self.clock_domain = clock_domain
        self.camera_to_base = np.linalg.inv(rigid(settings["T_base_camera"]))
        self.odometry = deque(maxlen=1000)
        self.transform = None
        self.accepted_stamp = 0.0
        self.reason = "localization_missing"

    def add_odometry(self, stamp: float, x: float, y: float, yaw: float) -> None:
        """Retain ordered wheel poses; Clock resets require restarting the pose tracker."""
        if all(math.isfinite(v) for v in (stamp, x, y, yaw)) and (
            not self.odometry or stamp > self.odometry[-1][0]
        ):
            self.odometry.append((stamp, x, y, yaw))

    def receive_odometry(self, message: Any) -> None:
        """Keep wheel poses expressed as odom-to-base_link ROS messages."""
        if message.header.frame_id != "odom" or message.child_frame_id != "base_link":
            return
        p, q = message.pose.pose.position, message.pose.pose.orientation
        self.add_odometry(
            message.header.stamp.sec + message.header.stamp.nanosec * 1e-9,
            p.x,
            p.y,
            math.atan2(2 * (q.w * q.z + q.x * q.y), 1 - 2 * (q.y * q.y + q.z * q.z)),
        )

    def odom_at(self, stamp: float) -> np.ndarray:
        """Interpolate bracketing samples, never extrapolate wheel motion."""
        for before, after in pairwise(self.odometry):
            if before[0] <= stamp <= after[0] and after[0] - before[0] <= 0.25:
                f = (stamp - before[0]) / (after[0] - before[0])
                return planar(
                    before[1] + f * (after[1] - before[1]),
                    before[2] + f * (after[2] - before[2]),
                    before[3] + f * angle(after[3] - before[3]),
                )
        raise ValueError("No bracketing fresh odometry for image timestamp")

    def accept(self, result: dict, now: float) -> bool:
        """Validate identity, age and correction jumps before updating map->odom."""
        try:
            if result.get("clock_domain") != self.clock_domain:
                raise ValueError("Localization clock domain mismatch")
            stamp = result["stamp_ns"] * 1e-9
            if not math.isfinite(stamp) or stamp <= self.accepted_stamp:
                return False
            if not 0 <= now - stamp <= self.settings["localization_max_age"]:
                return False
            if (
                result["map_frame_id"] != "map"
                or result["camera_frame_id"]
                != self.settings.get("camera_frame_id", "camera_optical_frame")
                or result["camera_calibrated"] is not True
            ):
                raise ValueError("Localization map/camera identity mismatch")
            if result["status"] != "localized":
                return False
            base = rigid(result["T_map_camera"]) @ self.camera_to_base
            try:
                odom = self.odom_at(stamp)
            except ValueError:
                return False
            candidate = planar(
                base[0, 3], base[1, 3], math.atan2(base[1, 0], base[0, 0])
            ) @ np.linalg.inv(odom)
            if self.transform is not None:
                delta = np.linalg.inv(self.transform) @ candidate
                if np.linalg.norm(delta[:2, 2]) > self.settings[
                    "max_correction_translation"
                ] or abs(math.atan2(delta[1, 0], delta[0, 0])) > math.radians(
                    self.settings["max_correction_yaw_deg"]
                ):
                    raise ValueError("Localization correction jump")
            self.transform, self.accepted_stamp = candidate, stamp
            self.reason = "ready"
            return True
        except (KeyError, TypeError, ValueError) as exc:
            self.reason = str(exc)
            return False

    def ready(self, now: float) -> bool:
        """Use the last accepted fix until it expires or a structural error invalidates it."""
        return (
            self.reason == "ready"
            and 0 <= now - self.accepted_stamp <= self.settings["localization_max_age"]
        )


def publish_transform() -> None:
    """Publish map-to-odom corrections and source-time health."""
    from robots.lekiwi.stereo import StereoCalibration

    settings = dict(
        localization_settings(),
        T_base_camera=camera_mount_transform(StereoCalibration()).tolist(),
        camera_frame_id="zed_left_rectified_optical_frame",
    )
    import rclpy
    from geometry_msgs.msg import TransformStamped
    from nav_msgs.msg import Odometry
    from rclpy.executors import ExternalShutdownException
    from std_msgs.msg import String
    from tf2_ros import TransformBroadcaster

    rclpy.init(args=["--ros-args", "--disable-external-lib-logs"])
    node = rclpy.create_node("lekiwi_global_localization")
    state = Localization(settings)
    broadcaster = TransformBroadcaster(node)
    publisher = node.create_publisher(String, "/localization/status", 1)

    def localization(msg):
        try:
            result = json.loads(msg.data)
            state.accept(result, node.get_clock().now().nanoseconds * 1e-9)
        except (ValueError, TypeError) as exc:
            state.reason = str(exc)

    def tick():
        now = node.get_clock().now()
        ready = state.ready(now.nanoseconds * 1e-9)
        if ready:
            t = state.transform
            transform = TransformStamped()
            transform.header.stamp = now.to_msg()
            transform.header.frame_id, transform.child_frame_id = "map", "odom"
            transform.transform.translation.x = float(t[0, 2])
            transform.transform.translation.y = float(t[1, 2])
            yaw = math.atan2(t[1, 0], t[0, 0])
            transform.transform.rotation.z = math.sin(yaw / 2)
            transform.transform.rotation.w = math.cos(yaw / 2)
            broadcaster.sendTransform(transform)
        publisher.publish(
            String(
                data=json.dumps(
                    {
                        "ready": ready,
                        "source_stamp": state.accepted_stamp,
                        "reason": state.reason
                        if ready or state.reason != "ready"
                        else "localization_expired",
                    }
                )
            )
        )

    node.create_subscription(Odometry, "/odom", state.receive_odometry, 100)
    node.create_subscription(String, "/localization/result", localization, 1)
    node.create_timer(0.05, tick)
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


def wait_for_localization(client, stopping, *, timeout_s=60.0) -> None:
    """Wait for the configured service to load its map, allowing shutdown."""
    deadline = time.monotonic() + timeout_s
    while not stopping():
        try:
            client.call("healthz", timeout_s=1)
            return
        except Exception as exc:
            if time.monotonic() >= deadline:
                raise TimeoutError("Localization service did not become ready") from exc
            time.sleep(0.1)


def stream_localization(cfg: dict) -> None:
    """Query visual localization with the current calibration."""

    import cv2
    import rclpy
    from nav_msgs.msg import Odometry
    from rclpy.qos import qos_profile_sensor_data
    from sensor_msgs.msg import Image
    from std_msgs.msg import String

    from robots.lekiwi.navigation.bridge import image_bgr
    from robots.lekiwi.stereo import StereoCalibration
    from rpent.utils.rpc import make_rpc_client

    cal = StereoCalibration()
    camera = cal.localization_camera()
    size = (camera["width"], camera["height"])
    mount = camera_mount_transform(cal).tolist()
    client = make_rpc_client(cfg["endpoint"])
    cv2.setNumThreads(2)
    rclpy.init(args=["--ros-args", "--disable-external-lib-logs"])
    node = rclpy.create_node("lekiwi_localization_client")
    pub = node.create_publisher(String, "/localization/result", 1)
    latest = []
    wheel = Localization({"T_base_camera": mount})
    ending = False
    previous = 0
    maximum_age = stereo_config()["maximum_age_seconds"]

    def stop(*_):
        nonlocal ending
        ending = True

    signal.signal(signal.SIGTERM, stop)

    def receive(msg):
        stamp = msg.header.stamp.sec * 10**9 + msg.header.stamp.nanosec
        if stamp > previous and 0 <= (time.time_ns() - stamp) / 1e9 <= maximum_age:
            latest[:] = [(msg, stamp)]

    node.create_subscription(Odometry, "/odom", wheel.receive_odometry, 100)
    _subscription = node.create_subscription(
        Image, "/camera/image_raw", receive, qos_profile_sensor_data
    )
    try:
        wait_for_localization(client, lambda: ending)
        # Nav2's wheel bridge starts alongside this process. Wait for live
        # odometry before selecting the first camera timestamp to interpolate.
        warmup = time.monotonic() + 10
        while not ending:
            if (
                len(wheel.odometry) >= 2
                and 0 <= time.time() - wheel.odometry[-1][0] < 0.25
            ):
                break
            if time.monotonic() >= warmup:
                raise TimeoutError("No fresh wheel odometry for localization")
            rclpy.spin_once(node, timeout_sec=0.01)
        print(json.dumps({"ready": True, "camera": camera}), flush=True)
        while not ending and rclpy.ok():
            while not latest and not ending and rclpy.ok():
                rclpy.spin_once(node, timeout_sec=0.01)
            if ending or not rclpy.ok():
                break
            msg, stamp = latest.pop()
            previous = stamp
            if not 0 <= (time.time_ns() - stamp) / 1e9 <= maximum_age:
                continue
            wait_odom = time.monotonic() + 0.2
            while True:
                try:
                    pose = wheel.odom_at(stamp * 1e-9)
                    break
                except ValueError:
                    if time.monotonic() >= wait_odom:
                        pose = None
                        break
                    rclpy.spin_once(node, timeout_sec=0.01)
            if pose is None:
                continue
            raw = image_bgr(msg)
            if raw.shape != (cal.size[1], cal.size[0] * 2, 3):
                raise ValueError("Unexpected ZED dimensions")
            left = cv2.remap(raw[:, : cal.size[0]], *cal.maps[0], cv2.INTER_LINEAR)
            image = cv2.resize(left, size, interpolation=cv2.INTER_AREA)
            ok, jpeg = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, 90])
            if not ok:
                raise ValueError("Encoding failed")
            packet = {
                "stamp_ns": stamp,
                "frame_id": "zed_left_rectified_optical_frame",
                "camera": camera,
                "T_base_camera": mount,
                "odometry": [
                    float(pose[0, 2]),
                    float(pose[1, 2]),
                    math.atan2(pose[1, 0], pose[0, 0]),
                ],
                "clock_domain": "ros",
            }
            response = client.call(
                "localization.localize",
                kwargs={"packet": packet, "jpeg": jpeg},
            )
            age = (time.time_ns() - stamp) / 1e6
            pub.publish(String(data=json.dumps(response)))
            print(
                json.dumps(
                    {
                        "localization": response,
                        "source_age_at_localization_ms": age,
                        "jpeg_bytes": jpeg.nbytes,
                        "ros_published": True,
                    }
                ),
                flush=True,
            )
    finally:
        client.close()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


def main(argv=None) -> None:
    import argparse
    from pathlib import Path

    from robots.lekiwi.runtime_config import (
        localization_service_config,
        set_robot_config_path,
    )

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--robot-config", type=Path)
    parser.add_argument("mode", choices=("stream", "transform"))
    args = parser.parse_args(argv)
    set_robot_config_path(args.robot_config)
    if args.mode == "stream":
        stream_localization(localization_service_config())
    else:
        publish_transform()


if __name__ == "__main__":
    main()
