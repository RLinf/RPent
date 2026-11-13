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

"""Bridge Env-owned camera and base feedback to ROS."""

from __future__ import annotations

import argparse
import math
import signal
import threading
import time
import urllib.parse
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from rpent.utils.rpc import RpcError, make_rpc_client


class BaseClient:
    """Use the Env RPC for generation-bound velocity commands and wheel samples."""

    def __init__(self, endpoint: str, timeout: float = 0.2) -> None:
        parsed = urllib.parse.urlsplit(endpoint)
        if (
            parsed.scheme != "http"
            or parsed.hostname not in {"127.0.0.1", "localhost"}
            or parsed.username
            or parsed.password
            or parsed.path not in {"", "/"}
            or parsed.query
            or parsed.fragment
        ):
            raise ValueError("base endpoint must be a loopback HTTP URL")
        if not math.isfinite(timeout) or timeout <= 0:
            raise ValueError("RPC timeout must be finite and positive")
        self.rpc = make_rpc_client(endpoint)
        self.timeout = timeout
        self.generation: str | None = None

    def call(self, method: str, **kwargs: Any) -> Any:
        return self.rpc.call(method, kwargs=kwargs, timeout_s=self.timeout)

    def connect(self) -> dict[str, Any]:
        result = self.call("base.navigation_info")
        self.generation = result["generation"]
        return result

    def send(self, velocity: list[float], expires_at: float) -> dict[str, Any]:
        return self.call(
            "base.set_velocity",
            generation=self.generation,
            velocity=velocity,
            expires_at=expires_at,
        )

    def state(self) -> dict[str, Any]:
        return self.call("base.get_state")

    def close(self) -> None:
        self.rpc.close()


def image_bgr(message: Any) -> np.ndarray:
    if message.encoding not in ("rgb8", "bgr8"):
        raise ValueError("Expected RGB/BGR image")
    pixels = (
        np.frombuffer(message.data, np.uint8)
        .reshape(message.height, message.step)[:, : message.width * 3]
        .reshape(message.height, message.width, 3)
    )
    return (
        cv2.cvtColor(pixels, cv2.COLOR_RGB2BGR)
        if message.encoding == "rgb8"
        else pixels.copy()
    )


def frame_to_message(
    packet: dict, message: Any, *, offset_ns: int, after: float, max_age: float
) -> float:
    """Preserve capture time and reject stale or repeated local frames."""
    stamp = packet["source_stamp"]
    frame = packet["frame"]
    if packet["clock_domain"] != "host_monotonic":
        raise ValueError("Expected the local Env monotonic clock")
    if not math.isfinite(stamp):
        raise ValueError("Invalid camera timestamp")
    if not 0 <= time.monotonic() - stamp <= max_age:
        return after
    if frame.shape not in {(376, 1344, 3), (720, 2560, 3)} or frame.dtype != np.uint8:
        raise ValueError("Expected native VGA or HD720 BGR stereo")
    if stamp <= after:
        return after
    ns = round(stamp * 1e9) + offset_ns
    message.header.stamp.sec, message.header.stamp.nanosec = divmod(ns, 10**9)
    message.header.frame_id = "zed_stereo_raw_frame"
    message.height, message.width = frame.shape[:2]
    message.encoding, message.is_bigendian = "bgr8", False
    message.step = message.width * 3
    message.data = frame.tobytes()
    return stamp


def publish_camera(*, endpoint: str) -> None:
    """Publish the Env-owned camera stream to ROS."""
    from urllib.parse import urlsplit

    import rclpy
    from rclpy.qos import qos_profile_sensor_data
    from rclpy.signals import SignalHandlerOptions
    from sensor_msgs.msg import Image

    if urlsplit(endpoint).hostname not in {"localhost", "127.0.0.1", "::1"}:
        raise ValueError("Camera bridge requires a local Env endpoint")
    client = make_rpc_client(endpoint)
    stop = threading.Event()
    offset_ns = time.time_ns() - time.monotonic_ns()
    signal.signal(signal.SIGTERM, lambda *_: stop.set())
    signal.signal(signal.SIGINT, lambda *_: stop.set())
    node = None
    try:
        rclpy.init(
            args=["--ros-args", "--disable-external-lib-logs"],
            signal_handler_options=SignalHandlerOptions.NO,
        )
        node = rclpy.create_node("lekiwi_camera")
        from geometry_msgs.msg import TransformStamped
        from tf2_ros import StaticTransformBroadcaster

        from robots.lekiwi.navigation.config import perception_profile
        from robots.lekiwi.stereo import optical_to_base_quaternion

        config = perception_profile()
        static_tf = StaticTransformBroadcaster(node)
        transform = TransformStamped()
        transform.header.stamp = node.get_clock().now().to_msg()
        transform.header.frame_id = "base_link"
        transform.child_frame_id = config["camera_frame"]
        for key, value in zip("xyz", config["mount_xyz_m"]):
            setattr(transform.transform.translation, key, value)
        for key, value in zip(
            "xyzw", optical_to_base_quaternion(tuple(config["mount_rpy_deg"]))
        ):
            setattr(transform.transform.rotation, key, float(value))
        static_tf.sendTransform(transform)
        publisher = node.create_publisher(
            Image, "/camera/image_raw", qos_profile_sensor_data
        )
        previous = 0.0
        while not stop.is_set() and rclpy.ok():
            started = time.monotonic()
            try:
                # Allow the 0.5 s frame wait and capture-thread shutdown on reconnect.
                packet = client.call("env.get_stereo_frame", timeout_s=3.0)
            except RpcError as exc:
                if exc.server_traceback:
                    raise
                node.get_logger().warning(f"Camera RPC unavailable: {exc}")
                packet = None
            if packet is not None:
                message = Image()
                stamp = frame_to_message(
                    packet,
                    message,
                    offset_ns=offset_ns,
                    after=previous,
                    max_age=config["maximum_age_seconds"],
                )
                if stamp > previous:
                    publisher.publish(message)
                    previous = stamp
            stop.wait(max(0, 1 / 15 - (time.monotonic() - started)))
    finally:
        client.close()
        if node is not None:
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


def velocity_command(
    message: Any, *, now: float, max_age: float, base_frame: str
) -> tuple[list[float], float]:
    """Validate a stamped planar command, preserving its original expiry."""
    if message.header.frame_id not in {"", base_frame}:
        raise ValueError(
            f"velocity must be in {base_frame}, not {message.header.frame_id}"
        )
    stamp = message.header.stamp.sec + message.header.stamp.nanosec * 1e-9
    if stamp <= 0 or stamp > now + 0.05 or now - stamp >= max_age:
        raise ValueError("velocity timestamp is missing, future-dated, or stale")
    twist = message.twist
    values = [
        twist.linear.x,
        twist.linear.y,
        twist.linear.z,
        twist.angular.x,
        twist.angular.y,
        twist.angular.z,
    ]
    if not all(math.isfinite(value) for value in values):
        raise ValueError("velocity must contain only finite values")
    if any(value != 0 for value in [twist.linear.z, twist.angular.x, twist.angular.y]):
        raise ValueError("LeKiwi accepts only planar velocity")
    return [twist.linear.x, twist.linear.y, twist.angular.z], min(stamp, now) + max_age


def bridge(argv=None) -> None:
    from robots.lekiwi.runtime_config import set_robot_config_path

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("base", "camera"))
    parser.add_argument("--robot-config", type=Path)
    parser.add_argument("--endpoint", required=True)
    parser.add_argument(
        "--require-mock", action="store_true", help="refuse physical hardware"
    )
    parser.add_argument("--command-topic", default="/lekiwi/cmd_vel")
    parser.add_argument("--wheel-topic", default="/lekiwi/wheel_velocity")
    parser.add_argument("--base-frame", default="base_link")
    parser.add_argument(
        "--odometry", action="store_true", help="publish fresh EKF odom/TF"
    )
    parser.add_argument(
        "--linear-stddev", type=float, default=0.03, help="initial m/s noise estimate"
    )
    parser.add_argument(
        "--angular-stddev", type=float, default=0.1, help="initial rad/s noise estimate"
    )
    parser.add_argument("--command-max-age", type=float, default=0.5)
    parser.add_argument("--feedback-max-age", type=float, default=0.5)
    parser.add_argument("--feedback-frequency", type=float, default=20.0)
    parser.add_argument("--rpc-timeout", type=float, default=0.2)
    args, ros_args = parser.parse_known_args(argv)
    set_robot_config_path(args.robot_config)
    if args.mode == "camera":
        if ros_args:
            parser.error(f"unrecognized arguments: {' '.join(ros_args)}")
        return publish_camera(endpoint=args.endpoint)
    for name in (
        "command_max_age",
        "feedback_max_age",
        "feedback_frequency",
        "linear_stddev",
        "angular_stddev",
    ):
        value = getattr(args, name)
        if not math.isfinite(value) or value <= 0:
            parser.error(f"{name} must be finite and positive")

    import rclpy
    from geometry_msgs.msg import (
        TransformStamped,
        TwistStamped,
        TwistWithCovarianceStamped,
    )
    from nav_msgs.msg import Odometry
    from rclpy.executors import ExternalShutdownException
    from rclpy.node import Node
    from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
    from tf2_ros import TransformBroadcaster

    class BaseBridge(Node):
        def __init__(self) -> None:
            super().__init__("lekiwi_base_bridge")
            if self.get_parameter("use_sim_time").value:
                raise ValueError("the physical base bridge requires use_sim_time=false")
            self.client = BaseClient(args.endpoint, args.rpc_timeout)
            self.last_stamp = 0.0
            self.last_feedback_stamp = 0.0
            self.feedback_valid_until = 0.0
            self.last_odometry_stamp = 0.0
            qos = QoSProfile(
                history=HistoryPolicy.KEEP_LAST,
                depth=1,
                reliability=ReliabilityPolicy.RELIABLE,
                durability=DurabilityPolicy.VOLATILE,
            )
            self.wheel_feedback = self.create_publisher(
                TwistWithCovarianceStamped, args.wheel_topic, qos
            )
            if args.odometry:
                self.odometry = self.create_publisher(Odometry, "/odom", qos)
                self.tf = TransformBroadcaster(self)
                self.create_subscription(
                    Odometry, "/lekiwi/odometry_raw", self.publish_odometry, qos
                )
            self.create_subscription(
                TwistStamped, args.command_topic, self.command, qos
            )
            self.create_timer(1 / args.feedback_frequency, self.publish_feedback)
            info = self.client.connect()
            if args.require_mock and info["simulated"] is not True:
                self.client.close()
                raise RuntimeError("this bridge invocation requires a mock base")
            self.max_age = min(args.command_max_age, info["command_timeout_s"])
            mode = "MOCK (no hardware)" if info["simulated"] else "HARDWARE"
            self.get_logger().info(
                f"{mode}: base connected; waiting for stamped velocity"
            )

        def command(self, message: Any) -> None:
            try:
                now = self.get_clock().now().nanoseconds * 1e-9
                velocity, expiry = velocity_command(
                    message, now=now, max_age=self.max_age, base_frame=args.base_frame
                )
                stamp = message.header.stamp.sec + message.header.stamp.nanosec * 1e-9
                if stamp <= self.last_stamp:
                    raise ValueError("velocity timestamp must increase")
            except ValueError as exc:
                self.get_logger().warning(f"Rejected velocity: {exc}")
                return
            # Callback runs once per new message, never renewing an old target.
            self.last_stamp = stamp
            self.client.send(velocity, expires_at=time.time() + (expiry - now))

        def publish_feedback(self) -> None:
            state = self.client.state()
            if state["fault"] or state["closed"]:
                raise RuntimeError(f"base unavailable: {state}")
            stamp = state["sampled_at"]
            age = max(state["sample_age_s"], time.time() - stamp)
            if age > args.feedback_max_age or stamp > time.time() + 0.05:
                raise RuntimeError("base feedback is stale")
            if stamp < self.last_feedback_stamp:
                raise RuntimeError(
                    "base feedback clock moved backwards; restart odometry"
                )
            if stamp == self.last_feedback_stamp:
                return
            self.last_feedback_stamp = stamp
            self.feedback_valid_until = time.monotonic() + args.feedback_max_age - age
            wheel = TwistWithCovarianceStamped()
            wheel.header.stamp.sec = int(stamp)
            wheel.header.stamp.nanosec = int((stamp - int(stamp)) * 1e9)
            wheel.header.frame_id = args.base_frame
            x, y, yaw = state["measured_velocity"]
            wheel.twist.twist.linear.x = float(x)
            wheel.twist.twist.linear.y = float(y)
            wheel.twist.twist.angular.z = float(yaw)
            # Initial uncertainty settings, not calibrated sensor accuracy.
            for index in (0, 7):
                wheel.twist.covariance[index] = args.linear_stddev**2
            wheel.twist.covariance[35] = args.angular_stddev**2
            for index in (14, 21, 28):
                wheel.twist.covariance[index] = 1e6
            self.wheel_feedback.publish(wheel)

        def publish_odometry(self, message: Any) -> None:
            # EKF sensor_timeout only changes its prediction behavior: it does
            # not stop output. Only this bridge may publish navigation odom/TF.
            if self.last_feedback_stamp == 0:
                return
            if time.monotonic() >= self.feedback_valid_until:
                raise RuntimeError("wheel feedback expired; odometry disabled")
            if (
                message.header.frame_id != "odom"
                or message.child_frame_id != args.base_frame
            ):
                raise ValueError("unexpected EKF odometry frames")
            stamp = message.header.stamp.sec + message.header.stamp.nanosec * 1e-9
            age = self.get_clock().now().nanoseconds * 1e-9 - stamp
            if age < -0.05 or age > args.feedback_max_age:
                raise RuntimeError("EKF odometry timestamp is stale or future-dated")
            if stamp <= self.last_odometry_stamp:
                return
            self.last_odometry_stamp = stamp
            self.odometry.publish(message)
            transform = TransformStamped()
            transform.header = message.header
            transform.child_frame_id = message.child_frame_id
            transform.transform.translation.x = message.pose.pose.position.x
            transform.transform.translation.y = message.pose.pose.position.y
            transform.transform.translation.z = message.pose.pose.position.z
            transform.transform.rotation = message.pose.pose.orientation
            self.tf.sendTransform(transform)

    rclpy.init(args=ros_args)
    node = None
    try:
        node = BaseBridge()
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        if node is not None:
            try:
                node.client.close()
            except Exception as exc:
                node.get_logger().error(
                    f"Client close failed; command timeout remains active: {exc}"
                )
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    bridge()
