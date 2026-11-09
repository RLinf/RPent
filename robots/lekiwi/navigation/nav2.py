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

"""Persistent Nav2 Action client, velocity gate and navigation status RPC."""

from __future__ import annotations

import json
import math
import queue
import signal
import threading
import time
import uuid
from concurrent.futures import Future
from concurrent.futures import TimeoutError as FutureTimeoutError
from functools import partial

from rpent.utils.rpc import RpcFacade


class NavigationFacade(RpcFacade):
    """Use the existing RPC transport; ROS callbacks own all navigation state."""

    def __init__(self, node):
        super().__init__()
        self._rpc.update(
            {
                "navigation.navigate": partial(node.request, node.begin),
                "navigation.state": partial(node.request, node.state),
                "navigation.cancel": partial(node.request, node.cancel_motion),
            }
        )
        self._readonly_methods.add("navigation.state")


class Navigator:
    """Navigation state owned exclusively by the ROS executor thread."""

    def __init__(self, node):
        from geometry_msgs.msg import TwistStamped
        from nav2_msgs.action import NavigateToPose
        from rclpy.action import ActionClient
        from std_msgs.msg import String
        from tf2_ros import Buffer, TransformListener

        from robots.lekiwi.navigation.config import localization_settings

        self.node = node
        self.settings = localization_settings()
        self.action = ActionClient(node, NavigateToPose, "/navigate_to_pose")
        self.publisher = node.create_publisher(TwistStamped, "/lekiwi/cmd_vel", 1)
        node.create_subscription(TwistStamped, "/lekiwi/nav_checked", self.forward, 1)
        node.create_subscription(String, "/localization/status", self.localize, 1)
        self.buffer = Buffer()
        self.listener = TransformListener(self.buffer, node)
        self.localization = {}
        self.operation = {"state": "idle"}
        self.goal_handle = None
        self.cancel_reason = None
        self.unavailable_since = None
        self.deadline = self.armed_at = 0.0
        self.requests = queue.Queue()
        node.create_timer(0.05, self.tick)

    def localize(self, msg):
        self.localization = json.loads(msg.data)

    def health(self):
        from rclpy.time import Time
        from tf2_ros import TransformException

        now = self.node.get_clock().now().nanoseconds * 1e-9
        status = self.localization
        if (
            not status.get("ready")
            or not 0
            <= now - status.get("source_stamp", 0)
            <= self.settings["localization_max_age"]
        ):
            return {"ready": False, "reason": "localization_unavailable"}
        try:
            tf = self.buffer.lookup_transform("map", "base_link", Time())
            stamp = tf.header.stamp.sec + tf.header.stamp.nanosec * 1e-9
            if not 0 <= now - stamp <= self.settings["localization_max_age"]:
                raise ValueError("stale_transform")
            t, q = tf.transform.translation, tf.transform.rotation
        except (TransformException, ValueError) as exc:
            return {"ready": False, "reason": str(exc)}
        ready = self.action.server_is_ready()
        return {
            "ready": ready,
            "reason": "ready" if ready else "nav2_unavailable",
            "pose": [
                t.x,
                t.y,
                math.atan2(
                    2 * (q.w * q.z + q.x * q.y), 1 - 2 * (q.y * q.y + q.z * q.z)
                ),
            ],
            "source_age_s": now - status["source_stamp"],
        }

    def state(self):
        return dict(self.health(), operation=dict(self.operation))

    def cancel_motion(self):
        self.cancel("cancelled")
        self.zero()
        return dict(self.operation)

    def request(self, handler, *args, **kwargs):
        request = Future()
        self.requests.put((request, handler, args, kwargs))
        try:
            return request.result(timeout=3)
        except FutureTimeoutError as exc:
            request.cancel()
            raise TimeoutError("Navigation executor did not respond") from exc

    def begin(self, target, goal, timeout_s):
        from geometry_msgs.msg import PoseStamped
        from nav2_msgs.action import NavigateToPose

        if not math.isfinite(timeout_s) or timeout_s <= 0:
            raise ValueError("timeout_s must be finite and positive")
        if self.operation["state"] == "running":
            raise RuntimeError("Another navigation goal is active")
        if not self.health()["ready"]:
            raise RuntimeError("Navigation is not ready")
        x, y = goal["xy"]
        yaw = goal["yaw"]
        if not all(math.isfinite(v) for v in (x, y, yaw)):
            raise ValueError("Goal coordinates must be finite")
        pose = PoseStamped()
        pose.header.frame_id = "map"
        pose.header.stamp = self.node.get_clock().now().to_msg()
        pose.pose.position.x, pose.pose.position.y = float(x), float(y)
        pose.pose.orientation.z, pose.pose.orientation.w = (
            math.sin(yaw / 2),
            math.cos(yaw / 2),
        )
        self.operation = {
            "id": uuid.uuid4().hex,
            "state": "running",
            "target": target,
            "node_id": goal["node_id"],
            "xy": [x, y],
        }
        self.cancel_reason = None
        self.unavailable_since = None
        self.deadline = time.monotonic() + timeout_s
        self.armed_at = self.node.get_clock().now().nanoseconds * 1e-9
        future = self.action.send_goal_async(NavigateToPose.Goal(pose=pose))
        future.add_done_callback(self.accepted)
        return dict(self.operation)

    def accepted(self, future):
        self.goal_handle = future.result()
        if not self.goal_handle.accepted:
            self.finish("failed", "Nav2 rejected the goal")
            return
        self.goal_handle.get_result_async().add_done_callback(self.completed)
        if self.cancel_reason is not None:
            self.goal_handle.cancel_goal_async()

    def completed(self, future):
        from action_msgs.msg import GoalStatus

        status = future.result().status
        state = {
            GoalStatus.STATUS_SUCCEEDED: "succeeded",
            GoalStatus.STATUS_CANCELED: "cancelled",
        }.get(status, "failed")
        if self.cancel_reason and self.cancel_reason != "cancelled":
            state = "failed"
        self.finish(state, self.cancel_reason)

    def zero(self):
        from geometry_msgs.msg import TwistStamped

        msg = TwistStamped()
        msg.header.stamp = self.node.get_clock().now().to_msg()
        msg.header.frame_id = "base_link"
        self.publisher.publish(msg)

    def finish(self, state, error=None):
        self.zero()
        self.operation["state"] = state
        if error and error != "cancelled":
            self.operation["error"] = error
        self.goal_handle = None

    def cancel(self, reason):
        if self.operation["state"] != "running" or self.cancel_reason is not None:
            return
        self.cancel_reason = reason
        self.zero()
        if self.goal_handle is not None:
            self.goal_handle.cancel_goal_async()

    def forward(self, msg):
        stamp = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        if (
            self.operation["state"] == "running"
            and self.goal_handle is not None
            and self.cancel_reason is None
            and stamp >= self.armed_at
            and self.health()["ready"]
        ):
            self.publisher.publish(msg)

    def tick(self):
        while not self.requests.empty():
            future, handler, args, kwargs = self.requests.get_nowait()
            if not future.set_running_or_notify_cancel():
                continue
            try:
                future.set_result(handler(*args, **kwargs))
            except Exception as exc:
                future.set_exception(exc)
        health = self.health()
        if self.operation["state"] == "running":
            now = time.monotonic()
            if now >= self.deadline:
                self.cancel("Navigation goal timed out")
            elif health["ready"]:
                self.unavailable_since = None
            else:
                self.zero()
                if self.unavailable_since is None:
                    self.unavailable_since = now
                if (
                    now - self.unavailable_since
                    >= self.settings["localization_max_age"]
                ):
                    self.cancel(health["reason"])


def serve(port: int) -> None:
    import rclpy
    from rclpy.signals import SignalHandlerOptions

    rclpy.init(signal_handler_options=SignalHandlerOptions.NO)
    node = rclpy.create_node("lekiwi_navigation")
    navigator = Navigator(node)
    facade = NavigationFacade(navigator)
    signal.signal(signal.SIGTERM, lambda *_: facade._shutdown_event.set())
    signal.signal(signal.SIGINT, lambda *_: facade._shutdown_event.set())

    def rpc():
        try:
            facade.serve(transport="http", host="127.0.0.1", port=port)
        finally:
            facade._shutdown_event.set()

    thread = threading.Thread(target=rpc, daemon=True)
    thread.start()
    try:
        while not facade._shutdown_event.is_set():
            rclpy.spin_once(node, timeout_sec=0.05)
    finally:
        facade._shutdown_event.set()
        navigator.cancel("cancelled")
        deadline = time.monotonic() + 2
        while navigator.operation["state"] == "running" and time.monotonic() < deadline:
            rclpy.spin_once(node, timeout_sec=0.05)
        thread.join(timeout=5)
        node.destroy_node()
        rclpy.shutdown()


def main(argv=None) -> None:
    import argparse
    from pathlib import Path

    from robots.lekiwi.runtime_config import set_robot_config_path

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--robot-config", type=Path)
    parser.add_argument("--port", type=int, required=True)
    args = parser.parse_args(argv)
    set_robot_config_path(args.robot_config)
    serve(args.port)


if __name__ == "__main__":
    main()
