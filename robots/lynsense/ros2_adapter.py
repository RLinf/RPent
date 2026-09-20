# Copyright 2026 The RPent Authors.
# SPDX-License-Identifier: Apache-2.0

"""Owned, subscription-only ROS runtime with no import-time ROS dependency."""

from __future__ import annotations

import os
import re
import threading
import time
from datetime import datetime, timezone
from typing import Any

from robots.lynsense.state import StateCache, finite_number

_CONTEXT_LOCK = threading.Lock()
_TOPIC = re.compile(r"(?:/[A-Za-z_][A-Za-z0-9_]*)+")


class Ros2StateAdapter:
    """Read two ROS state streams; connection does not imply motion readiness."""

    _CLOSE_TIMEOUT_S = 2.0

    def __init__(
        self,
        *,
        expected_domain_id: int = 3,
        joint_topic: str = "/right_xarm/joint_states",
        robot_topic: str = "/right_xarm/robot_states",
        connect_timeout_s: float = 5.0,
        stale_after_s: float = 2.0,
    ) -> None:
        if type(expected_domain_id) is not int or not 0 <= expected_domain_id <= 232:
            raise ValueError("expected_domain_id must be an integer in 0..232")
        for topic in (joint_topic, robot_topic):
            if not isinstance(topic, str) or not _TOPIC.fullmatch(topic) or "__" in topic:
                raise ValueError("state topics must be valid absolute ROS topic names")
        if joint_topic == robot_topic:
            raise ValueError("joint and robot state topics must differ")
        self._timeout = finite_number(connect_timeout_s)
        if self._timeout <= 0:
            raise ValueError("connect_timeout_s must be positive")
        self._cache = StateCache(stale_after_s)
        self._domain = expected_domain_id
        self._joint_topic = joint_topic
        self._robot_topic = robot_topic
        self._condition = threading.Condition()
        self._lifecycle = threading.RLock()
        self._stop = threading.Event()
        self._rclpy: Any = None
        self._context: Any = None
        self._node: Any = None
        self._executor: Any = None
        self._thread: threading.Thread | None = None
        self._owns_context = False
        self._toolkit_claimed = False
        self._started = False
        self._closed = False
        self._failure: str | None = None
        self._runtime_failure: str | None = None

    def claim_toolkit(self) -> None:
        """Transfer an inert adapter to exactly one toolkit, even through aliases."""
        with self._lifecycle:
            if self._toolkit_claimed:
                raise RuntimeError("adapter is already claimed by a toolkit")
            if self._started or self._stop.is_set():
                raise RuntimeError("only an inert adapter can be claimed by a toolkit")
            self._toolkit_claimed = True

    def _start_ros(self) -> None:
        domain = os.environ.get("ROS_DOMAIN_ID")
        if domain is None or not domain.isdecimal() or int(domain) != self._domain:
            raise ValueError(f"ROS_DOMAIN_ID must explicitly equal {self._domain}; got {domain!r}")

        # Only the selected robot's connection path loads ROS and generated types.
        import rclpy
        from rclpy.executors import SingleThreadedExecutor
        from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
        from rclpy.signals import SignalHandlerOptions
        from sensor_msgs.msg import JointState
        from xarm_msgs.msg import RobotMsg

        self._rclpy = rclpy
        with _CONTEXT_LOCK:
            self._context = rclpy.get_default_context()
            if self._context.ok():
                raise RuntimeError("default ROS context already initialized; use a dedicated RPent process")
            try:
                rclpy.init(args=[], signal_handler_options=SignalHandlerOptions.NO)
            finally:
                # Also reclaim an init that raised after initializing its context.
                self._owns_context = self._context.ok()
        self._node = rclpy.create_node(
            "rpent_lynsense_readonly",
            enable_rosout=False,
            start_parameter_services=False,
            use_global_arguments=False,
        )
        qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
            history=HistoryPolicy.KEEP_LAST,
            depth=10,
        )
        self._node.create_subscription(JointState, self._joint_topic, self._on_joints, qos)
        self._node.create_subscription(RobotMsg, self._robot_topic, self._on_driver, qos)
        self._executor = SingleThreadedExecutor()
        self._executor.add_node(self._node)
        self._thread = threading.Thread(target=self._spin, name="rpent-lynsense-state", daemon=True)
        self._thread.start()

    def _on_joints(self, message: Any) -> None:
        with self._condition:
            if not self._stop.is_set():
                self._cache.update_joints(
                    message, time.monotonic(), datetime.now(timezone.utc).isoformat()
                )
                self._condition.notify_all()

    def _on_driver(self, message: Any) -> None:
        with self._condition:
            if not self._stop.is_set():
                self._cache.update_driver(time.monotonic())
                self._condition.notify_all()

    def _spin(self) -> None:
        try:
            while not self._stop.is_set():
                self._executor.spin_once(timeout_sec=0.1)
        except Exception as exc:
            with self._condition:
                if self._stop.is_set():
                    return
                self._runtime_failure = f"ROS executor failed: {exc}"
                self._failure = self._runtime_failure
                self._stop.set()
                self._condition.notify_all()
            # spin_once has unwound: this thread no longer uses ROS callbacks.
            try:
                self.close()
            except Exception:
                # close records the failure and retains resources for owner retry.
                return

    def connect(self) -> dict[str, str]:
        """Wait for both fresh samples, cleaning up any failed connection."""
        setup_error: BaseException | None = None
        with self._lifecycle:
            if self._closed or self._stop.is_set():
                raise RuntimeError("adapter is closed or stopping; create a new adapter")
            if self._started:
                current = self.get_snapshot()
                return {
                    "status": "ok" if current["status"] == "ok" else "failed",
                    "reason": current["reason"],
                }
            self._started = True
            try:
                self._start_ros()
            except BaseException as exc:
                setup_error = exc
        if setup_error is not None:
            self.close()
            if not isinstance(setup_error, Exception):
                raise setup_error
            return {"status": "failed", "reason": f"ROS connection failed: {setup_error}"}

        deadline = time.monotonic() + self._timeout
        try:
            with self._condition:
                while True:
                    if self._stop.is_set():
                        reason = self._runtime_failure or self._failure or "connection cancelled by close"
                        break
                    snapshot = self._cache.snapshot(time.monotonic())
                    if snapshot["status"] == "ok":
                        return {"status": "ok", "reason": snapshot["reason"]}
                    if snapshot["status"] == "failed":
                        reason = snapshot["reason"]
                        break
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        reason = f"state connection timed out: {snapshot['reason']}"
                        break
                    self._condition.wait(timeout=remaining)
        except BaseException:
            self.close()
            raise
        self.close()
        return {"status": "failed", "reason": reason}

    def get_snapshot(self) -> dict[str, Any]:
        """Read an immediate copied snapshot without spinning or doing I/O."""
        with self._condition:
            snapshot = self._cache.snapshot(time.monotonic())
            if self._runtime_failure:
                reason = self._runtime_failure
                if self._closed:
                    reason += "; state adapter resources released"
                elif self._failure and self._failure != self._runtime_failure:
                    reason += f"; {self._failure}"
                snapshot.update(status="failed", reason=reason)
            elif self._closed:
                snapshot.update(status="closed", reason="state adapter is closed")
            elif self._failure:
                snapshot.update(status="failed", reason=self._failure)
            elif self._stop.is_set():
                snapshot.update(status="failed", reason="state adapter is stopping")
            return snapshot

    def _acquire_lifecycle(self, deadline: float) -> None:
        if not self._lifecycle.acquire(timeout=max(0.0, deadline - time.monotonic())):
            raise RuntimeError("timed out waiting for ROS lifecycle operation to stop")

    def close(self) -> None:
        """Boundedly stop owned work; retained resources allow cleanup retry."""
        deadline = time.monotonic() + self._CLOSE_TIMEOUT_S
        self._stop.set()
        with self._condition:
            self._condition.notify_all()
        try:
            self._acquire_lifecycle(deadline)
            try:
                if self._executor is not None:
                    self._executor.wake()
                thread = self._thread
            finally:
                self._lifecycle.release()

            # Never hold the lifecycle lock while joining: the faulting worker
            # may itself need that lock to finish automatic resource cleanup.
            if thread is not None and thread is not threading.current_thread():
                # Thread.start can fail before a native thread exists.
                if thread.ident is not None:
                    thread.join(timeout=max(0.0, deadline - time.monotonic()))
                if thread.is_alive():
                    raise RuntimeError("ROS callback thread did not stop within close budget")

            self._acquire_lifecycle(deadline)
            try:
                if thread is not threading.current_thread():
                    self._thread = None
                if self._closed:
                    return
                if self._executor is not None:
                    if not self._executor.shutdown(timeout_sec=max(0.0, deadline - time.monotonic())):
                        raise RuntimeError("ROS executor did not shut down within close budget")
                    self._executor = None
                if self._node is not None:
                    self._node.destroy_node()
                    self._node = None
                if self._owns_context:
                    with _CONTEXT_LOCK:
                        # Keep the owned context reachable if shutdown raises.
                        self._rclpy.shutdown(context=self._context, uninstall_handlers=False)
                        if self._rclpy.get_default_context() is self._context:
                            self._rclpy.get_default_context(shutting_down=True)
                        self._owns_context = False
                with self._condition:
                    self._closed = True
            finally:
                self._lifecycle.release()
        except Exception as exc:
            with self._condition:
                self._failure = f"ROS cleanup failed: {exc}"
            raise
