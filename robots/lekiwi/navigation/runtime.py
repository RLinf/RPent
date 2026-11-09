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

"""Own one ROS launch process and forward navigation requests over RPent RPC."""

from __future__ import annotations

import sys
import threading
from pathlib import Path

from rpent.utils.daemon import ProcessDaemon, pick_free_port
from rpent.utils.rpc import make_rpc_client
from rpent.utils.rpc.client_utils import wait_for_ready


class NavigationRuntime:
    def __init__(self, config: dict, control):
        self.config, self.control = config, control
        self.lock = threading.RLock()
        self.daemon = self.client = None
        self.endpoint = self.localization_config = None
        self.operation = {"state": "idle"}

    def start(self) -> dict:
        with self.lock:
            if self.daemon is not None and self.daemon.poll() is None:
                return self.state()
            if self.endpoint is None:
                raise RuntimeError("Env RPC endpoint is not bound")
            self.stop()
            self.localization_config = None
            port = pick_free_port()
            command = [
                str(self.config.get("ros_python") or sys.executable),
                "-B",
                "-m",
                "robots.lekiwi.navigation.launch",
                "--robot-config",
                self.config["robot_config"],
                "--endpoint",
                self.endpoint,
                "--map",
                self.config["map_yaml"],
                "--rpc-port",
                str(port),
                "--parent-watch",
            ]
            if not self.control.simulated:
                command.append("--hardware")
            log = Path(self.config["robot_config"]).parent / "navigation.log"
            self.daemon = ProcessDaemon(
                "lekiwi_navigation",
                command,
                log_path=str(log),
                env_overrides={
                    "ROS_DOMAIN_ID": str(self.config["ros_domain_id"]),
                    "ROS_LOCALHOST_ONLY": "1",
                    "RMW_IMPLEMENTATION": "rmw_cyclonedds_cpp",
                },
            )
            self.client = make_rpc_client(f"http://127.0.0.1:{port}")
            try:
                self.control.switch_mode("navigation")
                self.operation = {"state": "idle"}
                self.daemon.start()
                wait_for_ready(self.client, timeout_s=120, daemon=self.daemon)
            except BaseException:
                self.stop()
                raise
            return self.state()

    def state(self) -> dict:
        with self.lock:
            if self.daemon is None:
                return {
                    "ready": False,
                    "session_active": False,
                    "operation": dict(self.operation),
                }
            session_active = self.daemon.poll() is None
            if session_active:
                try:
                    result = self.client.call("navigation.state", timeout_s=1)
                except Exception as exc:
                    reason = str(exc)
                else:
                    self.operation = dict(result["operation"])
                    return dict(result, session_active=True)
            else:
                reason = "Navigation launch exited"
                self.control.stop()
                if self.operation["state"] == "running":
                    self.operation.update(state="failed", error=reason)
            return {
                "ready": False,
                "session_active": session_active,
                "reason": reason,
                "operation": dict(self.operation),
            }

    def _localization_call(self, method: str, **kwargs) -> dict:
        from robots.lekiwi.runtime_config import localization_service_config

        if self.localization_config is None:
            self.localization_config = localization_service_config(
                self.config["robot_config"]
            )
        client = make_rpc_client(self.localization_config["endpoint"])
        try:
            return client.call(
                "localization." + method,
                kwargs=kwargs,
                timeout_s=5,
            )
        finally:
            client.close()

    def list_map_nodes(self, *, label=None) -> dict:
        return self._localization_call("list_map_nodes", label=label)

    def inspect_map_node(self, node_id: int) -> dict:
        return self._localization_call("inspect_map_node", node_id=node_id)

    def view_map(self, *, center_node_id=None, radius_m=None) -> dict:
        from robots.lekiwi.map_view import render_map

        page = self._localization_call(
            "list_map_nodes",
            limit=100,
            center_node_id=center_node_id,
            radius_m=radius_m,
        )
        return render_map(
            self.config["map_yaml"], page, self.state(), radius_m=radius_m
        )

    def navigate_to(self, target: int | str, *, timeout_s: float = 60) -> dict:
        # Resolve the goal before taking the lock so state and stop stay responsive.
        stop_revision = self.control.stop_revision
        if not self.state()["ready"]:
            raise RuntimeError("Navigation is not ready")
        goal = self._localization_call("goal", target=target)
        with self.lock:
            if self.client is None:
                raise RuntimeError("Navigation is not ready")
            self.operation = self.client.call(
                "navigation.navigate",
                kwargs={
                    "target": target,
                    "goal": goal,
                    "timeout_s": timeout_s,
                },
                timeout_s=5,
            )
            self.control.arm_navigation(stop_revision)
            return dict(self.operation)

    def cancel(self) -> dict:
        """Stop motion and cancel the goal while keeping the ROS graph alive."""
        self.control.stop()
        with self.lock:
            if self.client is not None:
                self.operation = self.client.call("navigation.cancel", timeout_s=5)
            return dict(self.operation)

    def stop(self) -> dict:
        # Revoke the old navigation generation before shutting down ROS.
        self.control.switch_mode("stopping")
        with self.lock:
            try:
                if self.daemon is not None:
                    self.daemon.stop()
            finally:
                if self.client is not None:
                    self.client.close()
                self.client = self.daemon = None
                if self.operation["state"] == "running":
                    self.operation["state"] = "cancelled"
                self.control.switch_mode("manual")
        return {"stopped": True}

    def close(self) -> None:
        self.stop()
