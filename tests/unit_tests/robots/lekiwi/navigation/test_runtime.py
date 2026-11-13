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

"""Offline checks for persistent navigation and the local stop gate."""

import time
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from robots.lekiwi.control import EnvControl
from robots.lekiwi.navigation.runtime import NavigationRuntime


@pytest.fixture
def runtime(monkeypatch):
    raw = Mock(config=SimpleNamespace(step_frequency=20))
    control = EnvControl(None, raw, max_linear=1, max_angular=1, command_timeout=0.3)
    control.switch_mode("navigation")
    runtime = NavigationRuntime({"robot_config": "/path/to/robot.yaml"}, control)
    runtime.daemon = Mock()
    runtime.daemon.poll.return_value = None
    client = Mock()
    operation = {"id": "first", "state": "running"}

    def call(method, **kwargs):
        if method == "navigation.state":
            return {"ready": True, "operation": dict(operation)}
        if method == "navigation.cancel":
            operation["state"] = "cancelled"
            return dict(operation)
        if method == "navigation.navigate":
            operation.update(
                id="next", state="running", target=kwargs["kwargs"]["target"]
            )
            return dict(operation)
        raise AssertionError(method)

    client.call.side_effect = call
    runtime.client = client
    monkeypatch.setattr(
        "robots.lekiwi.runtime_config.load_mapping",
        lambda _: {"localization": {"endpoint": "http://server.example:18125"}},
    )
    cloud = Mock()
    cloud.call.return_value = {"node_id": 42, "xy": [1, 2], "yaw": 0}
    monkeypatch.setattr(
        "robots.lekiwi.navigation.runtime.make_rpc_client", lambda _: cloud
    )
    return runtime, cloud


def send_velocity(control):
    return control.set_velocity(
        control.generation, [0.1, 0, 0], time.time() + control.command_timeout
    )


def test_cancel_keeps_service_and_next_goal_can_move(runtime):
    runtime, _ = runtime
    control = runtime.control
    control.arm_navigation(control.stop_revision)
    assert send_velocity(control)["accepted"]
    daemon, client = runtime.daemon, runtime.client
    generation = control.generation

    assert runtime.cancel()["state"] == "cancelled"
    assert not send_velocity(control)["accepted"]
    assert control._target == [0, 0, 0]
    assert control.mode == "navigation"
    assert control.generation == generation
    assert runtime.state()["session_active"]
    daemon.stop.assert_not_called()
    client.close.assert_not_called()

    assert runtime.navigate_to(42)["state"] == "running"
    assert send_velocity(control)["accepted"]
    assert runtime.daemon is daemon

    runtime.stop()
    daemon.stop.assert_called_once()
    client.close.assert_called_once()
    assert control.mode == "manual"
    assert not runtime.state()["session_active"]


def test_cancel_rpc_failure_keeps_wheels_blocked(runtime):
    runtime, _ = runtime
    runtime.control.arm_navigation(runtime.control.stop_revision)
    runtime.client.call.side_effect = TimeoutError("cancel unavailable")
    with pytest.raises(TimeoutError):
        runtime.cancel()
    assert not send_velocity(runtime.control)["accepted"]
    runtime.daemon.stop.assert_not_called()


def test_stop_during_goal_resolution_cannot_rearm_motion(runtime):
    runtime, cloud = runtime

    def resolve(*args, **kwargs):
        runtime.control.stop()
        return {"node_id": 42, "xy": [1, 2], "yaw": 0}

    cloud.call.side_effect = resolve
    with pytest.raises(RuntimeError, match="interrupted by stop"):
        runtime.navigate_to(42)
    assert not send_velocity(runtime.control)["accepted"]


def test_cancel_before_service_start_does_not_launch_anything():
    control = Mock()
    runtime = NavigationRuntime({}, control)
    assert runtime.cancel() == {"state": "idle"}
    control.stop.assert_called_once()
    assert runtime.daemon is None


def test_delayed_command_from_before_stop_is_rejected_after_new_goal(runtime):
    runtime, _ = runtime
    control = runtime.control
    old_expiry = time.time() + control.command_timeout
    runtime.cancel()
    runtime.navigate_to(42)
    assert not control.set_velocity(control.generation, [0.1, 0, 0], old_expiry)[
        "accepted"
    ]
    assert send_velocity(control)["accepted"]


@pytest.mark.parametrize(
    "error", [TimeoutError("status delayed"), RuntimeError("status unavailable")]
)
def test_status_rpc_failure_does_not_revoke_active_control(runtime, error):
    runtime, _ = runtime
    control = runtime.control
    runtime.navigate_to(42)
    assert send_velocity(control)["accepted"]
    healthy_call = runtime.client.call.side_effect
    runtime.client.call.side_effect = error
    unavailable = runtime.state()
    assert not unavailable["ready"]
    assert unavailable["session_active"]
    assert unavailable["operation"]["state"] == "running"
    assert send_velocity(control)["accepted"]

    runtime.client.call.side_effect = healthy_call
    recovered = runtime.state()
    assert recovered["ready"]
    assert recovered["operation"]["state"] == "running"
    assert send_velocity(control)["accepted"]


def test_exited_navigation_process_still_revokes_control(runtime):
    runtime, _ = runtime
    runtime.navigate_to(42)
    assert send_velocity(runtime.control)["accepted"]
    runtime.daemon.poll.return_value = 1
    state = runtime.state()
    assert not state["ready"]
    assert not state["session_active"]
    assert state["operation"]["state"] == "failed"
    assert not send_velocity(runtime.control)["accepted"]
    assert runtime.control._target == [0, 0, 0]


def test_runtime_starts_launch_module_directly(runtime, monkeypatch, tmp_path):
    from robots.lekiwi.navigation import runtime as runtime_module

    runtime, _ = runtime
    daemon, client = runtime.daemon, runtime.client
    runtime.daemon = runtime.client = None
    runtime.endpoint = "http://127.0.0.1:8120"
    runtime.config.update(
        robot_config=str(tmp_path / "robot.yaml"),
        map_yaml=str(tmp_path / "map.yaml"),
        ros_domain_id=7,
    )
    factory = Mock(return_value=daemon)
    monkeypatch.setattr(runtime_module, "ProcessDaemon", factory)
    monkeypatch.setattr(runtime_module, "make_rpc_client", lambda endpoint: client)
    wait = Mock()
    monkeypatch.setattr(runtime_module, "wait_for_ready", wait)
    assert runtime.start()["ready"]
    wait.assert_called_once_with(client, timeout_s=120, daemon=daemon)
    command = factory.call_args.args[1]
    assert command[command.index("-m") + 1] == "robots.lekiwi.navigation.launch"
    assert (
        command[command.index("--robot-config") + 1] == runtime.config["robot_config"]
    )
    assert "--hardware" in command
    assert "--parent-watch" in command
