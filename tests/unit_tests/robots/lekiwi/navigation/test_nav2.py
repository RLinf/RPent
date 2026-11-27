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

"""Exercise the ROS worker with fake messages and an asynchronous action server."""

import sys
import threading
import time
from concurrent.futures import Future, ThreadPoolExecutor
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock

import pytest

from robots.lekiwi.navigation import config, nav2


@pytest.fixture
def fake_ros(monkeypatch):
    published = []
    goals = []
    status = SimpleNamespace(STATUS_SUCCEEDED=4, STATUS_CANCELED=5)

    class Message:
        def __init__(self):
            self.header = SimpleNamespace(stamp=SimpleNamespace(sec=0, nanosec=0))
            self.pose = SimpleNamespace(
                position=SimpleNamespace(), orientation=SimpleNamespace()
            )

    class Node:
        def __init__(self, name):
            pass

        def create_publisher(self, *args):
            return SimpleNamespace(publish=published.append)

        def create_subscription(self, *args):
            pass

        def create_timer(self, interval, callback):
            self.timer_callback = callback

        def get_clock(self):
            return SimpleNamespace(
                now=lambda: SimpleNamespace(
                    nanoseconds=time.time_ns(), to_msg=lambda: None
                )
            )

        def destroy_node(self):
            pass

    class ActionClient:
        def __init__(self, *args):
            pass

        def send_goal_async(self, goal):
            future = Future()
            goals.append(future)
            return future

    modules = {
        "rclpy": {
            "init": lambda **kw: None,
            "shutdown": lambda: None,
            "create_node": Node,
        },
        "rclpy.node": {"Node": Node},
        "rclpy.action": {"ActionClient": ActionClient},
        "rclpy.signals": {"SignalHandlerOptions": SimpleNamespace(NO=0)},
        "action_msgs.msg": {"GoalStatus": status},
        "geometry_msgs.msg": {"PoseStamped": Message, "TwistStamped": Message},
        "nav2_msgs.action": {"NavigateToPose": SimpleNamespace(Goal=lambda **kw: kw)},
        "std_msgs.msg": {"String": Message},
        "tf2_ros": {
            "Buffer": lambda: None,
            "TransformException": RuntimeError,
            "TransformListener": lambda *args: None,
        },
    }
    for name, members in modules.items():
        module = ModuleType(name)
        module.__dict__.update(members)
        monkeypatch.setitem(sys.modules, name, module)
    monkeypatch.setattr(
        config, "localization_settings", lambda: {"localization_max_age": 1}
    )
    monkeypatch.setattr(nav2.signal, "signal", lambda *args: None)
    return published, goals, status


@pytest.mark.parametrize("accept_before_stop", [True, False])
@pytest.mark.parametrize("outage", ["none", "recover", "persistent", "deadline"])
def test_cancel_then_next_goal_on_same_worker(
    monkeypatch, fake_ros, accept_before_stop, outage
):
    published, goals, status = fake_ros

    serving = threading.Event()
    holder = {}

    def serve_rpc(self, **kwargs):
        holder["facade"] = self
        serving.set()
        self._shutdown_event.wait(10)

    monkeypatch.setattr(nav2.NavigationFacade, "serve", serve_rpc)

    def spin_once(ros_node, **kwargs):
        node = ros_node.timer_callback.__self__
        assert serving.wait(2)
        facade = holder["facade"]
        node.health = lambda: {"ready": True}
        try:
            with ThreadPoolExecutor(max_workers=1) as executor:

                def call(method, **params):
                    future = executor.submit(facade._rpc[method], **params)
                    deadline = time.monotonic() + 2
                    while node.requests.empty() and not future.done():
                        assert time.monotonic() < deadline
                        time.sleep(0.001)
                    node.tick()
                    return future.result(timeout=2)

                # Read-only RPCs also run on the ROS thread and see live state.
                assert call("navigation.state")["operation"]["state"] == "idle"
                first = call(
                    "navigation.navigate",
                    target=42,
                    goal={"xy": [1, 2], "yaw": 0, "node_id": 42},
                    timeout_s=60,
                )
                result = Future()
                handle = SimpleNamespace(
                    accepted=True,
                    get_result_async=lambda: result,
                    cancel_goal_async=Mock(),
                )
                if accept_before_stop:
                    goals[0].set_result(handle)
                if outage != "none":
                    node.health = lambda: {
                        "ready": False,
                        "reason": "localization_unavailable",
                    }
                    count = len(published)
                    node.tick()
                    assert len(published) > count, (
                        "loss of health must send zero velocity"
                    )
                    assert node.operation["state"] == "running"
                    assert node.cancel_reason is None
                    assert not handle.cancel_goal_async.called
                    if outage == "recover":
                        node.health = lambda: {"ready": True}
                        node.tick()
                        assert node.unavailable_since is None
                        assert node.cancel_reason is None
                    else:
                        if outage == "persistent":
                            node.unavailable_since -= 2
                        else:
                            node.deadline = time.monotonic() - 1
                        node.tick()
                        assert node.cancel_reason == (
                            "localization_unavailable"
                            if outage == "persistent"
                            else "Navigation goal timed out"
                        )
                assert call("navigation.cancel")["state"] == "running"
                assert (
                    node.cancel_reason
                    == {
                        "none": "cancelled",
                        "recover": "cancelled",
                        "persistent": "localization_unavailable",
                        "deadline": "Navigation goal timed out",
                    }[outage]
                )
                assert published
                if not accept_before_stop:
                    goals[0].set_result(handle)
                handle.cancel_goal_async.assert_called_once()
                count = len(published)
                node.forward(sys.modules["geometry_msgs.msg"].TwistStamped())
                assert len(published) == count
                with pytest.raises(RuntimeError, match="Another navigation goal"):
                    call(
                        "navigation.navigate",
                        target=43,
                        goal={"xy": [2, 3], "yaw": 0, "node_id": 43},
                        timeout_s=60,
                    )
                result.set_result(SimpleNamespace(status=status.STATUS_CANCELED))
                node.tick()
                expected = (
                    "failed" if outage in {"persistent", "deadline"} else "cancelled"
                )
                assert call("navigation.state")["operation"]["state"] == expected
                node.health = lambda: {"ready": True}
                second = call(
                    "navigation.navigate",
                    target=43,
                    goal={"xy": [2, 3], "yaw": 0, "node_id": 43},
                    timeout_s=60,
                )
                assert first["id"] != second["id"]
                assert second["state"] == "running"
                final = Future()
                goals[1].set_result(
                    SimpleNamespace(accepted=True, get_result_async=lambda: final)
                )
                final.set_result(SimpleNamespace(status=status.STATUS_SUCCEEDED))
                node.tick()
                assert call("navigation.state")["operation"]["state"] == "succeeded"
        finally:
            # Prevent the shutdown loop from entering this scripted spin again.
            node.operation["state"] = "cancelled"
            facade._shutdown_event.set()

    monkeypatch.setattr(sys.modules["rclpy"], "spin_once", spin_once, raising=False)
    nav2.serve(0)


@pytest.mark.parametrize("method", ["navigation.state", "navigation.navigate"])
def test_stalled_executor_expires_requests_without_executing_them(fake_ros, method):
    node = sys.modules["rclpy"].create_node("test_navigation")
    navigator = nav2.Navigator(node)
    navigator.health = Mock(return_value={"ready": True})
    navigator.state = Mock(return_value={"ready": True})
    navigator.begin = Mock()
    facade = nav2.NavigationFacade(navigator)

    # No ROS ticks: neither a status read nor a goal may use cached state or
    # wait indefinitely. An expired goal must not execute when ticks resume.
    with pytest.raises(TimeoutError, match="Navigation executor did not respond"):
        facade._rpc[method]()
    navigator.tick()
    navigator.state.assert_not_called()
    navigator.begin.assert_not_called()
    assert navigator.operation == {"state": "idle"}


def test_localization_limits_do_not_require_camera_calibration(tmp_path, monkeypatch):
    from robots.lekiwi.navigation import config

    path = tmp_path / "robot.yaml"
    path.write_text("localization:\n  localization_max_age: 1.5\n")
    monkeypatch.setattr(config, "get_robot_config_path", lambda: path)
    assert config.localization_settings() == {
        "localization_max_age": 1.5,
        "max_correction_translation": 0.5,
        "max_correction_yaw_deg": 30.0,
    }
