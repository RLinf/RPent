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

"""Timestamp handling at the local camera-to-ROS boundary."""

from types import SimpleNamespace

import numpy as np
import pytest

from robots.lekiwi.navigation import bridge


def test_camera_bridge_discards_delayed_frame_and_accepts_next(monkeypatch):
    monkeypatch.setattr(bridge.time, "monotonic", lambda: 10)
    packet = {
        "source_stamp": 8.0,
        "clock_domain": "host_monotonic",
        "frame": np.zeros((376, 1344, 3), np.uint8),
    }
    message = SimpleNamespace(header=SimpleNamespace(stamp=SimpleNamespace()))
    assert (
        bridge.frame_to_message(
            packet, message, offset_ns=100000000000, after=7, max_age=1
        )
        == 7
    )
    packet["source_stamp"] = 9.5
    assert (
        bridge.frame_to_message(
            packet, message, offset_ns=100000000000, after=7, max_age=1
        )
        == 9.5
    )
    assert message.header.stamp.sec == 109
    assert message.header.stamp.nanosec == 500000000
    assert (
        bridge.frame_to_message(
            packet, message, offset_ns=100000000000, after=9.5, max_age=1
        )
        == 9.5
    )
    packet["clock_domain"] = "remote_monotonic"
    with pytest.raises(ValueError, match="local Env"):
        bridge.frame_to_message(packet, message, offset_ns=0, after=9.5, max_age=1)


@pytest.mark.parametrize("server_error", [False, True])
def test_camera_worker_retries_missing_data_but_surfaces_server_faults(
    monkeypatch, server_error
):
    import sys
    import time
    from types import ModuleType
    from unittest.mock import Mock

    from robots.lekiwi.navigation import config
    from rpent.utils.rpc import RpcError

    published = []
    running = [True]
    logger = Mock()

    class Image:
        def __init__(self):
            self.header = SimpleNamespace(stamp=SimpleNamespace())

    class Transform(Image):
        def __init__(self):
            super().__init__()
            self.transform = SimpleNamespace(
                translation=SimpleNamespace(), rotation=SimpleNamespace()
            )

    def publish(message):
        published.append(message)
        running[0] = False

    node = SimpleNamespace(
        get_clock=lambda: SimpleNamespace(
            now=lambda: SimpleNamespace(to_msg=lambda: None)
        ),
        create_publisher=lambda *args: SimpleNamespace(publish=publish),
        get_logger=lambda: logger,
        destroy_node=lambda: None,
    )
    modules = {
        "rclpy": {
            "init": lambda **kw: None,
            "create_node": lambda name: node,
            "ok": lambda: running[0],
            "shutdown": lambda: None,
        },
        "rclpy.qos": {"qos_profile_sensor_data": None},
        "rclpy.signals": {"SignalHandlerOptions": SimpleNamespace(NO=0)},
        "sensor_msgs.msg": {"Image": Image},
        "geometry_msgs.msg": {"TransformStamped": Transform},
        "tf2_ros": {
            "StaticTransformBroadcaster": lambda node: SimpleNamespace(
                sendTransform=lambda msg: None
            )
        },
    }
    for name, members in modules.items():
        module = ModuleType(name)
        module.__dict__.update(members)
        monkeypatch.setitem(sys.modules, name, module)
    monkeypatch.setattr(
        config,
        "perception_profile",
        lambda: {
            "camera_frame": "zed_left_optical_frame",
            "mount_xyz_m": [0, 0, 0],
            "mount_rpy_deg": [0, 0, 0],
            "maximum_age_seconds": 1,
        },
    )
    monkeypatch.setattr(bridge.signal, "signal", lambda *args: None)
    client = Mock()
    responses = iter(
        [
            None,
            RpcError(
                "env.get_stereo_frame",
                "unavailable",
                traceback="device failed" if server_error else None,
            ),
            "frame",
        ]
    )

    def call(method, **kwargs):
        assert method == "env.get_stereo_frame"
        assert kwargs["timeout_s"] == 3
        response = next(responses)
        if isinstance(response, Exception):
            raise response
        if response is None:
            return None
        return {
            "source_stamp": time.monotonic(),
            "clock_domain": "host_monotonic",
            "frame": np.zeros((376, 1344, 3), np.uint8),
        }

    client.call.side_effect = call
    monkeypatch.setattr(bridge, "make_rpc_client", lambda endpoint: client)
    if server_error:
        with pytest.raises(RpcError, match="unavailable"):
            bridge.publish_camera(endpoint="http://127.0.0.1:8120")
        assert not published
    else:
        bridge.publish_camera(endpoint="http://127.0.0.1:8120")
        assert len(published) == 1
        logger.warning.assert_called_once()
        assert published[0].width == 1344
    client.close.assert_called_once()
