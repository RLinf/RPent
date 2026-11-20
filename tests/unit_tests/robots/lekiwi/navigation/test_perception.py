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

"""Run the stereo worker against ROS message fakes without opening hardware."""

import sys
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest

from robots.lekiwi import stereo
from robots.lekiwi.navigation import config, perception


@pytest.mark.parametrize("processing_delay", [0.0, 1.0])
def test_stereo_uses_recent_buffered_frame_and_preserves_its_stamp(
    monkeypatch, processing_delay
):
    now = [10.0]
    published = []
    callbacks = []
    calls = []
    frame = SimpleNamespace(
        header=SimpleNamespace(stamp=SimpleNamespace(sec=9, nanosec=800000000)),
        width=2,
        height=1,
        step=6,
        encoding="bgr8",
        data=bytes(6),
    )
    running = [True]

    class Node:
        def create_publisher(self, *args):
            return SimpleNamespace(publish=published.append)

        def create_subscription(self, kind, topic, callback, qos):
            callbacks.append(callback)

        def destroy_node(self):
            pass

    def spin_once(*args, **kwargs):
        calls.append(1)
        assert len(calls) == 1, (
            "The worker discarded a recent frame captured before its iteration"
        )
        callbacks[0](frame)

    def compute(self, image):
        assert image.shape == (1, 2, 3)
        now[0] += processing_delay
        running[0] = False
        return np.ones((100, 3), np.float32), {
            "depth_valid_fraction": 1.0,
            "point_count": 100,
        }

    modules = {
        "rclpy": {
            "init": lambda **kw: None,
            "create_node": lambda name: Node(),
            "ok": lambda: running[0],
            "spin_once": spin_once,
            "shutdown": lambda: None,
        },
        "rclpy.qos": {"qos_profile_sensor_data": None},
        "sensor_msgs.msg": {"Image": object, "PointCloud2": object},
        "sensor_msgs_py.point_cloud2": {
            "create_cloud_xyz32": lambda header, points: header
        },
        "std_msgs.msg": {"Header": lambda **kw: SimpleNamespace(**kw)},
    }
    for name, members in modules.items():
        module = ModuleType(name)
        module.__dict__.update(members)
        monkeypatch.setitem(sys.modules, name, module)
    monkeypatch.setattr(
        config,
        "stereo_config",
        lambda: {"maximum_age_seconds": 0.5, "minimum_depth_coverage": 0.2},
    )
    monkeypatch.setattr(stereo, "StereoCalibration", lambda: None)
    monkeypatch.setattr(perception.LocalStereo, "__init__", lambda self, cal, cfg: None)
    monkeypatch.setattr(perception.LocalStereo, "compute", compute)
    monkeypatch.setattr(perception.time, "time_ns", lambda: round(now[0] * 1e9))
    monkeypatch.setattr(perception.signal, "signal", lambda *args: None)
    perception.run_stereo()
    assert len(published) == (1 if processing_delay == 0 else 0)
    if published:
        assert published[0].stamp is frame.header.stamp
