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

"""Offline tests for the in-process localization boundary."""

import sys
import threading
from types import SimpleNamespace

import numpy as np
import pytest

from robots.lekiwi import localization_server as server


@pytest.fixture
def native(monkeypatch):
    class Native:
        def __init__(self, database):
            self.database = database
            self.closed = 0
            self.calls = []

        def localize(self, *args):
            self.calls.append(args)
            return {"status": "localized", "T_map_camera": np.eye(4).tolist()}

        def goal(self, target):
            return {"status": "goal" if target == "home" else "unknown_goal"}

        def close(self):
            self.closed += 1

    monkeypatch.setitem(
        sys.modules, "lekiwi_localization", SimpleNamespace(Localizer=Native)
    )
    return Native


def test_image_is_passed_in_memory_and_close_is_forwarded(native, tmp_path):
    localizer = server.Localizer(tmp_path / "map.db")
    backend = localizer._native
    rgb = np.zeros((32, 32, 3), dtype=np.uint8)
    camera = {"width": 32, "height": 32, "params": [30, 30, 16, 16]}
    packet = {
        "stamp_ns": 123,
        "T_base_camera": np.eye(4).tolist(),
        "odometry": [0, 0, 0],
    }
    result = localizer.localize(rgb, camera, packet)
    assert backend.calls[0][0] is rgb
    assert backend.calls[0][1:] == (camera, 123, packet["T_base_camera"], [0, 0, 0])
    assert result["status"] == "localized"
    assert result["processing_ms"] >= 0
    assert localizer.goal("home")["status"] == "goal"
    with pytest.raises(ValueError, match="Unknown"):
        localizer.goal("missing")
    localizer.close()
    assert backend.closed == 1


def test_late_result_is_discarded_without_closing_instance(native, monkeypatch):
    localizer = server.Localizer("map.db", max_processing_ms=100)
    times = iter([1.0, 1.2, 2.0, 2.01])
    monkeypatch.setattr(server.time, "monotonic", lambda: next(times))
    with pytest.raises(TimeoutError, match="deadline"):
        localizer.goal("home")
    assert localizer._native.closed == 0
    assert localizer.goal("home")["status"] == "goal"
    localizer.close()


def test_native_exception_propagates_without_closing_instance(native, monkeypatch):
    localizer = server.Localizer("map.db")
    original = localizer._native.goal

    def fail(target):
        raise RuntimeError("native failure")

    monkeypatch.setattr(localizer._native, "goal", fail)
    with pytest.raises(RuntimeError, match="native failure"):
        localizer.goal("home")
    monkeypatch.setattr(localizer._native, "goal", original)
    assert localizer.goal("home")["status"] == "goal"
    localizer.close()


@pytest.mark.parametrize("budget", [0, -1, float("nan"), float("inf")])
def test_invalid_deadline(native, budget):
    with pytest.raises(ValueError, match="positive and finite"):
        server.Localizer("map.db", max_processing_ms=budget)


def test_rpc_serializes_localization_and_goal_requests():
    entered = threading.Event()
    release = threading.Event()
    goal_attempted = threading.Event()
    goal_entered = threading.Event()
    errors = []

    class Backend:
        def localize(self, packet, jpeg):
            entered.set()
            if not release.wait(2):
                raise RuntimeError("test did not release call")
            return {}

        def goal(self, target):
            goal_entered.set()
            return {"status": "goal"}

    backend = Backend()
    facade = server.LocalizationFacade(backend)
    # Bypass image decoding so this tests the actual RPC dispatch boundary.
    facade._rpc["localization.localize"] = backend.localize

    def dispatch(method, args):
        try:
            if method == "localization.goal":
                goal_attempted.set()
            facade._dispatch(method, args, {})
        except BaseException as error:
            errors.append(error)

    worker = threading.Thread(
        target=dispatch, args=("localization.localize", ({}, b""))
    )
    reader = threading.Thread(target=dispatch, args=("localization.goal", ("home",)))
    worker.start()
    try:
        assert entered.wait(2)
        reader.start()
        assert goal_attempted.wait(2)
        assert not goal_entered.wait(0.05)
    finally:
        release.set()
        worker.join(2)
        if reader.ident is not None:
            reader.join(2)
    assert not worker.is_alive()
    assert not reader.is_alive()
    assert not errors
    assert goal_entered.is_set()
