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

"""Opt-in CPU checks against the companion RLinf camera implementation."""

import os
import threading
import time
from types import SimpleNamespace

import numpy as np
import pytest

from robots.lekiwi.env_server import LeKiwiEnvFacade
from rpent.utils.rpc.http_rpc import HttpRpcClient, HttpRpcServer


@pytest.mark.skipif(
    os.environ.get("RPENT_RUN_LEKIWI_INTEGRATION") != "1",
    reason="set RPENT_RUN_LEKIWI_INTEGRATION=1 with companion RLinf installed",
)
def test_stereo_rpc_recovers_after_a_missing_frame():
    import queue
    from types import MethodType
    from unittest.mock import Mock

    from rlinf.robotics.parts.cameras.base import BaseCamera

    from robots.lekiwi.env_server import LeKiwiEnvBackend

    # Exercise the real RLinf queue reader without starting its hardware thread.
    camera = SimpleNamespace(
        _frame_capturing_start=True,
        _latest_frame=None,
        _frame_queue=queue.Queue(),
        name="offline",
        reopen=Mock(),
    )
    camera.get_frame = MethodType(BaseCamera.get_frame, camera)
    backend = LeKiwiEnvBackend.__new__(LeKiwiEnvBackend)
    backend.camera = camera
    facade = LeKiwiEnvFacade(backend)
    server = HttpRpcServer(("127.0.0.1", 0), facade._dispatch)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    client = HttpRpcClient(f"http://127.0.0.1:{server.server_port}")
    try:
        assert client.call("env.get_stereo_frame", timeout_s=3) is None
        camera.reopen.assert_called_once()
        stamp = time.monotonic()
        frame = np.full((2, 4, 3), 42, np.uint8)
        camera._frame_queue.put((stamp, frame))
        recovered = client.call("env.get_stereo_frame", timeout_s=3)
        assert recovered["source_stamp"] == stamp
        np.testing.assert_array_equal(recovered["frame"], frame)
        assert recovered["clock_domain"] == "host_monotonic"
    finally:
        client.close()
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)
