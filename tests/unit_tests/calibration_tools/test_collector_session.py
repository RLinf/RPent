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
"""Offline functional checks for reviewed and resumed calibration sessions."""

from __future__ import annotations

import base64
import copy
import importlib
import io
import json
from pathlib import Path
from types import SimpleNamespace

import pytest


@pytest.fixture
def session(monkeypatch, tmp_path):
    cv2 = pytest.importorskip("cv2")
    if not hasattr(cv2, "aruco") or not hasattr(cv2.aruco, "CharucoDetector"):
        pytest.skip("Calibration requires OpenCV with the ChArUco detector")
    pytest.importorskip("scipy")
    monkeypatch.syspath_prepend(
        str(Path(__file__).resolve().parents[3] / "calibration_tools")
    )
    collector_module = importlib.import_module("base_handeye_collect")
    config = collector_module.CaptureConfig(
        arm="left",
        camera_serial="offline-camera",
        camera_url="http://offline.invalid/frame",
        reader_command=("offline-reader",),
    )
    collector = collector_module.Collector(config, tmp_path / "session")
    state = {
        "O_T_EE": collector_module.np.eye(4).flatten(order="F").tolist(),
        "dq": [0.0] * 7,
        "robot_mode": 1,
        "has_errors": False,
    }
    state["F_T_EE"] = state["O_T_EE"].copy()
    image = collector.board.generateImage((680, 880), marginSize=40)
    success, encoded = collector_module.cv2.imencode(".png", image)
    assert success
    camera = {
        "serial": config.camera_serial,
        "width": 680,
        "height": 880,
        "K": [[800.0, 0.0, 340.0], [0.0, 800.0, 440.0], [0.0, 0.0, 1.0]],
        "distortion": [0.0] * 5,
        "distortion_model": "distortion.brown_conrady",
        "png_b64": base64.b64encode(encoded.tobytes()).decode(),
    }

    def camera_response(*args, **kwargs):
        return io.BytesIO(
            json.dumps(
                {**camera, "host_received_s": collector_module.time.time()}
            ).encode()
        )

    monkeypatch.setattr(
        collector_module.Collector, "read_state", lambda self: copy.deepcopy(state)
    )
    clock = [1000.0]

    def advance_clock(seconds):
        clock[0] += seconds

    monkeypatch.setattr(
        collector_module,
        "time",
        SimpleNamespace(time=lambda: clock[0], sleep=advance_clock),
    )
    monkeypatch.setattr(collector_module.urllib.request, "urlopen", camera_response)
    return collector_module, collector, state


def snapshot(folder):
    return {
        str(path.relative_to(folder)): path.read_bytes()
        for path in folder.rglob("*")
        if path.is_file()
    }


def request_handler(module, collector, path, method="POST"):
    """Exercise actual HTTP routing without opening a listening socket."""
    handler = object.__new__(module.make_handler(collector))
    handler.path = path
    responses = []
    handler.reply = lambda code, body, kind="application/json": responses.append(
        (code, body, kind)
    )
    getattr(handler, f"do_{method}")()
    assert len(responses) == 1
    return responses[0]


def test_review_exclude_resume_and_continue_capture(session):
    collector_module, collector, state = session
    code, page, kind = request_handler(collector_module, collector, "/", "GET")
    assert (code, kind) == (200, "text/html; charset=utf-8")
    assert b"Fix the board to the end effector" in page
    code, body, _ = request_handler(collector_module, collector, "/sample")
    assert code == 200 and json.loads(body)["accepted"]
    original = snapshot(collector.root / "sample_001")
    code, image, kind = request_handler(
        collector_module, collector, "/samples/1/annotated.png", "GET"
    )
    assert (code, image, kind) == (200, original["annotated.png"], "image/png")
    with collector.lock:
        for path in ("/sample", "/samples/1/delete"):
            assert request_handler(collector_module, collector, path)[0] == 409
    code, body, _ = request_handler(collector_module, collector, "/samples/1/delete")
    assert code == 200 and json.loads(body)["deleted"]
    archived = collector.root / "excluded" / "sample_001"
    assert snapshot(archived) == original

    resumed = collector_module.Collector(collector.config, collector.root, resume=True)
    assert resumed.status()["count"] == 0
    state["O_T_EE"][12] = 0.1
    code, body, _ = request_handler(collector_module, resumed, "/sample")
    result = json.loads(body)
    assert code == 200 and result["accepted"]
    assert Path(result["saved"]).name == "sample_002"
    final = collector_module.Collector(collector.config, collector.root, resume=True)
    assert [sample["index"] for sample in final.status()["samples"]] == [2]
    collector = final
    for index in range(1, 10):
        state["O_T_EE"][12] = 0.1 + index * 0.02
        collector.sample()
    root = collector.root
    solver = importlib.import_module("solve_base_handeye")
    assert len(solver.load_samples(root, "left")[1]) == 10
    sample_path = root / "sample_005" / "sample.json"
    changed = json.loads(sample_path.read_text())
    changed["T_left_base_ee"][0][3] += 0.1
    sample_path.write_text(json.dumps(changed))
    with pytest.raises(ValueError, match="stored pose disagrees"):
        solver.load_samples(root, "left")
    with pytest.raises(ValueError, match="stored pose disagrees"):
        collector_module.Collector(collector.config, root, resume=True)
