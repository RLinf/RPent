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
import hashlib
import importlib
import io
import json
from pathlib import Path

import pytest


@pytest.fixture
def collector_module(monkeypatch):
    cv2 = pytest.importorskip("cv2")
    if not hasattr(cv2, "aruco") or not hasattr(cv2.aruco, "CharucoDetector"):
        pytest.skip("Calibration requires OpenCV with the ChArUco detector")
    pytest.importorskip("scipy")
    monkeypatch.syspath_prepend(
        str(Path(__file__).resolve().parents[3] / "calibration_tools")
    )
    return importlib.import_module("base_handeye_collect")


@pytest.fixture
def session(collector_module, tmp_path, monkeypatch):
    config = collector_module.CaptureConfig(
        arm="left",
        camera_serial="offline-camera",
        camera_url="http://offline.invalid/frame",
        reader_command=("offline-reader",),
    )
    collector = collector_module.Collector(config, tmp_path / "session")
    state = {
        "O_T_EE": collector_module.np.eye(4).flatten(order="F").tolist(),
        "F_T_EE": collector_module.np.eye(4).flatten(order="F").tolist(),
        "dq": [0.0] * 7,
        "robot_mode": 1,
        "has_errors": False,
    }
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
    monkeypatch.setattr(collector_module.time, "sleep", lambda _: None)
    monkeypatch.setattr(collector_module.urllib.request, "urlopen", camera_response)
    return collector, state, camera


def snapshot(folder):
    return {
        str(path.relative_to(folder)): path.read_bytes()
        for path in folder.rglob("*")
        if path.is_file()
    }


def request_handler(module, collector, path, method="POST"):
    """Exercise actual HTTP routing without opening a listening socket."""
    handler_type = module.make_handler(collector)
    handler = handler_type.__new__(handler_type)
    handler.path = path
    responses = []
    handler.reply = lambda code, body, kind="application/json": responses.append(
        (code, body, kind)
    )
    handler.send_error = lambda code, *args: responses.append((code, b"", ""))
    if method == "GET":
        handler.do_GET()
    else:
        handler.do_POST()
    assert len(responses) == 1
    return responses[0]


def test_review_exclude_resume_and_continue_capture(collector_module, session):
    collector, state, _ = session
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
    assert snapshot(archived) == original


def test_reviewed_samples_require_a_new_calibration_result(collector_module, session):
    collector, state, _ = session
    for index in range(10):
        state["O_T_EE"][12] = index * 0.02
        collector.sample()
    root = collector.root
    pose = collector_module.np.eye(4).tolist()
    candidate = {
        "status": "candidate_requires_independent_validation",
        "arm": "left",
        "calibration_mode": "eye_to_hand",
        "sample_count": 10,
        "selected_method": "PARK",
        "camera_serial": collector.config.camera_serial,
        "T_left_base_camera": pose,
        "T_ee_board": pose,
    }
    report = {
        **candidate,
        "serial": collector.config.camera_serial,
        "source_hashes": {
            str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in root.glob("sample_*/sample.json")
        },
        "methods": {"PARK": {"T_left_base_camera": pose, "T_ee_board": pose}},
    }
    candidate_path = root / "candidate.json"
    candidate_path.write_text(json.dumps(candidate))
    (root / "quality_report.json").write_text(json.dumps(report))
    common = importlib.import_module("common")
    assert common.load_candidate(candidate_path) == candidate

    collector.delete_sample(5)

    with pytest.raises(ValueError, match="Training samples changed"):
        common.load_candidate(candidate_path)
    assert json.loads(candidate_path.read_text()) == candidate
