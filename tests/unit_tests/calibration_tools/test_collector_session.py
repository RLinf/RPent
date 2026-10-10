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
"""Offline regression coverage for interrupted and reviewed calibration sessions."""

from __future__ import annotations

import base64
import copy
import hashlib
import importlib
import io
import json
import sys
from pathlib import Path

import pytest


@pytest.fixture
def collector_module(monkeypatch):
    cv2 = pytest.importorskip("cv2")
    if not hasattr(cv2, "aruco") or not hasattr(cv2.aruco, "CharucoDetector"):
        pytest.skip("Calibration requires OpenCV with the ChArUco detector")
    pytest.importorskip("scipy")
    root = Path(__file__).resolve().parents[3]
    monkeypatch.syspath_prepend(str(root / "calibration_tools"))
    return importlib.import_module("base_handeye_collect")


@pytest.fixture
def config(collector_module):
    return collector_module.CaptureConfig(
        arm="left",
        camera_serial="offline-camera",
        camera_url="http://offline.invalid/frame",
        reader_command=("offline-reader",),
    )


@pytest.fixture
def record(collector_module, config):
    identity = collector_module.np.eye(4)
    pose = identity.copy()
    pose[0, 3] = 0.1
    target = identity.copy()
    target[2, 3] = 0.8
    state = {
        "O_T_EE": pose.flatten(order="F").tolist(),
        "F_T_EE": identity.flatten(order="F").tolist(),
        "dq": [0.0] * 7,
        "robot_mode": 1,
        "has_errors": False,
    }
    return {
        "index": 1,
        "arm": config.arm,
        "calibration_mode": "eye_to_hand",
        "board": config.board_metadata,
        "robot_before": state,
        "robot_after": copy.deepcopy(state),
        "T_left_base_ee": pose.tolist(),
        "T_camera_board": target.tolist(),
        "camera": {
            "serial": config.camera_serial,
            "width": 680,
            "height": 880,
            "K": [[800.0, 0.0, 340.0], [0.0, 800.0, 440.0], [0.0, 0.0, 1.0]],
            "distortion": [0.0] * 5,
            "distortion_model": "distortion.brown_conrady",
            "host_received_s": 1.0,
        },
        "corners": 20,
        "markers": 12,
        "charuco_ids": list(range(20)),
        "charuco_corners_px": [[float(i % 5), float(i // 5)] for i in range(20)],
        "reprojection_rms_px": 0.2,
        "drift_m": 0.0,
        "drift_rad": 0.0,
    }


def save_record(root, record, index):
    """Create a complete saved sample, preserving its serialized bytes for checks."""
    folder = root / f"sample_{index:03d}"
    folder.mkdir(parents=True)
    value = copy.deepcopy(record)
    value["index"] = index
    (folder / "sample.json").write_text(json.dumps(value), encoding="utf-8")
    (folder / "color.png").write_bytes(b"original-color-png")
    (folder / "annotated.png").write_bytes(b"original-annotated-png")
    return folder


def snapshot(folder):
    return {
        str(path.relative_to(folder)): path.read_bytes()
        for path in folder.rglob("*")
        if path.is_file()
    }


@pytest.fixture
def session_root(tmp_path, record):
    root = tmp_path / "session"
    save_record(root, record, 1)
    return root


@pytest.fixture
def collector(collector_module, config, session_root):
    return collector_module.Collector(config, session_root, resume=True)


def test_existing_session_requires_explicit_resume(
    collector_module, config, session_root
):
    original = snapshot(session_root)
    with pytest.raises((ValueError, FileExistsError)):
        collector_module.Collector(config, session_root)
    assert snapshot(session_root) == original


def test_resume_preserves_originals_and_continues_after_gaps(
    collector_module, config, record, tmp_path
):
    root = tmp_path / "session"
    save_record(root, record, 7)
    save_record(root, record, 2)
    original = snapshot(root)

    collector = collector_module.Collector(config, root, resume=True)

    assert [sample["index"] for sample in collector.samples] == [2, 7]
    assert collector.next_index == 8
    assert collector.status()["count"] == 2
    assert collector.status()["samples"] == [
        {"index": index, "corners": 20, "reprojection_rms_px": 0.2} for index in [2, 7]
    ]
    assert snapshot(root) == original


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("arm", "right"),
        ("calibration_mode", "eye_in_hand"),
        ("board", {"squares": [7, 8]}),
        ("index", 99),
    ],
)
def test_resume_rejects_incompatible_sample_metadata(
    collector_module, config, session_root, field, value
):
    folder = session_root / "sample_001"
    changed = json.loads((folder / "sample.json").read_text())
    changed[field] = value
    (folder / "sample.json").write_text(json.dumps(changed))
    original = snapshot(session_root)

    with pytest.raises(ValueError):
        collector_module.Collector(config, session_root, resume=True)

    assert snapshot(session_root) == original


@pytest.mark.parametrize("field", ["serial", "K", "F_T_EE"])
def test_resume_rejects_changed_camera_or_end_effector(
    collector_module, config, record, session_root, field
):
    changed = copy.deepcopy(record)
    if field == "serial":
        changed["camera"]["serial"] = "another-camera"
    elif field == "K":
        changed["camera"]["K"][0][0] += 10.0
    else:
        for key in ("robot_before", "robot_after"):
            changed[key]["F_T_EE"][12] = 0.05
    save_record(session_root, changed, 2)

    with pytest.raises(ValueError):
        collector_module.Collector(config, session_root, resume=True)


@pytest.mark.parametrize("missing", ["sample.json", "color.png", "annotated.png"])
def test_resume_rejects_incomplete_sample(
    collector_module, config, session_root, missing
):
    (session_root / "sample_001" / missing).unlink()

    with pytest.raises((ValueError, OSError)):
        collector_module.Collector(config, session_root, resume=True)


def test_delete_archives_original_files_and_retains_monotonic_ids(
    collector_module, config, record, tmp_path
):
    root = tmp_path / "session"
    folder = save_record(root, record, 7)
    original = snapshot(folder)
    collector = collector_module.Collector(config, root, resume=True)

    result = collector.delete_sample(7)

    archived = root / "excluded" / "sample_007"
    assert result == {
        "deleted": True,
        "index": 7,
        "count": 0,
        "archived": str(archived),
    }
    assert not folder.exists()
    assert snapshot(archived) == original
    assert collector.samples == []
    assert collector.next_index == 8
    resumed = collector_module.Collector(config, root, resume=True)
    assert resumed.samples == []
    assert resumed.next_index == 8


@pytest.mark.parametrize("index", [0, -1, True, "1", None, 99])
def test_delete_rejects_invalid_or_missing_ids_without_mutation(collector, index):
    original = snapshot(collector.root)

    with pytest.raises(ValueError):
        collector.delete_sample(index)

    assert snapshot(collector.root) == original
    assert [sample["index"] for sample in collector.samples] == [1]


def test_archive_conflict_never_overwrites_either_copy(collector, record):
    archived = save_record(collector.root / "excluded", record, 1)
    (archived / "color.png").write_bytes(b"previously-excluded-image")
    original = snapshot(collector.root)

    with pytest.raises((ValueError, FileExistsError)):
        collector.delete_sample(1)

    assert snapshot(collector.root) == original
    assert collector.status()["count"] == 1


def test_failed_archive_rename_preserves_active_sample(collector, monkeypatch):
    original = snapshot(collector.root)

    def fail_rename(path, target):
        raise OSError("simulated archive rename failure")

    monkeypatch.setattr(Path, "rename", fail_rename)
    with pytest.raises(OSError, match="simulated"):
        collector.delete_sample(1)

    assert snapshot(collector.root) == original
    assert collector.status()["count"] == 1
    assert collector.next_index == 2


def test_capture_after_resume_and_delete_never_reuses_saved_ids(
    collector_module, config, record, tmp_path, monkeypatch
):
    root = tmp_path / "session"
    save_record(root, record, 2)
    save_record(root / "excluded", record, 9)
    collector = collector_module.Collector(config, root, resume=True)
    collector.delete_sample(2)
    original = snapshot(root)

    state = copy.deepcopy(record["robot_before"])
    state["O_T_EE"][12] = 0.3
    monkeypatch.setattr(collector, "read_state", lambda: copy.deepcopy(state))
    monkeypatch.setattr(collector_module.time, "sleep", lambda _: None)
    board_image = collector.board.generateImage((680, 880), marginSize=40)
    success, encoded = collector_module.cv2.imencode(".png", board_image)
    assert success

    def camera_response(*args, **kwargs):
        camera = copy.deepcopy(record["camera"])
        camera["host_received_s"] = collector_module.time.time()
        camera["png_b64"] = base64.b64encode(encoded.tobytes()).decode()
        return io.BytesIO(json.dumps(camera).encode())

    monkeypatch.setattr(collector_module.urllib.request, "urlopen", camera_response)

    result = collector.sample()

    assert result["accepted"] is True
    assert result["count"] == 1
    assert Path(result["saved"]).name == "sample_010"
    assert collector.samples[0]["index"] == 10
    assert collector.next_index == 11
    assert all(
        (root / name).read_bytes() == content for name, content in original.items()
    )
    resumed = collector_module.Collector(config, root, resume=True)
    assert [sample["index"] for sample in resumed.samples] == [10]
    assert resumed.next_index == 11


def request_handler(collector_module, collector, path, method):
    """Exercise the real routing methods without binding a network socket."""
    handler_type = collector_module.make_handler(collector)
    handler = handler_type.__new__(handler_type)
    handler.path = path
    responses = []
    handler.reply = lambda code, body, kind="application/json": responses.append(
        (code, body, kind)
    )
    handler.send_error = lambda code, *args: responses.append((code, b"", ""))
    getattr(handler, f"do_{method}")()
    assert len(responses) == 1
    return responses[0]


def test_http_review_delete_and_capture_share_mutation_lock(
    collector_module, collector
):
    code, body, kind = request_handler(
        collector_module, collector, "/samples/1/annotated.png", "GET"
    )
    assert (code, body, kind) == (200, b"original-annotated-png", "image/png")

    with collector.lock:
        for path in ("/sample", "/samples/1/delete"):
            code, _, _ = request_handler(collector_module, collector, path, "POST")
            assert code == 409
    assert collector.status()["count"] == 1

    code, body, _ = request_handler(
        collector_module, collector, "/samples/1/delete", "POST"
    )
    assert code == 200
    assert json.loads(body)["deleted"] is True
    assert collector.status()["count"] == 0
    assert not collector.lock.locked()
    code, _, _ = request_handler(
        collector_module, collector, "/samples/1/annotated.png", "GET"
    )
    assert code == 404


@pytest.mark.parametrize("index", ["0", "-1", "wrong", "99"])
def test_http_delete_invalid_id_retains_session(collector_module, collector, index):
    code, _, _ = request_handler(
        collector_module, collector, f"/samples/{index}/delete", "POST"
    )

    assert code in (404, 422)
    assert collector.status()["count"] == 1
    assert not collector.lock.locked()


@pytest.mark.parametrize("restart", [False, True])
@pytest.mark.parametrize("changed_field", ["K", "F_T_EE"])
def test_deleting_all_samples_retains_camera_and_frame_baseline(
    collector_module, collector, config, record, monkeypatch, restart, changed_field
):
    root = collector.root
    collector.delete_sample(1)
    if restart:
        collector = collector_module.Collector(config, root, resume=True)
    original = snapshot(root)
    state = copy.deepcopy(record["robot_before"])
    camera = copy.deepcopy(record["camera"])
    if changed_field == "K":
        camera["K"][0][0] += 20.0
        expected_error = "Camera K"
    else:
        state["F_T_EE"][12] = 0.05
        expected_error = "末端坐标定义与先前样本不同"
    monkeypatch.setattr(collector, "read_state", lambda: copy.deepcopy(state))
    monkeypatch.setattr(collector_module.time, "sleep", lambda _: None)

    def camera_response(*args, **kwargs):
        camera["host_received_s"] = collector_module.time.time()
        return io.BytesIO(json.dumps(camera).encode())

    monkeypatch.setattr(collector_module.urllib.request, "urlopen", camera_response)

    with pytest.raises(ValueError, match=expected_error):
        collector.sample()

    assert collector.samples == []
    assert collector.next_index == 2
    assert snapshot(root) == original


def test_deleting_sample_invalidates_previously_solved_candidate(
    collector_module, config, record, tmp_path
):
    root = tmp_path / "session"
    for index in range(1, 11):
        save_record(root, record, index)
    pose = collector_module.np.eye(4).tolist()
    candidate = {
        "status": "candidate_requires_independent_validation",
        "arm": "left",
        "calibration_mode": "eye_to_hand",
        "sample_count": 10,
        "selected_method": "PARK",
        "camera_serial": config.camera_serial,
        "T_left_base_camera": pose,
        "T_ee_board": pose,
    }
    report = {
        **candidate,
        "serial": config.camera_serial,
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

    collector_module.Collector(config, root, resume=True).delete_sample(5)

    with pytest.raises(ValueError, match="Training samples changed"):
        common.load_candidate(candidate_path)
    assert json.loads(candidate_path.read_text()) == candidate


@pytest.mark.parametrize("resume_without_output", [False, True])
def test_cli_requires_explicit_existing_session_selection(
    collector_module,
    config,
    session_root,
    monkeypatch,
    capsys,
    resume_without_output,
):
    original = snapshot(session_root)
    arguments = [
        "base_handeye_collect.py",
        "--arm",
        config.arm,
        "--camera-serial",
        config.camera_serial,
        "--camera-url",
        config.camera_url,
        "--reader",
        "offline-reader",
        "--robot-ip",
        "192.0.2.1",
    ]
    if resume_without_output:
        arguments.append("--resume")
        message = "--resume requires --output"
    else:
        arguments.extend(["--output", str(session_root)])
        message = "already exists"
    monkeypatch.setattr(sys, "argv", arguments)
    monkeypatch.setattr(
        collector_module,
        "serve",
        lambda *args: pytest.fail("Must reject before serving"),
    )

    with pytest.raises(SystemExit) as error:
        collector_module.main()

    assert error.value.code == 2
    assert message in capsys.readouterr().err
    assert snapshot(session_root) == original
