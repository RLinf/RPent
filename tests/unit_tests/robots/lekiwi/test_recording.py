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

"""Offline checks for timestamped stereo recording."""

import json
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import pytest

from robots.lekiwi.recording import StereoRecording


def test_recording_preserves_pair_pixels_and_capture_timestamp(tmp_path, monkeypatch):
    from robots.lekiwi import recording

    monkeypatch.setattr(recording.time, "time_ns", lambda: 1_700_000_000_000_000_000)
    monkeypatch.setattr(recording.time, "monotonic_ns", lambda: 10_000_000_000)
    monkeypatch.setattr(recording.time, "monotonic", lambda: 11.0)
    calibration = tmp_path / "zed.conf"
    calibration.write_text("synthetic calibration")
    image = np.zeros((2, 8, 3), dtype=np.uint8)
    image[:, :4] = 31
    image[:, 4:] = 127

    def get_frame(*, timeout, max_age, return_timestamp):
        assert return_timestamp
        capture.ending.set()
        return 10.25, image

    camera = SimpleNamespace(
        camera_info=SimpleNamespace(resolution=(8, 2)), get_frame=get_frame
    )
    capture = StereoRecording(
        camera,
        {
            "root": str(tmp_path / "recordings"),
            "calibration": str(calibration),
            "serial": "12345",
            "sample_fps": 5,
            "maximum_recording_seconds": 10,
        },
        "test",
    )
    capture.run()
    assert capture.state["phase"] == "recorded"
    assert capture.state["frames"] == 1
    directory = Path(capture.state["path"])
    name = "1700000000.250000000.png"
    np.testing.assert_array_equal(
        cv2.imread(str(directory / "left" / name)), image[:, :4]
    )
    np.testing.assert_array_equal(
        cv2.imread(str(directory / "right" / name)), image[:, 4:]
    )
    metadata = json.loads((directory / "stereo.json").read_text())
    assert metadata["eye_size"] == [4, 2]
    assert (directory / metadata["calibration"]).read_text() == calibration.read_text()


@pytest.mark.parametrize(
    "name, error",
    [
        ("000000.png", "timestamp filenames"),
        ("1700000000.250000000.png", "Missing right image"),
    ],
)
def test_export_rejects_invalid_pairs_before_loading_model(tmp_path, name, error):
    from robots.lekiwi.mapping import export_stereo

    (tmp_path / "left").mkdir()
    (tmp_path / "left" / name).touch()
    with pytest.raises(ValueError, match=error):
        export_stereo(
            recording=tmp_path,
            output=tmp_path / "output",
            robot_config=tmp_path / "robot.yaml",
            raft_repo=tmp_path / "raft",
            checkpoint=tmp_path / "weights.pth",
        )
