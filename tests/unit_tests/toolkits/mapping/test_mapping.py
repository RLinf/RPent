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


"""Check stereo preparation, timestamps and metric RGB-D conversion."""

import json
import logging

import numpy as np
import pytest
import yaml

from toolkits.mapping.prepare_stereo import prepare_stereo
from toolkits.mapping.raft_to_rgbd import convert_depth

cv2 = pytest.importorskip("cv2")


@pytest.fixture
def dataset(tmp_path):
    source = tmp_path / "input with spaces"
    for side, value in (("left", 60), ("right", 180)):
        (source / side).mkdir(parents=True)
        for index in range(3):
            image = np.full((48, 64, 3), value + index, dtype=np.uint8)
            assert cv2.imwrite(str(source / side / f"{index:06d}.png"), image)
    raw = {
        "image_size": [64, 48],
        "left": {"K": [50, 0, 32, 0, 50, 24, 0, 0, 1], "D": [0] * 5},
        "right": {"K": [50, 0, 32, 0, 50, 24, 0, 0, 1], "D": [0] * 5},
        "R": np.eye(3).ravel().tolist(),
        "T_m": [-0.12, 0, 0],
    }
    calibration = tmp_path / "stereo.yaml"
    calibration.write_text(yaml.safe_dump(raw))
    return source, calibration


def test_prepare_and_convert_official_flow(dataset, tmp_path):
    source, calibration = dataset
    for side in ("left", "right"):
        for index, stamp in enumerate(("0.2", "0", "0.12")):
            (source / side / f"{index:06d}.png").rename(source / side / f"{stamp}.png")
    prepared, flows, output = (
        tmp_path / name for name in ("prepared", "flows", "rgbd")
    )
    prepare_stereo(
        source=source,
        output=prepared,
        calibration=calibration,
        camera_pose=[0, 0, 0.5, 0, 0, 0],
        timestamps_from_names=True,
    )
    assert not (prepared / "local_transform.txt").exists()
    meta = json.loads((prepared / "stereo.json").read_text())
    assert meta["frames"] == ["1.0", "1.12", "1.2"]
    flows.mkdir()
    for name in meta["frames"]:
        # Official RAFT flow is negative disparity: 50 px * 0.12 m / 4 px = 1.5 m.
        flow = np.full((48, 64), -4, np.float32)
        flow[0, :5] = [0, 4, float("nan"), float("inf"), -0.01]
        np.save(flows / f"{name}.npy", flow)
    convert_depth(source=prepared, flow_dir=flows, output=output)
    assert not (output / "local_transform.txt").exists()
    for side in ("rgb", "depth"):
        assert {p.name for p in (output / side).iterdir()} == {
            "1.00.png",
            "1.12.png",
            "1.20.png",
        }
    depth = cv2.imread(str(output / "depth/1.00.png"), cv2.IMREAD_UNCHANGED)
    assert depth.dtype == np.uint16
    assert depth[0, :5].tolist() == [0, 0, 0, 0, 0]
    assert np.all(depth[1:] == 1500)
    assert (prepared / "camera.yaml").read_bytes() == (
        output / "camera.yaml"
    ).read_bytes()


@pytest.mark.parametrize("baseline_tilt", [0, 5])
def test_logged_mounting_pose(dataset, tmp_path, caplog, monkeypatch, baseline_tilt):
    source, calibration = dataset
    raw = yaml.safe_load(calibration.read_text())
    tilt = np.deg2rad(baseline_tilt)
    # Parallel cameras with a tilted baseline require an image-plane rotation.
    raw["T_m"] = [-0.12 * np.cos(tilt), -0.12 * np.sin(tilt), 0]
    raw["T_m"] = [float(value) for value in raw["T_m"]]
    calibration.write_text(yaml.safe_dump(raw))
    monkeypatch.setattr(logging.getLogger("rpent"), "propagate", True)
    with caplog.at_level(logging.INFO, logger="rpent"):
        prepare_stereo(
            source=source,
            output=tmp_path / "prepared",
            calibration=calibration,
            camera_pose=[0.2, -0.1, 0.5, 20, -10, 30],
            fps=5,
        )
    message = next(
        record.getMessage()
        for record in caplog.records
        if record.getMessage().startswith("RTAB-Map localTransform=")
    )
    pose = np.fromstring(message.split("=", 1)[1], sep=" ")
    assert pose[:3] == pytest.approx([0.2, -0.1, 0.5])
    # Rectification rotates about the viewing axis: mechanical roll changes,
    # while pitch, yaw and the optical centre remain fixed.
    assert pose[3:] == pytest.approx(np.deg2rad([20 + baseline_tilt, -10, 30]))
