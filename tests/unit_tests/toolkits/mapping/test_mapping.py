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


"""Check metric depth conversion and exported ZED stereo calibration."""

import json
import subprocess
import sys

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
    meta = json.loads((prepared / "stereo.json").read_text())
    assert meta["frames"] == ["1.0", "1.12", "1.2"]
    flows.mkdir()
    for name in meta["frames"]:
        # Official RAFT flow is negative disparity: 50 px * 0.12 m / 4 px = 1.5 m.
        flow = np.full((48, 64), -4, np.float32)
        flow[0, :5] = [0, 4, float("nan"), float("inf"), -0.01]
        np.save(flows / f"{name}.npy", flow)
    convert_depth(source=prepared, flow_dir=flows, output=output)
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


@pytest.fixture
def zed_factory_calibration(tmp_path):
    calibration = tmp_path / "calibration.conf"
    sections = []
    for mode, focal in (("HD", 500), ("VGA", 250)):
        for side in ("LEFT", "RIGHT"):
            sections.append(
                f"[{side}_CAM_{mode}]\nfx={focal}\nfy={focal}\ncx=320\ncy=180\n"
                "k1=-0.01\nk2=0\np1=0\np2=0\nk3=0\n"
            )
    sections.append(
        "[STEREO]\nBaseline=120\nTY=0\nTZ=0\n"
        "RX_HD=0\nCV_HD=0\nRZ_HD=0\nRX_VGA=0\nCV_VGA=0\nRZ_VGA=0\n"
    )
    calibration.write_text("\n".join(sections))
    return calibration


@pytest.mark.parametrize("size, focal", [((1280, 720), 500), ((672, 376), 250)])
def test_zed_calibration_cli_preserves_raw_model_and_metric_baseline(
    zed_factory_calibration, tmp_path, size, focal
):
    output = tmp_path / "camera models"
    subprocess.run(
        [
            sys.executable,
            "-m",
            "toolkits.mapping.zed_calibration",
            "--calibration",
            str(zed_factory_calibration),
            "--image-size",
            *map(str, size),
            "--output",
            str(output),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    for side in ("left", "right"):
        storage = cv2.FileStorage(
            str(output / f"camera_{side}.yaml"), cv2.FILE_STORAGE_READ
        )
        try:
            assert storage.isOpened()
            assert storage.getNode("image_width").real() == size[0]
            assert storage.getNode("image_height").real() == size[1]
            k = storage.getNode("camera_matrix").getNode("data")
            d = storage.getNode("distortion_coefficients").getNode("data")
            p = storage.getNode("projection_matrix").getNode("data")
            assert k.at(0).real() == focal
            assert d.at(0).real() == pytest.approx(-0.01)
            baseline = -p.at(3).real() / p.at(0).real()
            assert baseline == pytest.approx(0.12 if side == "right" else 0)
        finally:
            storage.release()
