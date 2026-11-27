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

"""Export recorded stereo pairs as RTAB-Map RGB-D inputs using RAFT-Stereo."""

from __future__ import annotations

import argparse
import configparser
import json
import math
import re
import sys
from pathlib import Path

import numpy as np
import yaml


def export_stereo(
    *,
    recording: Path,
    output: Path,
    robot_config: Path,
    raft_repo: Path,
    checkpoint: Path,
    step: int = 1,
    iterations: int = 32,
    device: str = "cuda:0",
) -> None:
    """Rectify recorded pairs, run upstream RAFT, and save metric RGB-D inputs."""
    left_paths = sorted((recording / "left").glob("*.png"))
    if not left_paths:
        raise ValueError("Recording contains no left images")
    for path in left_paths:
        if not re.fullmatch(r"\d{10}\.\d{9}\.png", path.name):
            raise ValueError(
                "Recording requires timestamp filenames: seconds.nanoseconds.png"
            )
        if not (recording / "right" / path.name).is_file():
            raise ValueError(f"Missing right image: {path.name}")

    import cv2
    import torch

    from robots.lekiwi.stereo import StereoCalibration, load_geometry

    # Upstream uses core.* imports; keep it in this dedicated offline process.
    sys.path.insert(0, str(raft_repo.resolve()))
    from core.raft_stereo import RAFTStereo
    from core.utils.utils import InputPadder

    metadata = json.loads((recording / "stereo.json").read_text())
    cal = StereoCalibration(
        recording / metadata["calibration"],
        size=tuple(metadata["eye_size"]),
    )
    projection = cal.rectifications[0][1]
    fx_baseline = abs(cal.rectifications[1][1][0, 3])
    args = argparse.Namespace(
        hidden_dims=[128, 128, 128],
        corr_levels=4,
        corr_radius=4,
        n_downsample=2,
        n_gru_layers=3,
        mixed_precision=False,
        corr_implementation="reg",
        shared_backbone=False,
        context_norm="batch",
        slow_fast_gru=False,
    )
    model = torch.nn.DataParallel(RAFTStereo(args))
    model.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=True))
    model = model.module.to(device).eval()
    output = output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    calibration = {
        "width": cal.size[0],
        "height": cal.size[1],
        "fx": float(projection[0, 0]),
        "fy": float(projection[1, 1]),
        "cx": float(projection[0, 2]),
        "cy": float(projection[1, 2]),
    }
    write_rtabmap_config(output, calibration, load_geometry(robot_config))
    with torch.inference_mode():
        for index, left_path in enumerate(left_paths):
            if index % step:
                continue
            rectified = [
                cv2.remap(
                    cv2.imread(str(recording / side / left_path.name)),
                    *maps,
                    cv2.INTER_LINEAR,
                )
                for side, maps in zip(("left", "right"), cal.maps)
            ]
            tensors = [
                torch.from_numpy(cv2.cvtColor(im, cv2.COLOR_BGR2RGB))
                .permute(2, 0, 1)[None]
                .float()
                .to(device)
                for im in rectified
            ]
            padder = InputPadder(tensors[0].shape, divis_by=32)
            left, right = padder.pad(*tensors)
            _, flow = model(left, right, iters=iterations, test_mode=True)
            disparity = -padder.unpad(flow).squeeze().cpu().numpy()
            depth = np.zeros(disparity.shape, dtype=np.float32)
            np.divide(fx_baseline, disparity, out=depth, where=disparity > 0)
            name = left_path.name
            if not cv2.imwrite(str(output / "rgb" / name), rectified[0]):
                raise OSError("Could not write rectified RGB frame")
            write_depth(output / "depth" / name, depth)


def write_depth(path: Path, meters: np.ndarray) -> None:
    """Save standard uint16 millimetre PNG; invalid or out-of-range depth is zero."""
    import cv2

    valid = np.isfinite(meters) & (meters > 0) & (meters <= 65.535)
    millimeters = np.rint(np.where(valid, meters, 0) * 1000).astype(np.uint16)
    if not cv2.imwrite(str(path), millimeters):
        raise RuntimeError("Depth PNG write failed")


def write_rtabmap_config(output: Path, calibration: dict, geometry: dict) -> None:
    """Write official camera YAML and INI from measured robot and camera metadata."""
    output = output.resolve()
    for folder in ("rgb", "depth"):
        (output / folder).mkdir(exist_ok=False)
    fx, fy, cx, cy = (calibration[k] for k in ("fx", "fy", "cx", "cy"))
    intrinsic = [fx, 0.0, cx, 0.0, fy, cy, 0.0, 0.0, 1.0]
    camera = {
        "camera_name": "zed",
        "image_width": calibration["width"],
        "image_height": calibration["height"],
        "camera_matrix": {"rows": 3, "cols": 3, "data": intrinsic},
        "distortion_model": "plumb_bob",
        "distortion_coefficients": {"rows": 1, "cols": 5, "data": [0.0] * 5},
        "rectification_matrix": {
            "rows": 3,
            "cols": 3,
            "data": np.eye(3).ravel().tolist(),
        },
        "projection_matrix": {
            "rows": 3,
            "cols": 4,
            "data": [fx, 0.0, cx, 0.0, 0.0, fy, cy, 0.0, 0.0, 0.0, 1.0, 0.0],
        },
    }
    (output / "zed.yaml").write_text(
        "%YAML:1.0\n---\n"
        + yaml.safe_dump(camera, sort_keys=False, default_flow_style=None)
    )
    # Official Camera adds optical rotation itself; this is mechanical RPY in radians.
    mount = [*geometry["mount_xyz_m"], *map(math.radians, geometry["mount_rpy_deg"])]
    config = configparser.ConfigParser(interpolation=None)
    config.optionxform = str
    core = {
        "Rtabmap/WorkingDirectory": str(output),
        "Kp/DetectorStrategy": "6",
        "Vis/FeatureType": "6",
        "Rtabmap/DetectionRate": "2",
        "Odom/ResetCountdown": "5",
        "Optimizer/Strategy": "2",
        "Rtabmap/StartNewMapOnLoopClosure": "true",
        "Reg/Force3DoF": "false",
        "RGBD/ForceOdom3DoF": "false",
        "RGBD/CreateOccupancyGrid": "true",
        "Grid/Sensor": "1",
        "Grid/3D": "false",
        "Grid/CellSize": "0.05",
        "Grid/RangeMin": "0.1",
        "Grid/RangeMax": "4.0",
        "Grid/NormalsSegmentation": "true",
        "Grid/NoiseFilteringRadius": "0.10",
        "Grid/NoiseFilteringMinNeighbors": "5",
        "Grid/MinGroundHeight": "-0.03",
        "Grid/MaxGroundHeight": "0.05",
        "Grid/MaxObstacleHeight": str(geometry["collision_height_m"]),
        "Grid/RayTracing": "true",
        "Grid/MapFrameProjection": "false",
    }
    config["Core"] = {key.replace("/", "\\"): value for key, value in core.items()}
    config["Camera"] = {
        "type": "0",
        "rgbd\\driver": "7",
        "imgRate": "0",
        "calibrationName": str(output / "zed.yaml"),
        "localTransform": " ".join(map(str, mount)),
        "RGBDImages\\path_rgb": str(output / "rgb"),
        "RGBDImages\\path_depth": str(output / "depth"),
        "RGBDImages\\scale": "1",
        "Images\\filenames_as_stamps": "true",
        "Images\\sync_stamps": "false",
    }
    with (output / "rtabmap.ini").open("w") as stream:
        config.write(stream)


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(
        description="Export rectified RGB and RAFT-Stereo depth from recorded PNG pairs."
    )
    parser.add_argument("--input", dest="recording", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--robot-config", type=Path, required=True)
    parser.add_argument("--raft-repo", type=Path, required=True)
    parser.add_argument(
        "--checkpoint",
        type=Path,
        required=True,
        help="Official raftstereo-middlebury.pth",
    )
    parser.add_argument("--step", type=int, default=1)
    parser.add_argument("--iterations", type=int, default=32)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args(argv)

    export_stereo(**vars(args))


if __name__ == "__main__":
    main()
