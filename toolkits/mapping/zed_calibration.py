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

"""Convert ZED factory calibration to RTAB-Map YAMLs for raw stereo images."""

from __future__ import annotations

import argparse
import configparser
import logging
from pathlib import Path

import numpy as np
import yaml

from rpent.utils.logging import get_logger

logger = get_logger(__name__)


def _matrix(value: np.ndarray) -> dict:
    return {
        "rows": value.shape[0],
        "cols": value.shape[1],
        "data": value.ravel().tolist(),
    }


def convert_calibration(
    *, calibration: Path, image_size: tuple[int, int], output: Path
) -> None:
    """Export raw intrinsics and stereo rectification without overwriting files.

    Args:
        calibration: Factory .conf file for the recorded ZED camera.
        image_size: Native per-eye width and height, in pixels.
        output: Directory for camera_left.yaml and camera_right.yaml.
    """
    import cv2

    modes = {(672, 376): "VGA", (1280, 720): "HD"}
    if image_size not in modes:
        raise ValueError("Use native per-eye dimensions 1280 720 or 672 376")
    mode = modes[image_size]
    cfg = configparser.ConfigParser()
    with calibration.open() as stream:
        cfg.read_file(stream)
    try:
        cameras = []
        for side in ("LEFT", "RIGHT"):
            c = cfg[f"{side}_CAM_{mode}"]
            k = np.array(
                [
                    [float(c["fx"]), 0.0, float(c["cx"])],
                    [0.0, float(c["fy"]), float(c["cy"])],
                    [0.0, 0.0, 1.0],
                ]
            )
            d = np.array([float(c[key]) for key in ("k1", "k2", "p1", "p2", "k3")])
            if (
                not np.isfinite(k).all()
                or not np.isfinite(d).all()
                or k[0, 0] <= 0
                or k[1, 1] <= 0
            ):
                raise ValueError(f"Invalid intrinsics in {side}_CAM_{mode}")
            cameras.append((k, d))
        stereo = cfg["STEREO"]
        rotation_vector = np.array(
            [float(stereo[f"{key}_{mode}"]) for key in ("RX", "CV", "RZ")]
        )
        translation = (
            np.array(
                [-float(stereo["Baseline"]), float(stereo["TY"]), float(stereo["TZ"])]
            )
            / 1000.0
        )
    except KeyError as exc:
        raise ValueError(f"Missing ZED calibration section or field: {exc}") from exc
    if (
        not np.isfinite(rotation_vector).all()
        or not np.isfinite(translation).all()
        or translation[0] >= 0
    ):
        raise ValueError("STEREO requires finite extrinsics and a positive Baseline")
    rotation, _ = cv2.Rodrigues(rotation_vector)
    (k1, d1), (k2, d2) = cameras
    r1, r2, p1, p2, *_ = cv2.stereoRectify(
        k1,
        d1,
        k2,
        d2,
        image_size,
        rotation,
        translation,
        flags=cv2.CALIB_ZERO_DISPARITY,
        alpha=0,
    )
    if not all(np.isfinite(m).all() for m in (r1, r2, p1, p2)):
        raise ValueError("Calibration did not produce finite rectification matrices")
    paths = [output / f"camera_{side}.yaml" for side in ("left", "right")]
    for path in paths:
        if path.exists():
            raise FileExistsError(f"Refusing to overwrite {path}")
    output.mkdir(parents=True, exist_ok=True)
    for path, (k, d), r, p in zip(paths, cameras, (r1, r2), (p1, p2)):
        model = {
            "camera_name": path.stem,
            "image_width": image_size[0],
            "image_height": image_size[1],
            "camera_matrix": _matrix(k),
            "distortion_model": "plumb_bob",
            "distortion_coefficients": _matrix(d.reshape(1, -1)),
            "rectification_matrix": _matrix(r),
            "projection_matrix": _matrix(p),
        }
        with path.open("x") as stream:
            stream.write("%YAML:1.0\n---\n")
            yaml.safe_dump(model, stream, sort_keys=False, default_flow_style=None)
    logger.info(
        "Wrote stereo calibration to %s; use StereoImages\\rectify=true", output
    )


def main(argv: list[str] | None = None) -> None:
    """Convert a ZED factory file for the selected native image dimensions."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--calibration", type=Path, required=True)
    parser.add_argument(
        "--image-size",
        type=int,
        nargs=2,
        metavar=("WIDTH", "HEIGHT"),
        required=True,
        help="Native per-eye pixels: 1280 720 (HD) or 672 376 (VGA)",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    try:
        convert_calibration(
            calibration=args.calibration,
            image_size=tuple(args.image_size),
            output=args.output,
        )
    except (OSError, ValueError, configparser.Error) as exc:
        parser.exit(1, f"ZED calibration conversion failed: {exc}\n")


if __name__ == "__main__":
    main()
