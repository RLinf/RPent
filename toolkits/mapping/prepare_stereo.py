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

"""Rectify stereo images or side-by-side video for the official RAFT-Stereo demo."""

from __future__ import annotations

import argparse
import itertools
import json
import logging
import math
from collections.abc import Iterator
from contextlib import closing
from decimal import Decimal, InvalidOperation
from pathlib import Path

import numpy as np
import yaml

from rpent.utils.logging import get_logger

logger = get_logger(__name__)


IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".tif", ".tiff"}


def stereo_frames(
    source: Path, *, fps: float | None = None, timestamps_from_names: bool = False
) -> Iterator[tuple[int, np.ndarray, np.ndarray]]:
    """Yield nanosecond timestamps and BGR pairs from a directory or SBS video.

    Directory inputs contain left/ and right/ with identical filenames. With
    timestamps_from_names, stems are decimal seconds; otherwise fps is required
    and filenames are sorted lexicographically. Video timing uses its declared
    constant frame rate, optionally overridden by fps.
    """
    import cv2

    source = source.expanduser().resolve(strict=True)
    if fps is not None and (not math.isfinite(fps) or not 0 < fps <= 1000):
        raise ValueError("fps must be finite and in (0, 1000]")
    if timestamps_from_names and fps is not None:
        raise ValueError("Choose filename timestamps or fps, not both")
    if source.is_dir():
        for stamp, left_path, right_path in _image_pairs(
            source, fps, timestamps_from_names
        ):
            images = [
                cv2.imread(str(path), cv2.IMREAD_COLOR)
                for path in (left_path, right_path)
            ]
            if any(image is None for image in images):
                raise ValueError(f"Cannot decode stereo pair: {left_path.name}")
            yield stamp, images[0], images[1]
    else:
        if timestamps_from_names:
            raise ValueError(
                "Filename timestamps are only supported for image directories"
            )
        # A resolved regular file prevents OpenCV from interpreting input as a
        # device index, stream URL or image-sequence pattern.
        if not source.is_file() or source.suffix.lower() not in {
            ".mp4",
            ".avi",
            ".mkv",
            ".mov",
        }:
            raise ValueError("Expected a local MP4, AVI, MKV or MOV stereo video")
        capture = cv2.VideoCapture(str(source))
        try:
            if not capture.isOpened():
                raise ValueError(f"Cannot open video: {source}")
            rate = fps if fps is not None else capture.get(cv2.CAP_PROP_FPS)
            if not math.isfinite(rate) or not 0 < rate <= 1000:
                raise ValueError("Video has no valid frame rate; supply --fps")
            expected = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
            count = 0
            while True:
                ok, image = capture.read()
                if not ok:
                    break
                if image.shape[1] % 2:
                    raise ValueError("Side-by-side video width must be even")
                middle = image.shape[1] // 2
                yield round(count * 1e9 / rate), image[:, :middle], image[:, middle:]
                count += 1
            if count == 0 or (expected > 0 and count != expected):
                raise ValueError(
                    f"Video decoding incomplete: read {count} of {expected} frames"
                )
        finally:
            capture.release()


def _image_pairs(
    source: Path,
    fps: float | None,
    timestamps_from_names: bool,
) -> Iterator[tuple[int, Path, Path]]:
    pairs = []
    for side in ("left", "right"):
        folder = source / side
        if not folder.is_dir():
            raise ValueError(f"Missing image directory: {folder}")
        pairs.append(
            {
                p.name: p
                for p in folder.iterdir()
                if p.suffix.lower() in IMAGE_SUFFIXES and p.is_file()
            }
        )
    if not pairs[0] or pairs[0].keys() != pairs[1].keys():
        raise ValueError(
            "left/ and right/ must contain the same nonempty set of image filenames"
        )
    if not timestamps_from_names and fps is None:
        raise ValueError("Image sequences require --fps or --timestamps-from-names")
    if timestamps_from_names:
        names = sorted(pairs[0], key=_timestamp)
        stamps = [_timestamp(name) for name in names]
    else:
        names = sorted(pairs[0])
        stamps = [round(i * 1e9 / fps) for i in range(len(names))]
    if any(b <= a for a, b in zip(stamps, stamps[1:])):
        raise ValueError("Image timestamps must be distinct and increasing")
    for name, stamp in zip(names, stamps):
        yield stamp, pairs[0][name], pairs[1][name]


def _timestamp(name: str) -> int:
    try:
        seconds = Decimal(Path(name).stem)
        if not seconds.is_finite() or not 0 <= seconds < Decimal("1e10"):
            raise ValueError("Timestamp outside supported range")
        nanoseconds = seconds * 1_000_000_000
        if nanoseconds != nanoseconds.to_integral_value():
            raise ValueError("Timestamp precision exceeds nanoseconds")
        return int(nanoseconds)
    except (InvalidOperation, ValueError) as exc:
        raise ValueError(
            f"Expected a nonnegative seconds.nanoseconds filename: {name}"
        ) from exc


class StereoCalibration:
    """Rectify a calibrated pinhole stereo pair, independent of camera brand."""

    def __init__(self, path: Path, size: tuple[int, int]) -> None:
        import cv2

        self.size = size
        raw = yaml.safe_load(path.read_text())
        if not isinstance(raw, dict) or tuple(raw.get("image_size", ())) != size:
            raise ValueError(
                f"Calibration image_size must match native per-eye dimensions {size}"
            )
        cameras = []
        for side in ("left", "right"):
            k, d = _intrinsics(raw[side])
            cameras.append((k, d))
        rotation = np.asarray(raw["R"], dtype=float).reshape(3, 3)
        translation = np.asarray(raw["T_m"], dtype=float).reshape(3)
        if (
            not np.isfinite(rotation).all()
            or not np.isfinite(translation).all()
            or not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-5)
            or not np.isclose(np.linalg.det(rotation), 1, atol=1e-5)
            or np.linalg.norm(translation) <= 0
        ):
            raise ValueError(
                "R must be a rotation and T_m a nonzero translation in metres"
            )
        k1, d1 = cameras[0]
        k2, d2 = cameras[1]
        r1, r2, p1, p2, *_ = cv2.stereoRectify(
            k1,
            d1,
            k2,
            d2,
            size,
            rotation,
            translation,
            flags=cv2.CALIB_ZERO_DISPARITY,
            alpha=0,
        )
        if (
            not np.isfinite(p1).all()
            or p1[0, 0] <= 0
            or p2[0, 3] >= 0
            or not np.isclose(p2[1, 3], 0)
        ):
            raise ValueError(
                "RAFT-Stereo requires horizontal stereo with the right camera to the right of the left camera"
            )
        self.projection = p1
        self.left_rotation = r1
        self.fx_baseline = float(-p2[0, 3])
        self.maps = [
            cv2.initUndistortRectifyMap(k, d, r, p, size, cv2.CV_32FC1)
            for (k, d), r, p in zip(cameras, (r1, r2), (p1, p2))
        ]

    def rectify(
        self, left: np.ndarray, right: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """Rectify same-sized native BGR images; never silently resize calibration."""
        import cv2

        images = (left, right)
        if any(im.shape != (self.size[1], self.size[0], 3) for im in images):
            raise ValueError(
                "Stereo frame dimensions changed or disagree with calibration"
            )
        return tuple(
            cv2.remap(im, *maps, cv2.INTER_LINEAR)
            for im, maps in zip(images, self.maps)
        )


def _intrinsics(raw: dict) -> tuple[np.ndarray, np.ndarray]:
    k = np.asarray(raw["K"], dtype=float).reshape(3, 3)
    d = np.asarray(raw["D"], dtype=float).reshape(-1)
    if (
        not np.isfinite(k).all()
        or not np.isfinite(d).all()
        or k[0, 0] <= 0
        or k[1, 1] <= 0
        or not np.allclose(k[2], [0, 0, 1])
        or not np.allclose([k[0, 1], k[1, 0]], 0)
        or d.size not in (4, 5, 8, 12, 14)
    ):
        raise ValueError("Invalid pinhole intrinsics/distortion")
    return k, d


def _matrix(value: np.ndarray) -> dict:
    return {
        "rows": value.shape[0],
        "cols": value.shape[1],
        "data": value.ravel().tolist(),
    }


def prepare_stereo(
    *,
    source: Path,
    output: Path,
    calibration: Path,
    camera_pose: list[float],
    fps: float | None = None,
    timestamps_from_names: bool = False,
    step: int = 1,
) -> None:
    """Write rectified pairs and log the mounting pose for RTAB-Map.

    Args:
        source: Stereo image directory or side-by-side video.
        output: New directory for rectified pairs and calibration.
        calibration: Stereo calibration YAML matching the input images.
        camera_pose: Measured left camera mechanical pose in the robot frame:
            x/y/z in metres, then roll/pitch/yaw in degrees.
        fps: Sampling rate for numbered images or override for video timing.
        timestamps_from_names: Read image timestamps from filename stems.
        step: Keep one pair every this many input frames.
    """
    import cv2
    from scipy.spatial.transform import Rotation

    if step < 1 or len(camera_pose) != 6 or not np.isfinite(camera_pose).all():
        raise ValueError("Use a positive step and six finite camera pose values")
    with closing(
        stereo_frames(source, fps=fps, timestamps_from_names=timestamps_from_names)
    ) as frames:
        first = next(frames)
        cal = StereoCalibration(calibration, (first[1].shape[1], first[1].shape[0]))
        if output.resolve().is_relative_to(source.resolve()):
            raise ValueError("Output must be outside the input directory")
        output.mkdir(parents=True, exist_ok=False)
        # RTAB-Map adds optical rotation; compensate the left-eye rectification.
        optical = np.array([[0, 0, 1], [-1, 0, 0], [0, -1, 0]])
        mount = Rotation.from_euler("xyz", camera_pose[3:], degrees=True).as_matrix()
        corrected = mount @ optical @ cal.left_rotation.T @ optical.T
        angles = Rotation.from_matrix(corrected).as_euler("xyz")
        logger.info(
            "RTAB-Map localTransform=%s",
            " ".join(map(str, [*camera_pose[:3], *angles])),
        )
        k = cal.projection[:, :3]
        camera = {
            "camera_name": "camera",
            "image_width": cal.size[0],
            "image_height": cal.size[1],
            "camera_matrix": _matrix(k),
            "distortion_model": "plumb_bob",
            "distortion_coefficients": _matrix(np.zeros((1, 5))),
            "rectification_matrix": _matrix(np.eye(3)),
            "projection_matrix": _matrix(cal.projection),
        }
        (output / "camera.yaml").write_text(
            "%YAML:1.0\n---\n"
            + yaml.safe_dump(camera, sort_keys=False, default_flow_style=None)
        )
        names = []
        for index, (stamp, left, right) in enumerate(itertools.chain([first], frames)):
            if index % step:
                continue
            left, right = cal.rectify(left, right)
            # Start at one second: RTAB-Map treats a zero timestamp specially.
            stamp = stamp - first[0] + 1_000_000_000
            name = f"{stamp // 10**9}.{stamp % 10**9:09d}".rstrip("0")
            if name.endswith("."):
                name += "0"
            pair = output / "pairs" / name
            pair.mkdir(parents=True)
            for filename, image in (("im0.png", left), ("im1.png", right)):
                if not cv2.imwrite(str(pair / filename), image):
                    raise OSError(f"Cannot write {pair / filename}")
            names.append(name)
        (output / "stereo.json").write_text(
            json.dumps(
                {
                    "frames": names,
                    "image_size": cal.size,
                    "fx_baseline": cal.fx_baseline,
                    "source_first_timestamp_ns": first[0],
                },
                indent=2,
            )
            + "\n"
        )
    logger.info("Prepared %d stereo pairs in %s", len(names), output)


def main(argv: list[str] | None = None) -> None:
    """Prepare rectified stereo files for the official RAFT-Stereo demo."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", dest="source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--calibration", type=Path, required=True)
    parser.add_argument(
        "--camera-pose",
        type=float,
        nargs=6,
        required=True,
        help=(
            "Measured left camera mounting pose relative to the robot: "
            "x y z in metres, mechanical roll pitch yaw in degrees"
        ),
    )
    timing = parser.add_mutually_exclusive_group()
    timing.add_argument("--fps", type=float)
    timing.add_argument("--timestamps-from-names", action="store_true")
    parser.add_argument("--step", type=int, default=1)
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    try:
        prepare_stereo(**vars(args))
    except (OSError, ValueError) as exc:
        parser.exit(1, f"Stereo preparation failed: {exc}\n")


if __name__ == "__main__":
    main()
