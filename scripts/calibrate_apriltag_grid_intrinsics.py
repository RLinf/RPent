#!/usr/bin/env python3
"""Calibrate a ROS monocular camera from an AprilTag grid image stream."""

from __future__ import annotations

import argparse
import math
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np


def _image_msg_to_bgr(msg: Any) -> np.ndarray:
    encoding = msg.encoding.lower()
    if encoding in {"bgr8", "rgb8"}:
        image = np.frombuffer(msg.data, dtype=np.uint8).reshape(
            int(msg.height), int(msg.step) // 3, 3
        )[:, : int(msg.width), :]
        if encoding == "rgb8":
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        return image.copy()
    if encoding in {"mono8", "8uc1"}:
        gray = np.frombuffer(msg.data, dtype=np.uint8).reshape(
            int(msg.height), int(msg.step)
        )[:, : int(msg.width)]
        return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    raise ValueError(f"unsupported image encoding: {msg.encoding!r}")


def _ros_camera_yaml(
    *,
    camera_name: str,
    width: int,
    height: int,
    camera_matrix: np.ndarray,
    dist_coeffs: np.ndarray,
) -> str:
    k = np.asarray(camera_matrix, dtype=float).reshape(3, 3)
    d = np.asarray(dist_coeffs, dtype=float).reshape(-1)
    if d.size < 5:
        d = np.pad(d, (0, 5 - d.size))
    d = d[:5]
    r = np.eye(3, dtype=float)
    p = np.array(
        [
            [k[0, 0], 0.0, k[0, 2], 0.0],
            [0.0, k[1, 1], k[1, 2], 0.0],
            [0.0, 0.0, 1.0, 0.0],
        ],
        dtype=float,
    )

    def fmt(values: np.ndarray) -> str:
        return "[" + ", ".join(f"{float(v):.12g}" for v in values.reshape(-1)) + "]"

    return "\n".join(
        [
            f"image_width: {int(width)}",
            f"image_height: {int(height)}",
            f"camera_name: {camera_name}",
            "camera_matrix:",
            "  rows: 3",
            "  cols: 3",
            f"  data: {fmt(k)}",
            "distortion_model: plumb_bob",
            "distortion_coefficients:",
            "  rows: 1",
            "  cols: 5",
            f"  data: {fmt(d)}",
            "rectification_matrix:",
            "  rows: 3",
            "  cols: 3",
            f"  data: {fmt(r)}",
            "projection_matrix:",
            "  rows: 3",
            "  cols: 4",
            f"  data: {fmt(p)}",
            "",
        ]
    )


def _detector_parameters() -> Any:
    params = cv2.aruco.DetectorParameters_create()
    # Conservative defaults for 1280x1280 USB video. These fields exist in
    # OpenCV 4.2; guard newer AprilTag fields for portability.
    params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
    params.adaptiveThreshWinSizeMin = 5
    params.adaptiveThreshWinSizeMax = 45
    params.adaptiveThreshWinSizeStep = 5
    if hasattr(params, "aprilTagQuadDecimate"):
        params.aprilTagQuadDecimate = 1.0
    if hasattr(params, "aprilTagMinClusterPixels"):
        params.aprilTagMinClusterPixels = 5
    return params


def _calibrate(
    *,
    all_corners: list[np.ndarray],
    all_ids: list[np.ndarray],
    marker_counter: list[int],
    board: Any,
    image_size: tuple[int, int],
) -> tuple[float, np.ndarray, np.ndarray]:
    ids = np.concatenate(all_ids, axis=0).astype(np.int32)
    rms, camera_matrix, dist_coeffs, _rvecs, _tvecs = cv2.aruco.calibrateCameraAruco(
        all_corners,
        ids,
        np.asarray(marker_counter, dtype=np.int32),
        board,
        image_size,
        None,
        None,
    )
    return float(rms), camera_matrix, dist_coeffs


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Calibrate camera intrinsics from a 6x6 AprilTag 36h11 grid."
    )
    parser.add_argument("--image-topic", required=True)
    parser.add_argument("--camera-name", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--markers-x", type=int, default=6)
    parser.add_argument("--markers-y", type=int, default=6)
    parser.add_argument("--tag-size", type=float, default=0.0352)
    parser.add_argument("--tag-spacing", type=float, default=0.01056)
    parser.add_argument("--first-id", type=int, default=0)
    parser.add_argument("--max-samples", type=int, default=45)
    parser.add_argument("--min-samples", type=int, default=20)
    parser.add_argument("--min-markers", type=int, default=10)
    parser.add_argument("--sample-period", type=float, default=0.7)
    parser.add_argument("--display", action="store_true")
    args = parser.parse_args()

    import rospy
    from sensor_msgs.msg import Image

    if not hasattr(cv2, "aruco") or not hasattr(cv2.aruco, "DICT_APRILTAG_36h11"):
        raise RuntimeError("OpenCV aruco AprilTag support is unavailable.")

    dictionary = cv2.aruco.Dictionary_get(cv2.aruco.DICT_APRILTAG_36h11)
    board = cv2.aruco.GridBoard_create(
        args.markers_x,
        args.markers_y,
        args.tag_size,
        args.tag_spacing,
        dictionary,
        args.first_id,
    )
    params = _detector_parameters()

    latest: dict[str, Any] = {}

    def callback(msg: Image) -> None:
        latest["msg"] = msg

    rospy.init_node("apriltag_grid_intrinsics_calibrator", anonymous=True)
    rospy.Subscriber(args.image_topic, Image, callback, queue_size=1)

    all_corners: list[np.ndarray] = []
    all_ids: list[np.ndarray] = []
    marker_counter: list[int] = []
    image_size: tuple[int, int] | None = None
    last_sample_time = 0.0
    last_sample_center: np.ndarray | None = None

    print(
        "Collecting AprilTag grid samples: "
        f"{args.markers_x}x{args.markers_y}, tag={args.tag_size}m, "
        f"spacing={args.tag_spacing}m, ids {args.first_id}-"
        f"{args.first_id + args.markers_x * args.markers_y - 1}."
    )
    print("Move the board/camera through varied positions and rotations.")
    print("Press 'q' in the display window to finish early when --display is used.")

    rate = rospy.Rate(30)
    while not rospy.is_shutdown() and len(marker_counter) < args.max_samples:
        msg = latest.get("msg")
        if msg is None:
            rate.sleep()
            continue

        try:
            image = _image_msg_to_bgr(msg)
        except ValueError as exc:
            rospy.logerr(str(exc))
            return 2

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        corners, ids, _rejected = cv2.aruco.detectMarkers(gray, dictionary, parameters=params)
        if image_size is None:
            image_size = (int(image.shape[1]), int(image.shape[0]))

        marker_count = 0 if ids is None else int(len(ids))
        accepted = False
        if ids is not None and marker_count >= args.min_markers:
            now = time.monotonic()
            centers = np.concatenate(corners, axis=0).mean(axis=1)
            center = centers.mean(axis=0)
            center_shift = (
                math.inf
                if last_sample_center is None
                else float(np.linalg.norm(center - last_sample_center))
            )
            enough_time = now - last_sample_time >= args.sample_period
            enough_motion = center_shift >= 12.0 or last_sample_center is None
            if enough_time and enough_motion:
                all_corners.extend(corners)
                all_ids.append(ids.copy())
                marker_counter.append(marker_count)
                last_sample_time = now
                last_sample_center = center
                accepted = True
                print(
                    f"sample {len(marker_counter):02d}/{args.max_samples}: "
                    f"{marker_count} markers"
                )

        if args.display:
            vis = image.copy()
            if ids is not None:
                cv2.aruco.drawDetectedMarkers(vis, corners, ids)
            cv2.putText(
                vis,
                f"samples {len(marker_counter)}/{args.max_samples}  markers {marker_count}",
                (24, 44),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0,
                (0, 255, 0) if accepted else (0, 200, 255),
                2,
                cv2.LINE_AA,
            )
            cv2.imshow("apriltag_grid_intrinsics", vis)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break

        rate.sleep()

    if args.display:
        cv2.destroyAllWindows()

    if image_size is None:
        print("No image received.")
        return 1
    if len(marker_counter) < args.min_samples:
        print(
            f"Only collected {len(marker_counter)} samples; need at least "
            f"{args.min_samples}. No calibration written."
        )
        return 1

    rms, camera_matrix, dist_coeffs = _calibrate(
        all_corners=all_corners,
        all_ids=all_ids,
        marker_counter=marker_counter,
        board=board,
        image_size=image_size,
    )
    output = Path(args.output).expanduser()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        _ros_camera_yaml(
            camera_name=args.camera_name,
            width=image_size[0],
            height=image_size[1],
            camera_matrix=camera_matrix,
            dist_coeffs=dist_coeffs,
        )
    )
    print(f"wrote {output}")
    print(f"RMS reprojection error: {rms:.6f} px")
    print("K:")
    print(camera_matrix)
    print("D:")
    print(np.asarray(dist_coeffs).reshape(-1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
