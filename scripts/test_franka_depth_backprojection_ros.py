"""Capture RealSense depth frames and back-project all valid pixels to FR3 base.

This is a hardware diagnostic script for the two calibrated RealSense views:

- external: external_cam_color_optical_frame -> fr3_link0 from hand-eye JSON
- wrist: wrist_cam_color_optical_frame -> fr3_EE from hand-eye JSON, then
  fr3_EE -> fr3_link0 from live TF
"""
from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

DEFAULT_CALIBRATION_BUNDLE_PATH = (
    Path(__file__).resolve().parents[1]
    / "robots"
    / "franka"
    / "calibration"
    / "hand_eye_calibration.json"
)


@dataclass(frozen=True)
class CameraSpec:
    name: str
    depth_topic: str
    info_topic: str
    calibration_key: str
    target_frame: str


CAMERAS = (
    CameraSpec(
        name="external",
        depth_topic="/external_cam/aligned_depth_to_color/image_raw",
        info_topic="/external_cam/color/camera_info",
        calibration_key="external",
        target_frame="base",
    ),
    CameraSpec(
        name="wrist",
        depth_topic="/wrist_cam/aligned_depth_to_color/image_raw",
        info_topic="/wrist_cam/color/camera_info",
        calibration_key="wrist",
        target_frame="ee",
    ),
)


def _image_to_depth_m(msg: Any) -> np.ndarray:
    dtype: np.dtype[Any]
    scale = 1.0
    if msg.encoding == "16UC1":
        dtype = np.dtype(np.uint16)
        scale = 0.001
    elif msg.encoding == "32FC1":
        dtype = np.dtype(np.float32)
    else:
        raise ValueError(f"unsupported depth encoding {msg.encoding!r}")

    dtype = dtype.newbyteorder(">" if msg.is_bigendian else "<")
    depth = np.frombuffer(msg.data, dtype=dtype)
    depth = depth.reshape(int(msg.height), int(msg.step) // dtype.itemsize)
    depth = depth[:, : int(msg.width)]
    return depth.astype(np.float32, copy=False) * scale


def _camera_info_k(msg: Any) -> np.ndarray:
    k = np.asarray(msg.K, dtype=np.float64).reshape(3, 3)
    if not np.isfinite(k).all() or abs(float(k[2, 2]) - 1.0) > 1e-9:
        raise ValueError(f"invalid camera K matrix: {k}")
    return k


def _quat_xyzw_to_matrix(quat: np.ndarray) -> np.ndarray:
    x, y, z, w = np.asarray(quat, dtype=np.float64)
    norm = float(np.linalg.norm([x, y, z, w]))
    if norm <= 0:
        raise ValueError("zero-norm quaternion")
    x, y, z, w = x / norm, y / norm, z / norm, w / norm
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def _load_calibration_bundle(path: str | Path) -> dict[str, Any]:
    bundle_path = Path(path).expanduser().resolve()
    bundle = json.loads(bundle_path.read_text(errors="replace"))
    result: dict[str, Any] = {}
    for key in ("external", "wrist"):
        entry = bundle[key]
        transform = entry["transformation"]
        matrix = np.eye(4, dtype=np.float64)
        matrix[:3, :3] = _quat_xyzw_to_matrix(
            np.array(
                [
                    float(transform["qx"]),
                    float(transform["qy"]),
                    float(transform["qz"]),
                    float(transform["qw"]),
                ],
                dtype=np.float64,
            )
        )
        matrix[:3, 3] = [
            float(transform["x"]),
            float(transform["y"]),
            float(transform["z"]),
        ]
        result[key] = {
            "matrix": matrix,
            "source_name": entry.get("source_name"),
            "parameters": entry.get("parameters") or {},
        }
    return result


def _tf_to_matrix(transform: Any) -> np.ndarray:
    t = transform.transform.translation
    q = transform.transform.rotation
    matrix = np.eye(4, dtype=np.float64)
    matrix[:3, :3] = _quat_xyzw_to_matrix(
        np.array([q.x, q.y, q.z, q.w], dtype=np.float64)
    )
    matrix[:3, 3] = [t.x, t.y, t.z]
    return matrix


def _franka_state_to_base_ee_matrix(msg: Any) -> np.ndarray:
    values = np.asarray(msg.O_T_EE, dtype=np.float64)
    if values.shape != (16,):
        raise ValueError(f"expected O_T_EE with 16 values, got {values.shape}")
    return values.reshape(4, 4).T


def _lookup_base_ee_matrix(
    *,
    tf_buffer: Any,
    base_frame: str,
    ee_frame: str,
    timeout: float,
    pose_source: str,
) -> tuple[np.ndarray, str]:
    import rospy

    if pose_source in {"auto", "tf"}:
        try:
            tf_msg = tf_buffer.lookup_transform(
                base_frame,
                ee_frame,
                rospy.Time(0),
                rospy.Duration(timeout),
            )
            return _tf_to_matrix(tf_msg), "tf"
        except Exception:
            if pose_source == "tf":
                raise

    from franka_msgs.msg import FrankaState

    msg = rospy.wait_for_message(
        "/franka_state_controller/franka_states",
        FrankaState,
        timeout=timeout,
    )
    return _franka_state_to_base_ee_matrix(msg), "franka_state_controller/O_T_EE"


def _backproject_all(
    depth_m: np.ndarray,
    k: np.ndarray,
    *,
    max_depth_m: float,
) -> tuple[np.ndarray, np.ndarray]:
    valid = np.isfinite(depth_m) & (depth_m > 0.0) & (depth_m <= max_depth_m)
    rows, cols = np.nonzero(valid)
    z = depth_m[rows, cols].astype(np.float64)
    fx = float(k[0, 0])
    fy = float(k[1, 1])
    cx = float(k[0, 2])
    cy = float(k[1, 2])
    x = (cols.astype(np.float64) - cx) * z / fx
    y = (rows.astype(np.float64) - cy) * z / fy
    points_camera = np.column_stack((x, y, z))
    return points_camera, np.column_stack((rows, cols)).astype(np.int32)


def _transform_points(transform: np.ndarray, points: np.ndarray) -> np.ndarray:
    return points @ transform[:3, :3].T + transform[:3, 3]


def _points_image(
    shape: tuple[int, int],
    pixels_rc: np.ndarray,
    points_base: np.ndarray,
) -> np.ndarray:
    image = np.full((shape[0], shape[1], 3), np.nan, dtype=np.float32)
    image[pixels_rc[:, 0], pixels_rc[:, 1], :] = points_base.astype(np.float32)
    return image


def _summary(
    *,
    spec: CameraSpec,
    depth: np.ndarray,
    points_base: np.ndarray,
    pixels_rc: np.ndarray,
    t_base_camera: np.ndarray,
    depth_topic: str,
    info_topic: str,
) -> dict[str, Any]:
    valid_depth = depth[pixels_rc[:, 0], pixels_rc[:, 1]]
    bounds_min = np.nanmin(points_base, axis=0)
    bounds_max = np.nanmax(points_base, axis=0)
    return {
        "camera": spec.name,
        "depth_topic": depth_topic,
        "camera_info_topic": info_topic,
        "input_depth_shape": list(depth.shape),
        "valid_points": int(points_base.shape[0]),
        "valid_ratio": float(points_base.shape[0] / depth.size),
        "depth_m": {
            "min": float(np.nanmin(valid_depth)),
            "max": float(np.nanmax(valid_depth)),
            "mean": float(np.nanmean(valid_depth)),
        },
        "base_xyz_bounds_m": {
            "min": [float(x) for x in bounds_min],
            "max": [float(x) for x in bounds_max],
        },
        "T_base_camera": np.round(t_base_camera, 8).tolist(),
    }


def _write_visualizations(
    *,
    output_dir: Path,
    camera_name: str,
    points_base: np.ndarray,
    max_points: int = 120_000,
) -> dict[str, str]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    points = np.asarray(points_base, dtype=np.float64)
    if points.shape[0] > max_points:
        stride = int(np.ceil(points.shape[0] / max_points))
        points = points[::stride]

    views = {
        "xy_top": (0, 1, "base x (m)", "base y (m)", "top view: x/y"),
        "xz_front": (0, 2, "base x (m)", "base z (m)", "front view: x/z"),
        "yz_side": (1, 2, "base y (m)", "base z (m)", "side view: y/z"),
    }
    paths: dict[str, str] = {}
    color = points[:, 2] if points.size else np.array([])
    for name, (i, j, xlabel, ylabel, title) in views.items():
        fig, ax = plt.subplots(figsize=(8, 7), dpi=160)
        ax.scatter(points[:, i], points[:, j], c=color, s=0.25, cmap="viridis")
        ax.set_title(f"{camera_name} points in fr3_link0, {title}")
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_aspect("equal", adjustable="box")
        ax.grid(True, linewidth=0.3, alpha=0.4)
        path = output_dir / f"{camera_name}_points_base_{name}.png"
        fig.tight_layout()
        fig.savefig(path)
        plt.close(fig)
        paths[name] = str(path)
    return paths


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Capture one depth frame from the external and wrist RealSense "
            "cameras, then back-project every valid pixel into fr3_link0."
        )
    )
    parser.add_argument(
        "--calibration",
        default=str(DEFAULT_CALIBRATION_BUNDLE_PATH),
        help="Project hand_eye_calibration.json path.",
    )
    parser.add_argument(
        "--output-dir",
        default="depth_backprojection_test",
        help="Directory for .npy outputs and summary.json.",
    )
    parser.add_argument("--base-frame", default="fr3_link0")
    parser.add_argument("--ee-frame", default="fr3_EE")
    parser.add_argument(
        "--max-depth-m",
        type=float,
        default=2.0,
        help="Discard depth pixels farther than this many meters.",
    )
    parser.add_argument(
        "--ee-pose-source",
        choices=["auto", "tf", "franka_state"],
        default="auto",
        help="How to get T_base_EE for wrist projection.",
    )
    parser.add_argument("--timeout", type=float, default=10.0)
    parser.add_argument(
        "--camera",
        choices=["external", "wrist", "both"],
        default="both",
        help="Capture one camera or both cameras.",
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()

    import rospy
    import tf2_ros
    from sensor_msgs.msg import CameraInfo, Image

    rospy.init_node("franka_depth_backprojection_test", anonymous=True)
    tf_buffer = tf2_ros.Buffer()
    tf2_ros.TransformListener(tf_buffer)

    calibration = _load_calibration_bundle(args.calibration)
    output_root = Path(args.output_dir).expanduser().resolve()
    run_dir = output_root / time.strftime("%Y%m%d-%H%M%S")
    run_dir.mkdir(parents=True, exist_ok=True)

    selected = CAMERAS if args.camera == "both" else tuple(
        spec for spec in CAMERAS if spec.name == args.camera
    )
    summaries: dict[str, Any] = {
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "base_frame": args.base_frame,
        "ee_frame": args.ee_frame,
        "calibration_path": str(Path(args.calibration).expanduser().resolve()),
        "cameras": {},
    }

    for spec in selected:
        rospy.loginfo("waiting for %s camera_info: %s", spec.name, spec.info_topic)
        info_msg = rospy.wait_for_message(
            spec.info_topic, CameraInfo, timeout=args.timeout
        )
        rospy.loginfo("waiting for %s depth: %s", spec.name, spec.depth_topic)
        depth_msg = rospy.wait_for_message(spec.depth_topic, Image, timeout=args.timeout)

        k = _camera_info_k(info_msg)
        depth_m = _image_to_depth_m(depth_msg)
        points_camera, pixels_rc = _backproject_all(
            depth_m,
            k,
            max_depth_m=float(args.max_depth_m),
        )
        if points_camera.size == 0:
            raise RuntimeError(
                f"{spec.name} has no valid depth pixels within {args.max_depth_m}m"
            )

        t_calib = calibration[spec.calibration_key]["matrix"]
        if spec.target_frame == "base":
            t_base_camera = t_calib
        else:
            rospy.loginfo(
                "waiting for %s -> %s via %s",
                args.base_frame,
                args.ee_frame,
                args.ee_pose_source,
            )
            t_base_ee, ee_pose_source = _lookup_base_ee_matrix(
                tf_buffer=tf_buffer,
                base_frame=args.base_frame,
                ee_frame=args.ee_frame,
                timeout=args.timeout,
                pose_source=args.ee_pose_source,
            )
            t_base_camera = t_base_ee @ t_calib
            summaries["wrist_ee_pose_source"] = ee_pose_source

        points_base = _transform_points(t_base_camera, points_camera)
        points_base_image = _points_image(depth_m.shape, pixels_rc, points_base)

        np.save(run_dir / f"{spec.name}_depth_m.npy", depth_m.astype(np.float32))
        np.save(run_dir / f"{spec.name}_pixels_rc.npy", pixels_rc)
        np.save(run_dir / f"{spec.name}_points_camera.npy", points_camera.astype(np.float32))
        np.save(run_dir / f"{spec.name}_points_base.npy", points_base.astype(np.float32))
        np.save(run_dir / f"{spec.name}_points_base_image.npy", points_base_image)
        visualization_paths = _write_visualizations(
            output_dir=run_dir,
            camera_name=spec.name,
            points_base=points_base,
        )

        summaries["cameras"][spec.name] = _summary(
            spec=spec,
            depth=depth_m,
            points_base=points_base,
            pixels_rc=pixels_rc,
            t_base_camera=t_base_camera,
            depth_topic=spec.depth_topic,
            info_topic=spec.info_topic,
        )
        summaries["cameras"][spec.name]["max_depth_m"] = float(args.max_depth_m)
        summaries["cameras"][spec.name]["visualizations"] = visualization_paths

    (run_dir / "summary.json").write_text(
        json.dumps(summaries, indent=2, ensure_ascii=False) + "\n"
    )
    print(f"wrote {run_dir}")
    print(json.dumps(summaries, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
