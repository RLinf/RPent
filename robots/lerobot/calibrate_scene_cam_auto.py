#!/usr/bin/env python3
"""Automatic, markerless scene-camera -> base calibration for the SO101.

Drives the gripper through a non-coplanar grid of base-frame positions, detects
the moving jaw by toggling the gripper at each pose, and jointly fits the scene
camera extrinsic and the detector-to-tip offset. The environment server only
provides robot and camera operations; this script owns the calibration workflow.

Start the environment server first, then run::

    python robots/lerobot/calibrate_scene_cam_auto.py --port 53101

The server uses HTTP by default. Pass ``--transport socket`` when it was
started with the socket transport.

WARNING: this moves the arm through many poses. Clear the workspace first.

Offline math check (no hardware)::

    python robots/lerobot/calibrate_scene_cam_auto.py --self-test
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import warnings
from pathlib import Path

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from robots.lerobot import calibration as scene_calib  # noqa: E402
from robots.lerobot import geometry as geom  # noqa: E402
from rpent.utils.http_rpc import HttpRpcClient  # noqa: E402
from rpent.utils.rpc import RpcClient  # noqa: E402
from rpent.utils.socket_rpc import SocketRpcClient  # noqa: E402


def _build_client(transport: str, host: str, port: int) -> RpcClient:
    if transport == "http":
        return HttpRpcClient(f"http://{host}:{port}")
    return SocketRpcClient(host, port)


def _calibration_targets() -> list[list[float]]:
    """Return a non-coplanar target grid with varied free-IK orientations."""
    xs = [0.15, 0.21, 0.27, 0.33]
    ys = [-0.16, -0.05, 0.05, 0.16]
    zs = [0.08, 0.15, 0.22]
    return [[x, y, z] for x in xs for y in ys for z in zs]


def _capture_scene_median(
    client: RpcClient,
    n_frames: int = 5,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Capture scene frames and return median RGB, depth, and intrinsics."""
    rgbs: list[np.ndarray] = []
    depths: list[np.ndarray] = []
    intrinsics: np.ndarray | None = None
    for _ in range(max(1, int(n_frames))):
        frame = client.call("env.get_scene_frame", timeout_s=60.0)
        if "error" in frame:
            raise RuntimeError(frame["error"])
        rgbs.append(np.asarray(frame["color"], dtype=np.uint8))
        depth = np.asarray(frame["depth"], dtype=np.float32)
        depths.append(np.where(np.isfinite(depth) & (depth > 0), depth, np.nan))
        intrinsics = np.asarray(frame["K"], dtype=np.float64)

    rgb_median = np.median(np.stack(rgbs), axis=0).astype(np.uint8)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        depth_median = np.nanmedian(np.stack(depths), axis=0)
    depth_median = np.nan_to_num(depth_median, nan=0.0).astype(np.float32)
    assert intrinsics is not None
    return rgb_median, depth_median, intrinsics


def auto_calibrate_scene_camera(
    client: RpcClient,
    *,
    n_points: int = 24,
    gripper_open: float = 90.0,
    gripper_closed: float = 20.0,
    settle_s: float = 0.8,
    ransac_thresh_m: float = 0.015,
    save: bool = True,
) -> dict:
    """Run markerless scene-camera calibration through low-level env RPCs."""
    metadata = client.call("env.get_scene_camera_meta", timeout_s=30.0)
    if "error" in metadata:
        return {"error": metadata["error"]}

    targets = _calibration_targets()
    camera_points: list[list[float]] = []
    tip_origins: list[list[float]] = []
    tip_rotations: list[list[list[float]]] = []
    poses: list[dict] = []

    try:
        for target in targets:
            if len(camera_points) >= n_points:
                break
            move_result = client.call(
                "env.move_to",
                args=(target,),
                kwargs={"gripper": gripper_open, "settle_s": settle_s},
                timeout_s=180.0,
            )
            if "error" in move_result or not move_result.get("reached"):
                poses.append({"target": target, "skipped": "unreachable"})
                continue
            time.sleep(settle_s)

            ee_pose = client.call("env.get_ee_pose", timeout_s=30.0)
            if "error" in ee_pose:
                poses.append({"target": target, "skipped": "fk_unavailable"})
                continue
            tip_transform = np.asarray(
                ee_pose["T_base_gripper"], dtype=np.float64
            )

            client.call("env.get_scene_frame", timeout_s=60.0)
            rgb_open, _, intrinsics = _capture_scene_median(client)
            client.call(
                "env.set_gripper",
                args=(gripper_closed,),
                timeout_s=30.0,
            )
            time.sleep(settle_s)
            rgb_closed, depth, intrinsics = _capture_scene_median(client)
            client.call(
                "env.set_gripper",
                args=(gripper_open,),
                timeout_s=30.0,
            )

            detection = geom.detect_tip_pixel_by_motion(
                rgb_open,
                rgb_closed,
                depth,
                intrinsics,
            )
            if detection is None:
                poses.append({"target": target, "skipped": "no_tip_detected"})
                continue

            tip_origin = tip_transform[:3, 3]
            camera_points.append(detection["xyz_cam"])
            tip_origins.append(tip_origin.tolist())
            tip_rotations.append(tip_transform[:3, :3].tolist())
            poses.append(
                {
                    "target": target,
                    "base_xyz": tip_origin.round(4).tolist(),
                    "pixel": [round(value, 1) for value in detection["pixel"]],
                    "depth_m": round(detection["depth_m"], 4),
                    "area": detection["area"],
                }
            )
    finally:
        try:
            client.call(
                "env.set_gripper",
                args=(gripper_open,),
                timeout_s=30.0,
            )
        finally:
            client.call("env.reset", timeout_s=180.0)

    if len(camera_points) < 4:
        return {
            "error": f"only {len(camera_points)} usable points (need >= 4)",
            "poses": poses,
        }

    transform, rmse, inliers, tip_offset = geom.solve_extrinsic_with_offset(
        camera_points,
        tip_origins,
        tip_rotations,
        thresh_m=ransac_thresh_m,
    )
    accepted = bool(rmse <= scene_calib.MAX_ACCEPTABLE_RMSE_M)
    result = {
        "n_targets": len(targets),
        "n_used": len(camera_points),
        "n_inliers": int(np.asarray(inliers).sum()),
        "rmse_m": round(float(rmse), 4),
        "tip_offset_local_m": [round(float(value), 4) for value in tip_offset],
        "T_base_cam": transform.tolist(),
        "accepted": accepted,
        "saved": False,
        "poses": poses,
    }
    if not accepted:
        result["error"] = (
            f"calibration RMSE {rmse * 1000:.1f} mm exceeds the "
            f"{scene_calib.MAX_ACCEPTABLE_RMSE_M * 1000:.0f} mm limit; not "
            "saved. Clear the workspace, improve gripper visibility/lighting, "
            "and rerun."
        )
        return result

    if save:
        serial = metadata.get("serial")
        if not serial:
            result["error"] = "scene camera metadata did not include a serial"
            return result
        save_result = client.call(
            "env.save_scene_camera_calibration",
            args=(transform,),
            kwargs={
                "rmse_m": rmse,
                "num_points": int(np.asarray(inliers).sum()),
            },
            timeout_s=30.0,
        )
        if "error" in save_result:
            result["error"] = save_result["error"]
            return result
        result["saved"] = True
        result["path"] = save_result["path"]
    return result


def _self_test() -> int:
    """Validate robust fitting, motion segmentation, and offset recovery."""
    rng = np.random.default_rng(0)
    rotation, _ = np.linalg.qr(rng.standard_normal((3, 3)))
    if np.linalg.det(rotation) < 0:
        rotation[:, 0] = -rotation[:, 0]
    expected_transform = np.eye(4)
    expected_transform[:3, :3] = rotation
    expected_transform[:3, 3] = rng.standard_normal(3)
    camera_points = rng.standard_normal((10, 3))
    base_points = geom.transform_points(expected_transform, camera_points)
    base_points += rng.standard_normal((10, 3)) * 1e-3
    base_points[3] += [0.2, -0.15, 0.1]
    estimated_transform, rmse, inliers = geom.ransac_kabsch(
        camera_points,
        base_points,
        thresh_m=0.02,
    )
    fit_ok = np.allclose(
        estimated_transform,
        expected_transform,
        atol=2e-2,
    ) and not inliers[3]
    print(
        f"ransac_kabsch: rmse={rmse:.5f}m "
        f"inliers={int(inliers.sum())}/10 "
        f"outlier_excluded={not inliers[3]} -> {'OK' if fit_ok else 'FAIL'}"
    )

    height, width = 480, 640
    rgb_open = np.zeros((height, width, 3), np.uint8)
    rgb_closed = rgb_open.copy()
    rgb_closed[300:330, 400:430] = 255
    depth = np.full((height, width), 0.4, np.float32)
    intrinsics = np.array([[600, 0, 320], [0, 600, 240], [0, 0, 1]], float)
    detection = geom.detect_tip_pixel_by_motion(
        rgb_open,
        rgb_closed,
        depth,
        intrinsics,
    )
    detection_ok = (
        detection is not None
        and abs(detection["pixel"][0] - 314.5) < 3
        and abs(detection["pixel"][1] - 414.5) < 3
    )
    print(
        f"detect_tip: {detection if detection else 'None'} -> "
        f"{'OK' if detection_ok else 'FAIL'}"
    )

    offset_rng = np.random.default_rng(2)
    camera_rotation, _ = np.linalg.qr(offset_rng.standard_normal((3, 3)))
    if np.linalg.det(camera_rotation) < 0:
        camera_rotation[:, 0] = -camera_rotation[:, 0]
    camera_transform = np.eye(4)
    camera_transform[:3, :3] = camera_rotation
    camera_transform[:3, 3] = offset_rng.standard_normal(3)
    expected_offset = np.array([0.015, -0.010, 0.020])
    point_count = 24
    origins = offset_rng.uniform(
        [-0.1, -0.2, 0.0],
        [0.35, 0.2, 0.25],
        size=(point_count, 3),
    )
    rotations = np.empty((point_count, 3, 3))
    for index in range(point_count):
        pose_rotation, _ = np.linalg.qr(offset_rng.standard_normal((3, 3)))
        if np.linalg.det(pose_rotation) < 0:
            pose_rotation[:, 0] = -pose_rotation[:, 0]
        rotations[index] = pose_rotation
    feature_points = origins + np.einsum(
        "nij,j->ni",
        rotations,
        expected_offset,
    )
    offset_camera_points = geom.transform_points(
        geom.invert_transform(camera_transform),
        feature_points,
    )
    offset_camera_points += offset_rng.standard_normal((point_count, 3)) * 1e-3
    offset_camera_points[5] += np.array([0.18, -0.12, 0.15])
    estimated_camera, offset_rmse, offset_inliers, estimated_offset = (
        geom.solve_extrinsic_with_offset(
            offset_camera_points,
            origins,
            rotations,
            thresh_m=0.01,
        )
    )
    _, offset_blind_rmse, _ = geom.ransac_kabsch(
        offset_camera_points,
        origins,
        thresh_m=0.01,
    )
    offset_ok = (
        np.allclose(
            estimated_camera[:3, :3],
            camera_transform[:3, :3],
            atol=5e-3,
        )
        and np.allclose(
            estimated_camera[:3, 3],
            camera_transform[:3, 3],
            atol=5e-3,
        )
        and np.allclose(estimated_offset, expected_offset, atol=5e-3)
        and not bool(offset_inliers[5])
    )
    print(
        f"solve_offset: rmse={offset_rmse * 1000:.2f}mm "
        f"(offset-blind {offset_blind_rmse * 1000:.1f}mm) "
        f"d_err={np.linalg.norm(estimated_offset - expected_offset) * 1000:.2f}mm "
        f"outlier_excluded={not bool(offset_inliers[5])} -> "
        f"{'OK' if offset_ok else 'FAIL'}"
    )
    return 0 if fit_ok and detection_ok and offset_ok else 1


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, help="Environment server RPC port.")
    parser.add_argument(
        "--transport",
        choices=["http", "socket"],
        default="http",
        help="RPC transport used by the environment server.",
    )
    parser.add_argument(
        "--n-points",
        type=int,
        default=24,
        help="Target number of valid correspondences to collect.",
    )
    parser.add_argument(
        "--no-save",
        action="store_true",
        help="Compute T_base_cam but do not write it to disk.",
    )
    parser.add_argument(
        "--self-test",
        action="store_true",
        help="Run the offline math check and exit.",
    )
    args = parser.parse_args()

    if args.self_test:
        return _self_test()
    if args.port is None:
        parser.error("--port is required (the environment server RPC port).")

    print(
        "This moves the arm through a grid of poses. Ensure the workspace is "
        "clear. Starting..."
    )
    result = auto_calibrate_scene_camera(
        _build_client(args.transport, args.host, args.port),
        n_points=args.n_points,
        save=not args.no_save,
    )

    if "error" in result:
        print(f"Calibration failed: {result['error']}")
        if "poses" in result:
            print(json.dumps(result["poses"], indent=2))
        return 2

    print(
        f"\nUsed {result['n_used']} poses ({result['n_inliers']} inliers), "
        f"RMSE = {result['rmse_m'] * 1000:.1f} mm"
    )
    offset = result.get("tip_offset_local_m")
    if offset:
        print(
            "Estimated tip-detector offset (gripper frame): "
            f"[{offset[0] * 1000:.1f}, {offset[1] * 1000:.1f}, "
            f"{offset[2] * 1000:.1f}] mm"
        )
    if result.get("saved"):
        print(f"Saved T_base_cam -> {result['path']}")
        print("The environment server activated the new calibration.")
    else:
        print("Not saved (--no-save). T_base_cam:")
        print(json.dumps(result["T_base_cam"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
