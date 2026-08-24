#!/usr/bin/env python3
"""Manual point-correspondence calibration of the scene camera to arm base.

The operator free-drives the arm to several visible positions and clicks the
gripper tip in each scene image. Forward kinematics supplies the corresponding
base-frame points, and a rigid Kabsch fit produces ``T_base_cam``.

Start the environment server first, then run::

    python robots/lerobot/calibrate_scene_cam_manual.py --port 53101

The server uses HTTP by default. Pass ``--transport socket`` when it was
started with the socket transport.

Offline math check (no hardware)::

    python robots/lerobot/calibrate_scene_cam_manual.py --self-test
"""
from __future__ import annotations

import argparse
import sys
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


def _self_test() -> int:
    """Validate the Kabsch pipeline on synthetic correspondences."""
    rng = np.random.default_rng(0)
    matrix = rng.standard_normal((3, 3))
    rotation, _ = np.linalg.qr(matrix)
    if np.linalg.det(rotation) < 0:
        rotation[:, 0] = -rotation[:, 0]
    expected_transform = np.eye(4)
    expected_transform[:3, :3] = rotation
    expected_transform[:3, 3] = rng.standard_normal(3)

    camera_points = rng.standard_normal((8, 3))
    base_points = geom.transform_points(expected_transform, camera_points)
    base_points += rng.standard_normal((8, 3)) * 1e-3
    estimated_transform, rmse = geom.kabsch_umeyama(camera_points, base_points)
    passed = np.allclose(estimated_transform, expected_transform, atol=1e-2)
    print(f"self-test: rmse={rmse:.5f}m recovered={'OK' if passed else 'FAIL'}")
    return 0 if passed else 1


def _click_pixel(
    color: np.ndarray,
    index: int,
    total: int,
) -> tuple[float, float] | None:
    """Show the color frame and return the clicked column and row."""
    import matplotlib.pyplot as plt

    figure = plt.figure(figsize=(8, 6))
    plt.imshow(color)
    plt.title(
        f"[{index}/{total}] Click the gripper tip. Close the window to skip."
    )
    plt.tight_layout()
    points = plt.ginput(1, timeout=0)
    plt.close(figure)
    if not points:
        return None
    return float(points[0][0]), float(points[0][1])


def _collect(
    client: RpcClient,
    num_points: int,
    patch_radius: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Collect camera-frame and base-frame point correspondences."""
    camera_points: list[list[float]] = []
    base_points: list[list[float]] = []

    while len(camera_points) < num_points:
        input(
            f"\n[{len(camera_points) + 1}/{num_points}] Move the gripper tip to "
            "a distinct scene point, hold it, then press Enter to capture..."
        )
        ee_pose = client.call("env.get_ee_pose", timeout_s=30.0)
        if "error" in ee_pose:
            print(f"  get_ee_pose failed: {ee_pose['error']}")
            continue
        frame = client.call("env.get_scene_frame", timeout_s=60.0)
        if "error" in frame:
            print(f"  get_scene_frame failed: {frame['error']}")
            continue

        color = np.asarray(frame["color"], dtype=np.uint8)
        depth = np.asarray(frame["depth"], dtype=np.float32)
        intrinsics = np.asarray(frame["K"], dtype=np.float64)
        click = _click_pixel(color, len(camera_points) + 1, num_points)
        if click is None:
            print("  skipped (no pixel clicked).")
            continue

        column, row = click
        depth_m = geom.sample_depth_patch(
            depth,
            int(round(column)),
            int(round(row)),
            radius=patch_radius,
        )
        if not np.isfinite(depth_m) or depth_m <= 0:
            print(
                f"  no valid depth at ({int(row)}, {int(column)}); "
                "try another point or angle."
            )
            continue

        camera_point = geom.backproject_pixel(
            intrinsics,
            column,
            row,
            depth_m,
        )
        base_point = np.asarray(ee_pose["xyz"], dtype=np.float64)
        camera_points.append(camera_point.tolist())
        base_points.append(base_point.tolist())
        print(
            f"  captured: cam={np.round(camera_point, 3).tolist()} "
            f"base={np.round(base_point, 3).tolist()} depth={depth_m:.3f}m"
        )

    return np.asarray(camera_points), np.asarray(base_points)


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
        "--num-points",
        type=int,
        default=6,
        help="Number of correspondences to collect; must be at least four.",
    )
    parser.add_argument(
        "--patch-radius",
        type=int,
        default=2,
        help="Depth median radius around each clicked pixel.",
    )
    parser.add_argument(
        "--self-test",
        action="store_true",
        help="Run the offline Kabsch check and exit.",
    )
    args = parser.parse_args()

    if args.self_test:
        return _self_test()
    if args.port is None:
        parser.error("--port is required (the environment server RPC port).")
    if args.num_points < 4:
        parser.error("Need at least four correspondences for a stable fit.")

    client = _build_client(args.transport, args.host, args.port)
    metadata = client.call("env.get_scene_camera_meta", timeout_s=30.0)
    if "error" in metadata:
        print(f"Scene camera not available: {metadata['error']}")
        return 2
    serial = metadata.get("serial")
    if not serial:
        print("The scene camera metadata did not include a serial.")
        return 2
    print(f"Calibrating scene camera serial={serial}")

    client.call("env.set_torque", args=(False,), timeout_s=30.0)
    print("Arm torque disabled; you can move it by hand.")
    try:
        camera_points, base_points = _collect(
            client,
            args.num_points,
            args.patch_radius,
        )
    finally:
        client.call("env.set_torque", args=(True,), timeout_s=30.0)
        print("Arm torque re-enabled.")

    transform, rmse = geom.kabsch_umeyama(camera_points, base_points)
    print(
        f"\nFit complete: {len(camera_points)} points, "
        f"RMSE = {rmse * 1000:.1f} mm"
    )
    if rmse > scene_calib.MAX_ACCEPTABLE_RMSE_M:
        print(
            "WARNING: RMSE exceeds 2 cm; recollect with wider spacing and "
            "more accurate tip clicks."
        )

    save_result = client.call(
        "env.save_scene_camera_calibration",
        args=(transform,),
        kwargs={"rmse_m": rmse, "num_points": len(camera_points)},
        timeout_s=30.0,
    )
    if "error" in save_result:
        print(f"Calibration not saved: {save_result['error']}")
        return 2
    print(f"Saved T_base_cam -> {save_result['path']}")
    print("The environment server activated the new calibration.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
