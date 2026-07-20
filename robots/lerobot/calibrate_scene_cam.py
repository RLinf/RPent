#!/usr/bin/env python3
"""Touch / point-correspondence calibration of the scene camera -> arm base.

Computes the fixed extrinsic ``T_base_cam`` that maps scene-camera points into
the SO101 ``base_link`` world frame, using only the arm's own FK + the scene
camera's aligned depth (no marker / no extra hardware). The result is saved via
:mod:`robots.lerobot.calibration` and auto-loaded by the env server, after
which ``back_project`` returns world coordinates.

Procedure (per correspondence):

1. Free-drive the arm by hand so the gripper tip (``gripper_frame_link``,
   roughly the point between the fingertips) rests at a distinct location that
   is clearly visible to the scene camera.
2. The script reads the tip position in the base frame from FK
   (``env.get_ee_pose``) and grabs the scene color + aligned depth
   (``env.get_scene_frame``).
3. You click that same tip point in the color image; the script backprojects
   the clicked pixel (median depth over a small patch) into the camera frame.

After N>=4 non-coplanar points, a rigid Kabsch fit gives ``T_base_cam`` and the
fit RMSE (lower is better; aim for < ~1 cm).

Run the env server first (note its --transport-port), then::

    conda activate lerobot
    python toolkits/lerobot/calibrate_scene_cam.py --port 53101 --num-points 6

Offline self-test of the math (no hardware)::

    python toolkits/lerobot/calibrate_scene_cam.py --self-test
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
from rpent.utils.socket_rpc import SocketRpcClient  # noqa: E402


def _self_test() -> int:
    """Validate the Kabsch pipeline on synthetic correspondences (no hardware)."""
    rng = np.random.default_rng(0)
    A = rng.standard_normal((3, 3))
    Q, _ = np.linalg.qr(A)
    if np.linalg.det(Q) < 0:
        Q[:, 0] = -Q[:, 0]
    T_true = np.eye(4)
    T_true[:3, :3] = Q
    T_true[:3, 3] = rng.standard_normal(3)

    cam_pts = rng.standard_normal((8, 3))
    base_pts = geom.transform_points(T_true, cam_pts) + rng.standard_normal((8, 3)) * 1e-3
    T_est, rmse = geom.kabsch_umeyama(cam_pts, base_pts)
    ok = np.allclose(T_est, T_true, atol=1e-2)
    print(f"self-test: rmse={rmse:.5f}m  recovered={'OK' if ok else 'FAIL'}")
    return 0 if ok else 1


def _click_pixel(color: np.ndarray, idx: int, total: int) -> tuple[float, float] | None:
    """Show the color frame and return the clicked (col, row), or None if skipped."""
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=(8, 6))
    plt.imshow(color)
    plt.title(f"[{idx}/{total}] Click the gripper TIP, then close is automatic. "
              "Close window to skip.")
    plt.tight_layout()
    pts = plt.ginput(1, timeout=0)
    plt.close(fig)
    if not pts:
        return None
    return float(pts[0][0]), float(pts[0][1])


def _collect(client: SocketRpcClient, num_points: int, patch_radius: int) -> tuple[np.ndarray, np.ndarray]:
    """Collect (cam_point, base_point) correspondences interactively."""
    cam_pts: list[list[float]] = []
    base_pts: list[list[float]] = []

    i = 0
    while len(cam_pts) < num_points:
        i += 1
        input(
            f"\n[{len(cam_pts) + 1}/{num_points}] Move the gripper tip to a distinct "
            "scene point (vary x/y/z), hold it, then press Enter to capture..."
        )
        ee = client.call("env.get_ee_pose", timeout_s=15)
        if "error" in ee:
            print(f"  get_ee_pose failed: {ee['error']}")
            continue
        frame = client.call("env.get_scene_frame", timeout_s=15)
        if "error" in frame:
            print(f"  get_scene_frame failed: {frame['error']}")
            continue

        color = np.asarray(frame["color"], dtype=np.uint8)
        depth = np.asarray(frame["depth"], dtype=np.float32)
        K = np.asarray(frame["K"], dtype=np.float64)

        click = _click_pixel(color, len(cam_pts) + 1, num_points)
        if click is None:
            print("  skipped (no pixel clicked).")
            continue
        col, row = click
        z = geom.sample_depth_patch(depth, int(round(col)), int(round(row)), radius=patch_radius)
        if not np.isfinite(z) or z <= 0:
            print(f"  no valid depth at ({int(row)},{int(col)}); try another point/angle.")
            continue

        p_cam = geom.backproject_pixel(K, col, row, z)
        p_base = np.asarray(ee["xyz"], dtype=np.float64)
        cam_pts.append(p_cam.tolist())
        base_pts.append(p_base.tolist())
        print(f"  captured: cam={np.round(p_cam, 3).tolist()}  "
              f"base={np.round(p_base, 3).tolist()}  depth={z:.3f}m")

    return np.asarray(cam_pts), np.asarray(base_pts)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--host", default="127.0.0.1", help="env server host.")
    ap.add_argument("--port", type=int, help="env server transport port.")
    ap.add_argument("--num-points", type=int, default=6,
                    help="Number of correspondences (>=4; more is better).")
    ap.add_argument("--patch-radius", type=int, default=2,
                    help="Depth median patch radius (pixels) around each click.")
    ap.add_argument("--serial", default=None,
                    help="Override scene serial for the saved file "
                         "(default: from env.get_scene_camera_meta).")
    ap.add_argument("--self-test", action="store_true",
                    help="Run the offline Kabsch math check and exit.")
    args = ap.parse_args()

    if args.self_test:
        return _self_test()
    if args.port is None:
        ap.error("--port is required (the env server's transport port).")
    if args.num_points < 4:
        ap.error("need at least 4 correspondences for a stable fit.")

    client = SocketRpcClient(args.host, args.port)
    meta = client.call("env.get_scene_camera_meta", timeout_s=15)
    if "error" in meta:
        print(f"scene camera not available: {meta['error']}")
        return 2
    serial = args.serial or meta.get("serial")
    if not serial:
        print("could not determine scene camera serial; pass --serial.")
        return 2
    print(f"Calibrating scene camera serial={serial}")

    # Free-drive so the operator can position the tip by hand.
    client.call("env.set_torque", args=(False,), timeout_s=15)
    print("Arm torque DISABLED — you can move it by hand. (Re-enabled at the end.)")
    try:
        cam_pts, base_pts = _collect(client, args.num_points, args.patch_radius)
    finally:
        client.call("env.set_torque", args=(True,), timeout_s=15)
        print("Arm torque re-enabled.")

    T_base_cam, rmse = geom.kabsch_umeyama(cam_pts, base_pts)
    print(f"\nFit complete: {len(cam_pts)} points, RMSE = {rmse * 1000:.1f} mm")
    if rmse > 0.02:
        print("WARNING: RMSE > 2 cm — consider recollecting with more spread / "
              "better tip-pixel clicks.")

    path = scene_calib.save_extrinsic(
        serial, T_base_cam, K=np.asarray(meta["K"]), rmse_m=rmse, num_points=len(cam_pts)
    )
    print(f"Saved T_base_cam -> {path}")
    print("Restart the env server (or it will pick this up next launch) so "
          "back_project returns world coordinates.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
