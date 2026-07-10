#!/usr/bin/env python3
"""Automatic, markerless scene-camera -> base calibration for the SO101.

Triggers the env server's :meth:`auto_calibrate_scene_camera` routine, which:

1. drives the gripper to a spread grid of base-frame positions (``move_to`` is
   pure base-frame IK, so it needs no extrinsic),
2. at each pose toggles the gripper with the arm frozen and segments the motion
   in the scene image to locate the tip (centroid + median depth -> camera
   point); the achieved FK gives the base point,
3. fits ``T_base_cam`` with RANSAC Kabsch and saves it (hot-loaded by the
   server, so back_project returns world coords immediately).

No human input, no markers. Start the env server first, then run::

    conda activate lerobot
    python deployment/lerobot/auto_calibrate_scene_cam.py --port 53101

WARNING: this moves the arm through many poses. Clear the workspace first.

Offline math check (no hardware)::

    python deployment/lerobot/auto_calibrate_scene_cam.py --self-test
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from deployment.lerobot import geometry as geom  # noqa: E402
from rpent.rpc_driver.socket import SocketRpcClient  # noqa: E402


def _self_test() -> int:
    """Validate RANSAC Kabsch + motion segmentation offline (no hardware)."""
    rng = np.random.default_rng(0)
    Q, _ = np.linalg.qr(rng.standard_normal((3, 3)))
    if np.linalg.det(Q) < 0:
        Q[:, 0] = -Q[:, 0]
    T_true = np.eye(4)
    T_true[:3, :3] = Q
    T_true[:3, 3] = rng.standard_normal(3)
    cam = rng.standard_normal((10, 3))
    base = geom.transform_points(T_true, cam) + rng.standard_normal((10, 3)) * 1e-3
    base[3] += [0.2, -0.15, 0.1]  # inject an outlier
    T_est, rmse, inliers = geom.ransac_kabsch(cam, base, thresh_m=0.02)
    ok_fit = np.allclose(T_est, T_true, atol=2e-2) and not inliers[3]
    print(f"ransac_kabsch: rmse={rmse:.5f}m inliers={int(inliers.sum())}/10 "
          f"outlier_excluded={not inliers[3]} -> {'OK' if ok_fit else 'FAIL'}")

    # Synthetic two-frame motion: a blob that appears in the 'closed' frame.
    H, W = 480, 640
    rgb_open = np.zeros((H, W, 3), np.uint8)
    rgb_closed = rgb_open.copy()
    rgb_closed[300:330, 400:430] = 255  # 'fingers' light up at (row~315,col~415)
    depth = np.full((H, W), 0.4, np.float32)
    K = np.array([[600, 0, 320], [0, 600, 240], [0, 0, 1]], float)
    det = geom.detect_tip_pixel_by_motion(rgb_open, rgb_closed, depth, K)
    ok_det = det is not None and abs(det["pixel"][0] - 314.5) < 3 and abs(det["pixel"][1] - 414.5) < 3
    print(f"detect_tip: {det if det else 'None'} -> {'OK' if ok_det else 'FAIL'}")
    return 0 if (ok_fit and ok_det) else 1


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, help="env server transport port.")
    ap.add_argument("--n-points", type=int, default=10,
                    help="Target number of valid correspondences to collect.")
    ap.add_argument("--no-save", action="store_true",
                    help="Compute T_base_cam but do not write it to disk.")
    ap.add_argument("--self-test", action="store_true",
                    help="Run the offline math check and exit.")
    args = ap.parse_args()

    if args.self_test:
        return _self_test()
    if args.port is None:
        ap.error("--port is required (the env server's transport port).")

    print("This moves the arm through a grid of poses. Ensure the workspace is "
          "clear. Starting...")
    client = SocketRpcClient(args.host, args.port)
    result = client.call(
        "env.auto_calibrate_scene_camera",
        kwargs={"n_points": args.n_points, "save": not args.no_save},
        timeout_s=600.0,
    )

    if "error" in result:
        print(f"Calibration failed: {result['error']}")
        if "poses" in result:
            print(json.dumps(result["poses"], indent=2))
        return 2

    print(f"\nUsed {result['n_used']} poses "
          f"({result['n_inliers']} inliers), RMSE = {result['rmse_m'] * 1000:.1f} mm")
    if result["rmse_m"] > 0.02:
        print("WARNING: RMSE > 2 cm — check lighting / gripper visibility and rerun.")
    if result.get("saved"):
        print(f"Saved T_base_cam -> {result['path']}")
        print("The server hot-loaded it; back_project now returns world coords.")
    else:
        print("Not saved (--no-save). T_base_cam:")
        print(json.dumps(result["T_base_cam"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
