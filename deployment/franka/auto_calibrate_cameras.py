#!/usr/bin/env python3
"""Automatic Franka RGB-D camera calibration helpers.

The default mode calibrates the fixed scene camera into the Franka base frame
(``panda_link0``) without markers:

1. move the TCP through a small, conservative 3-D grid,
2. at each pose, toggle the Franka Hand while the arm is stationary,
3. segment the moving fingers in the scene RGB image and use aligned depth to
   get a camera-frame point,
4. pair that point with the live robot TCP position and fit ``T_base_cam`` with
   RANSAC Kabsch,
5. save the calibration record to
   ``~/.cache/physical_agent/franka/camera_calibration/<serial>.json`` and ask
   the running env server to reload it.

This is the right first calibration for ``back_project`` because the scene
camera is fixed. The wrist camera is eye-in-hand; calibrating it correctly needs
hand-eye calibration (``T_tcp_cam``) using a fixed fiducial/ChArUco/AprilTag or a
scene-calibrated reference target observed from multiple wrist poses. This file
keeps the wrist record format ready, but does not invent an unsafe automatic
wrist calibration from one moving camera alone.

Run the env server first, then run in the physicalagent env::

    python deployment/franka/auto_calibrate_cameras.py --port 5599 --yes

Offline math check::

    python deployment/franka/auto_calibrate_cameras.py --self-test
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import imageio.v2 as imageio
import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from deployment.franka import calibration as franka_calib  # noqa: E402
from deployment.lerobot import geometry as geom  # noqa: E402
from physical_agent.rpc_driver.socket import SocketRpcClient  # noqa: E402

_DEFAULT_GRID_X = (0.46, 0.54, 0.60)
_DEFAULT_GRID_Y = (-0.08, 0.0, 0.08)
_DEFAULT_GRID_Z = (0.20, 0.27)
_DEFAULT_DOWN_EULER = [float(np.pi), 0.0, 0.0]


def _self_test() -> int:
    """Validate Kabsch/RANSAC and motion-blob detection offline."""
    rng = np.random.default_rng(3)
    q, _ = np.linalg.qr(rng.standard_normal((3, 3)))
    if np.linalg.det(q) < 0:
        q[:, 0] = -q[:, 0]
    T_true = np.eye(4)
    T_true[:3, :3] = q
    T_true[:3, 3] = rng.normal(size=3)

    cam = rng.normal(size=(12, 3))
    base = geom.transform_points(T_true, cam) + rng.normal(scale=0.001, size=(12, 3))
    base[5] += [0.12, -0.08, 0.05]
    T_est, rmse, inliers = geom.ransac_kabsch(cam, base, thresh_m=0.015)
    fit_ok = np.allclose(T_est, T_true, atol=2e-2) and not bool(inliers[5])
    print(
        "ransac_kabsch: "
        f"rmse={rmse:.5f}m inliers={int(inliers.sum())}/12 "
        f"outlier_excluded={not bool(inliers[5])} -> {'OK' if fit_ok else 'FAIL'}"
    )

    H, W = 480, 640
    rgb_open = np.zeros((H, W, 3), np.uint8)
    rgb_closed = rgb_open.copy()
    rgb_closed[210:235, 330:365] = 255
    depth = np.full((H, W), 0.55, np.float32)
    K = np.array([[600.0, 0.0, 320.0], [0.0, 600.0, 240.0], [0.0, 0.0, 1.0]])
    det = _detect_tip_pixel_by_motion(rgb_open, rgb_closed, depth, K)
    det_ok = det is not None and abs(det["pixel"][0] - 222) < 4 and abs(det["pixel"][1] - 347) < 4
    print(f"detect_tip: {det if det else 'None'} -> {'OK' if det_ok else 'FAIL'}")
    return 0 if fit_ok and det_ok else 1


def _detect_motion_blob_numpy(
    rgb_open,
    rgb_closed,
    depth_m,
    K,
    *,
    diff_thresh: int,
    min_area: int,
    max_area: int,
) -> dict | None:
    """Locate the largest changed depth-valid blob without OpenCV."""
    a = np.asarray(rgb_open, dtype=np.float32).mean(axis=2)
    b = np.asarray(rgb_closed, dtype=np.float32).mean(axis=2)
    depth_m = np.asarray(depth_m, dtype=np.float64)
    mask = (np.abs(a - b) >= int(diff_thresh)) & np.isfinite(depth_m) & (depth_m > 0)
    if not np.any(mask):
        return None

    try:
        from scipy import ndimage

        labels, num = ndimage.label(mask)
        best = None
        best_area = 0
        for label in range(1, num + 1):
            comp = labels == label
            area = int(comp.sum())
            if min_area <= area <= max_area and area > best_area:
                best = comp
                best_area = area
        if best is None:
            return None
        rows, cols = np.nonzero(best)
        depths = depth_m[best]
    except Exception:
        rows, cols = np.nonzero(mask)
        depths = depth_m[mask]
        best_area = int(rows.size)
        if not (min_area <= best_area <= max_area):
            return None

    row = float(np.median(rows))
    col = float(np.median(cols))
    z = float(np.median(depths))
    return {
        "pixel": [row, col],
        "depth_m": z,
        "area": best_area,
        "xyz_cam": geom.backproject_pixel(K, col, row, z).tolist(),
    }


def _save_debug_images(
    *,
    debug_dir: Path,
    pose_idx: int,
    camera: str,
    rgb_open,
    rgb_closed,
    depth_m,
) -> dict[str, str]:
    """Save open/closed/diff/depth images for diagnosing failed detections."""
    debug_dir.mkdir(parents=True, exist_ok=True)
    prefix = debug_dir / f"{pose_idx:02d}_{camera}"
    rgb_open = np.asarray(rgb_open, dtype=np.uint8)
    rgb_closed = np.asarray(rgb_closed, dtype=np.uint8)
    depth_m = np.asarray(depth_m, dtype=np.float32)

    diff = np.abs(rgb_open.astype(np.int16) - rgb_closed.astype(np.int16)).max(axis=2)
    diff_img = np.clip(diff, 0, 255).astype(np.uint8)

    valid = np.isfinite(depth_m) & (depth_m > 0)
    depth_img = np.zeros(depth_m.shape, dtype=np.uint8)
    if np.any(valid):
        lo, hi = np.percentile(depth_m[valid], [2, 98])
        if hi > lo:
            depth_img[valid] = np.clip((depth_m[valid] - lo) / (hi - lo) * 255, 0, 255)

    paths = {
        "open": str(prefix.with_name(prefix.name + "_open.png")),
        "closed": str(prefix.with_name(prefix.name + "_closed.png")),
        "diff": str(prefix.with_name(prefix.name + "_diff.png")),
        "depth": str(prefix.with_name(prefix.name + "_depth.png")),
    }
    imageio.imwrite(paths["open"], rgb_open)
    imageio.imwrite(paths["closed"], rgb_closed)
    imageio.imwrite(paths["diff"], diff_img)
    imageio.imwrite(paths["depth"], depth_img)
    return paths


def _manual_pixel_from_terminal(
    *,
    image_path: str,
    depth_m,
    K,
    patch_radius: int,
) -> dict | None:
    """Prompt for a manual pixel and backproject it, or return None to skip."""
    print(f"  manual fallback: inspect {image_path}")
    print("  enter pixel as row,col (for example 240,320), or press Enter to skip")
    text = input("  row,col> ").strip()
    if not text:
        return None
    try:
        row_s, col_s = text.replace(" ", "").split(",", 1)
        row = int(round(float(row_s)))
        col = int(round(float(col_s)))
    except Exception:
        print("  invalid pixel format; skipped")
        return None

    z = geom.sample_depth_patch(depth_m, col, row, radius=patch_radius)
    if not np.isfinite(z) or z <= 0:
        print(f"  no valid depth near ({row},{col}); skipped")
        return None
    p_cam = geom.backproject_pixel(K, col, row, z)
    return {
        "pixel": [float(row), float(col)],
        "depth_m": float(z),
        "area": int((2 * patch_radius + 1) ** 2),
        "xyz_cam": p_cam.tolist(),
        "manual": True,
    }


def _detect_tip_pixel_by_motion(
    rgb_open,
    rgb_closed,
    depth_m,
    K,
    *,
    diff_thresh: int = 18,
    min_area: int = 40,
    max_area: int = 40000,
) -> dict | None:
    """Detect gripper motion, preferring OpenCV but falling back gracefully."""
    try:
        return geom.detect_tip_pixel_by_motion(
            rgb_open,
            rgb_closed,
            depth_m,
            K,
            diff_thresh=diff_thresh,
            min_area=min_area,
            max_area=max_area,
        )
    except ModuleNotFoundError as exc:
        if exc.name != "cv2":
            raise
        return _detect_motion_blob_numpy(
            rgb_open,
            rgb_closed,
            depth_m,
            K,
            diff_thresh=diff_thresh,
            min_area=min_area,
            max_area=max_area,
        )


def _parse_csv_floats(text: str, *, expected: int, name: str) -> tuple[float, ...]:
    try:
        values = tuple(float(part.strip()) for part in text.split(",") if part.strip())
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"{name} must be comma-separated floats") from exc
    if len(values) != expected:
        raise argparse.ArgumentTypeError(f"{name} expects {expected} values, got {len(values)}")
    return values


def _candidate_poses(args: argparse.Namespace) -> list[list[float]]:
    xs = args.grid_x
    ys = args.grid_y
    zs = args.grid_z
    poses = [[float(x), float(y), float(z)] for z in zs for y in ys for x in xs]
    # Visit the center-ish pose first, then spread out. This makes early aborts
    # less likely to leave the robot at a corner of the grid.
    center = np.array([np.mean(xs), np.mean(ys), np.mean(zs)], dtype=np.float64)
    poses.sort(key=lambda p: float(np.linalg.norm(np.asarray(p) - center)))
    return poses


def _obs_camera(obs: dict, camera: str) -> tuple[np.ndarray, np.ndarray, dict]:
    frames = obs.get("frames") or {}
    depths = obs.get("depth") or {}
    metas = obs.get("camera_meta") or {}
    if camera not in frames:
        raise RuntimeError(f"camera {camera!r} missing from observation frames")
    if camera not in depths:
        raise RuntimeError(f"camera {camera!r} missing from observation depth maps")
    if camera not in metas:
        raise RuntimeError(f"camera {camera!r} missing from observation metadata")
    return (
        np.asarray(frames[camera], dtype=np.uint8),
        np.asarray(depths[camera], dtype=np.float32),
        dict(metas[camera]),
    )


def _tcp_point_from_pose(ee: dict, tcp_offset: tuple[float, float, float]) -> np.ndarray:
    xyz = np.asarray(ee["xyz"], dtype=np.float64)
    offset = np.asarray(tcp_offset, dtype=np.float64)
    if np.allclose(offset, 0.0):
        return xyz
    from scipy.spatial.transform import Rotation as R

    quat = ee.get("quat_xyzw")
    if quat is None:
        raise RuntimeError("--tcp-offset requires get_ee_pose to return quat_xyzw")
    return xyz + R.from_quat(np.asarray(quat, dtype=np.float64)).as_matrix() @ offset


def _detect_correspondence(
    *,
    client: SocketRpcClient,
    camera: str,
    tcp_offset: tuple[float, float, float],
    diff_thresh: int,
    min_area: int,
    max_area: int,
    settle_s: float,
    detect_retries: int,
    manual_on_fail: bool,
    manual_always: bool,
    manual_patch_radius: int,
    debug_dir: Path,
    pose_idx: int,
) -> dict[str, Any]:
    """Toggle gripper once and return one cam/base correspondence."""
    det = None
    rgb_closed = None
    depth = None
    meta = None
    debug_paths: dict[str, str] = {}
    attempts = max(1, int(detect_retries))
    for attempt in range(1, attempts + 1):
        client.call("env.open_gripper", timeout_s=30.0)
        time.sleep(settle_s)
        obs_open = client.call("env.get_obs", timeout_s=30.0)
        rgb_open, _, _ = _obs_camera(obs_open, camera)

        client.call("env.close_gripper", timeout_s=30.0)
        time.sleep(settle_s)
        obs_closed = client.call("env.get_obs", timeout_s=30.0)
        rgb_closed, depth, meta = _obs_camera(obs_closed, camera)

        debug_paths = _save_debug_images(
            debug_dir=debug_dir,
            pose_idx=pose_idx * 10 + attempt,
            camera=camera,
            rgb_open=rgb_open,
            rgb_closed=rgb_closed,
            depth_m=depth,
        )

        if not manual_always:
            det = _detect_tip_pixel_by_motion(
                rgb_open,
                rgb_closed,
                depth,
                np.asarray(meta["K"], dtype=np.float64),
                diff_thresh=diff_thresh,
                min_area=min_area,
                max_area=max_area,
            )
        if det is not None or manual_always:
            break

    ee = client.call("env.get_ee_pose", timeout_s=15.0)
    client.call("env.open_gripper", timeout_s=30.0)

    if (det is None or manual_always) and manual_on_fail:
        assert rgb_closed is not None and depth is not None and meta is not None
        det = _manual_pixel_from_terminal(
            image_path=debug_paths.get("closed", "<closed image unavailable>"),
            depth_m=depth,
            K=np.asarray(meta["K"], dtype=np.float64),
            patch_radius=manual_patch_radius,
        )
    if det is None:
        raise RuntimeError(
            "could not segment gripper motion in camera image; debug images: "
            + json.dumps(debug_paths)
        )

    return {
        "xyz_cam": np.asarray(det["xyz_cam"], dtype=np.float64),
        "xyz_base": _tcp_point_from_pose(ee, tcp_offset),
        "pixel": det["pixel"],
        "depth_m": float(det["depth_m"]),
        "area": int(det["area"]),
        "manual": bool(det.get("manual", False)),
        "debug_paths": debug_paths,
        "ee": ee,
        "camera_meta": meta,
    }


def _calibrate_scene(args: argparse.Namespace) -> int:
    if not args.yes and not args.no_save:
        print(
            "Refusing to move the robot without --yes. This calibration drives "
            "the TCP through a small 3-D grid and toggles the gripper."
        )
        return 2

    client = SocketRpcClient(args.host, args.port)
    meta_all = client.call("env.get_camera_meta", timeout_s=15.0)
    if args.camera not in meta_all:
        print(f"camera {args.camera!r} not available; found {sorted(meta_all)}")
        return 2
    camera_meta = meta_all[args.camera]
    serial = args.serial or camera_meta.get("serial")
    if not serial:
        print("could not determine camera serial; pass --serial")
        return 2

    poses = _candidate_poses(args)
    print(
        f"Calibrating fixed camera {args.camera!r} serial={serial} with up to "
        f"{len(poses)} candidate poses; target valid points={args.n_points}."
    )
    print("Clear the workspace. The gripper will move and open/close at each pose.")

    cam_pts: list[np.ndarray] = []
    base_pts: list[np.ndarray] = []
    records: list[dict[str, Any]] = []

    try:
        for idx, xyz in enumerate(poses, start=1):
            if len(cam_pts) >= args.n_points:
                break
            print(f"\n[{idx}/{len(poses)}] move_to {np.round(xyz, 3).tolist()}")
            move = client.call(
                "env.move_to",
                args=(xyz,),
                kwargs={"euler_xyz": _DEFAULT_DOWN_EULER, "gripper": "open"},
                timeout_s=120.0,
            )
            print(f"  move: reached={move.get('reached')} err={move.get('pos_error_m')} final={move.get('final_xyz')}")
            if move.get("error"):
                print(f"  skipped: {move['error']}")
                continue

            try:
                corr = _detect_correspondence(
                    client=client,
                    camera=args.camera,
                    tcp_offset=args.tcp_offset,
                    diff_thresh=args.diff_thresh,
                    min_area=args.min_area,
                    max_area=args.max_area,
                    settle_s=args.settle_s,
                    detect_retries=args.detect_retries,
                    manual_on_fail=args.manual_on_fail,
                    manual_always=args.manual_always,
                    manual_patch_radius=args.manual_patch_radius,
                    debug_dir=Path(args.debug_dir),
                    pose_idx=idx,
                )
            except Exception as exc:
                print(f"  detection failed: {exc}")
                continue

            cam_pts.append(corr["xyz_cam"])
            base_pts.append(corr["xyz_base"])
            records.append(
                {
                    "target_xyz": xyz,
                    "pixel": corr["pixel"],
                    "depth_m": round(corr["depth_m"], 4),
                    "area": corr["area"],
                    "manual": corr["manual"],
                    "debug_paths": corr["debug_paths"],
                    "xyz_cam": np.round(corr["xyz_cam"], 5).tolist(),
                    "xyz_base": np.round(corr["xyz_base"], 5).tolist(),
                    "move": move,
                }
            )
            print(
                "  captured: "
                f"pixel={np.round(corr['pixel'], 1).tolist()} "
                f"depth={corr['depth_m']:.3f}m area={corr['area']} "
                f"base={np.round(corr['xyz_base'], 3).tolist()} "
                f"manual={corr['manual']}"
            )
    finally:
        try:
            client.call("env.open_gripper", timeout_s=30.0)
        except Exception:
            pass

    if len(cam_pts) < 4:
        print(f"Calibration failed: need >=4 correspondences, got {len(cam_pts)}")
        print(json.dumps(records, indent=2, default=str))
        return 2

    cam_arr = np.asarray(cam_pts, dtype=np.float64)
    base_arr = np.asarray(base_pts, dtype=np.float64)
    T_base_cam, rmse, inliers = geom.ransac_kabsch(
        cam_arr,
        base_arr,
        thresh_m=args.ransac_thresh_m,
        iters=args.ransac_iters,
        min_inliers=min(4, len(cam_pts)),
        seed=args.seed,
    )
    n_inliers = int(inliers.sum())
    print(
        f"\nFit complete: used={len(cam_pts)} inliers={n_inliers} "
        f"RMSE={rmse * 1000:.1f} mm"
    )
    if rmse > franka_calib.MAX_ACCEPTABLE_RMSE_M:
        print(
            "WARNING: RMSE exceeds the loader acceptance gate "
            f"({franka_calib.MAX_ACCEPTABLE_RMSE_M * 1000:.0f} mm). "
            "The server will reject this calibration unless you rerun with better detections."
        )

    result = {
        "camera": args.camera,
        "serial": serial,
        "n_collected": len(cam_pts),
        "n_inliers": n_inliers,
        "rmse_m": rmse,
        "T_base_cam": T_base_cam,
        "records": records,
        "inlier_mask": inliers.tolist(),
        "tcp_offset": args.tcp_offset,
        "method": "franka_markerless_gripper_motion",
    }

    if args.no_save:
        print("Not saved (--no-save). T_base_cam:")
        print(json.dumps(result["T_base_cam"].tolist(), indent=2))
        return 0

    path = franka_calib.save_scene_extrinsic(
        serial,
        T_base_cam,
        K=np.asarray(camera_meta["K"], dtype=np.float64),
        rmse_m=rmse,
        num_points=len(cam_pts),
        camera=args.camera,
        n_inliers=n_inliers,
        inlier_mask=inliers.tolist(),
        method="franka_markerless_gripper_motion",
        tcp_offset=args.tcp_offset,
        correspondences=records,
    )
    print(f"Saved T_base_cam -> {path}")

    reload_result = client.call(
        "env.reload_camera_calibration",
        kwargs={"camera": args.camera},
        timeout_s=15.0,
    )
    print("Reload result:")
    print(json.dumps(reload_result, indent=2, default=str))
    return 0 if rmse <= franka_calib.MAX_ACCEPTABLE_RMSE_M else 1


def _explain_wrist() -> int:
    print(
        "Wrist camera calibration is hand-eye calibration (T_tcp_cam), not the "
        "same fixed-camera problem as the scene camera. A reliable automated "
        "script needs either a fixed fiducial/ChArUco/AprilTag board observed "
        "from multiple wrist poses, or a scene-calibrated 3-D reference target. "
        "This repository now supports loading/saving T_tcp_cam records, and "
        "back_project(camera='wrist') will return panda_link0 xyz once such a "
        "record exists."
    )
    return 0


def _build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, help="Franka env server transport port.")
    parser.add_argument(
        "--mode",
        choices=["scene-auto", "wrist-info"],
        default="scene-auto",
        help="scene-auto calibrates fixed scene T_base_cam; wrist-info explains T_tcp_cam requirements.",
    )
    parser.add_argument("--camera", default="scene", help="Camera name to calibrate (default: scene).")
    parser.add_argument("--serial", default=None, help="Override saved RealSense serial.")
    parser.add_argument("--n-points", type=int, default=8, help="Valid correspondences to collect.")
    parser.add_argument("--yes", action="store_true", help="Confirm robot motion.")
    parser.add_argument("--no-save", action="store_true", help="Fit but do not save/reload calibration.")
    parser.add_argument("--self-test", action="store_true", help="Run offline math/detection self-test.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--grid-x", type=lambda s: _parse_csv_floats(s, expected=3, name="grid-x"), default=_DEFAULT_GRID_X)
    parser.add_argument("--grid-y", type=lambda s: _parse_csv_floats(s, expected=3, name="grid-y"), default=_DEFAULT_GRID_Y)
    parser.add_argument("--grid-z", type=lambda s: _parse_csv_floats(s, expected=2, name="grid-z"), default=_DEFAULT_GRID_Z)
    parser.add_argument(
        "--tcp-offset",
        type=lambda s: _parse_csv_floats(s, expected=3, name="tcp-offset"),
        default=(0.0, 0.0, 0.0),
        help="Optional calibration point offset in TCP frame, meters (default: 0,0,0).",
    )
    parser.add_argument("--settle-s", type=float, default=0.4, help="Wait after gripper open/close captures.")
    parser.add_argument("--diff-thresh", type=int, default=10)
    parser.add_argument("--min-area", type=int, default=12)
    parser.add_argument("--max-area", type=int, default=40000)
    parser.add_argument("--detect-retries", type=int, default=2,
                        help="Open/close detection attempts per pose before fallback/skip.")
    parser.add_argument("--manual-on-fail", action="store_true",
                        help="When auto detection fails, prompt for row,col on the saved closed image.")
    parser.add_argument("--manual-always", action="store_true",
                        help="Always prompt for row,col instead of using auto detection.")
    parser.add_argument("--manual-patch-radius", type=int, default=3,
                        help="Depth median patch radius for manually clicked pixels.")
    parser.add_argument("--debug-dir", default="/tmp/franka_camera_calib_debug",
                        help="Directory for per-pose open/closed/diff/depth debug images.")
    parser.add_argument("--ransac-thresh-m", type=float, default=0.02)
    parser.add_argument("--ransac-iters", type=int, default=500)
    return parser


def main() -> int:
    args = _build_argparser().parse_args()
    if args.manual_always:
        args.manual_on_fail = True
    if args.self_test:
        return _self_test()
    if args.mode == "wrist-info":
        return _explain_wrist()
    if args.port is None:
        raise SystemExit("--port is required unless --self-test or --mode wrist-info")
    if args.n_points < 4:
        raise SystemExit("--n-points must be >= 4")
    return _calibrate_scene(args)


if __name__ == "__main__":
    raise SystemExit(main())
