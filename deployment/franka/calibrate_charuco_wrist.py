#!/usr/bin/env python3
"""ChArUco-based wrist-camera hand-eye calibration for Franka.

This estimates ``T_tcp_cam`` for the wrist camera. The ChArUco board must be
fixed in the scene while the wrist camera observes it from multiple robot poses.
The script detects the board in the wrist camera, reads the live TCP pose, then
uses OpenCV hand-eye calibration to solve camera-in-TCP.

Typical workflow:

1. Start the Franka env server.
2. Place the printed ChArUco board flat and rigid on the table.
3. Move the wrist camera so the board is visible in the wrist image.
4. Check detection:

       python deployment/franka/calibrate_charuco_wrist.py --port 5599 check

5. Calibrate with a small automatic orbit around the current pose:

       python deployment/franka/calibrate_charuco_wrist.py --port 5599 calibrate --yes

The scene camera is different: a ChArUco board gives ``T_scene_cam_board`` but
not ``T_base_scene_cam`` unless the board pose in ``panda_link0`` is known. Use
``auto_calibrate_cameras.py`` for markerless scene-to-base calibration, or add a
known board pose / touch-corner workflow for scene calibration.
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
from scipy.spatial.transform import Rotation as R

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from deployment.franka import calibration as franka_calib  # noqa: E402
from deployment.lerobot import geometry as geom  # noqa: E402
from rpent.rpc_driver.socket import SocketRpcClient  # noqa: E402

_DEFAULT_BOARD_SPEC = (
    _REPO_ROOT / "resources" / "franka" / "calibration_boards" / "franka_charuco_7x5_25mm.json"
)
_DEFAULT_OFFSETS = (
    ([0.0, 0.0, 0.0], [0.0, 0.0, 0.0]),
    ([0.025, 0.0, 0.0], [0.0, 0.0, 12.0]),
    ([-0.025, 0.0, 0.0], [0.0, 0.0, -12.0]),
    ([0.0, 0.025, 0.0], [0.0, 10.0, 0.0]),
    ([0.0, -0.025, 0.0], [0.0, -10.0, 0.0]),
    ([0.0, 0.0, 0.025], [8.0, 0.0, 0.0]),
    ([0.0, 0.0, -0.015], [-8.0, 0.0, 0.0]),
    ([0.02, 0.02, 0.015], [6.0, -8.0, 8.0]),
    ([-0.02, -0.02, 0.015], [-6.0, 8.0, -8.0]),
)


def _require_cv2_aruco():
    try:
        import cv2
    except ModuleNotFoundError as exc:
        raise SystemExit(
            "Missing dependency: cv2. Install with `uv pip install -e .[calibration]` "
            "or run with an environment that has opencv-contrib-python-headless."
        ) from exc
    if not hasattr(cv2, "aruco"):
        raise SystemExit("Installed cv2 lacks aruco; install opencv-contrib-python-headless.")
    return cv2


def _load_board_spec(path: Path) -> dict[str, Any]:
    spec = json.loads(path.read_text())
    required = ["dictionary", "squares_x", "squares_y", "square_length_m", "marker_length_m"]
    missing = [key for key in required if key not in spec]
    if missing:
        raise SystemExit(f"board spec missing keys: {missing}")
    return spec


def _aruco_dictionary(cv2, dictionary_name: str):
    aruco = cv2.aruco
    key = dictionary_name.upper()
    if not key.startswith("DICT_"):
        key = f"DICT_{key}"
    if not hasattr(aruco, key):
        raise SystemExit(f"OpenCV does not know ArUco dictionary {dictionary_name!r}")
    return aruco.getPredefinedDictionary(getattr(aruco, key))


def _make_board(cv2, spec: dict[str, Any]):
    aruco = cv2.aruco
    dictionary = _aruco_dictionary(cv2, spec["dictionary"])
    squares = (int(spec["squares_x"]), int(spec["squares_y"]))
    square = float(spec["square_length_m"])
    marker = float(spec["marker_length_m"])
    try:
        return aruco.CharucoBoard(squares, square, marker, dictionary)
    except Exception:
        return aruco.CharucoBoard_create(squares[0], squares[1], square, marker, dictionary)


def _pose_to_matrix(xyz, quat_xyzw) -> np.ndarray:
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = R.from_quat(np.asarray(quat_xyzw, dtype=np.float64)).as_matrix()
    T[:3, 3] = np.asarray(xyz, dtype=np.float64)
    return T


def _pose_from_rvec_tvec(cv2, rvec, tvec) -> np.ndarray:
    T = np.eye(4, dtype=np.float64)
    R_cam_board, _ = cv2.Rodrigues(np.asarray(rvec, dtype=np.float64))
    T[:3, :3] = R_cam_board
    T[:3, 3] = np.asarray(tvec, dtype=np.float64).reshape(3)
    return T


def _detect_charuco_pose(cv2, image, K, dist_coeffs, board) -> dict | None:
    aruco = cv2.aruco
    gray = cv2.cvtColor(np.asarray(image, dtype=np.uint8), cv2.COLOR_RGB2GRAY)
    marker_corners = []
    marker_ids = None

    if hasattr(aruco, "CharucoDetector"):
        detector = aruco.CharucoDetector(board)
        charuco_corners, charuco_ids, marker_corners, marker_ids = detector.detectBoard(gray)
        count = 0 if charuco_ids is None else len(charuco_ids)
    else:
        params = aruco.DetectorParameters()
        try:
            detector = aruco.ArucoDetector(board.getDictionary(), params)
            marker_corners, marker_ids, _ = detector.detectMarkers(gray)
        except Exception:
            marker_corners, marker_ids, _ = aruco.detectMarkers(
                gray, board.getDictionary(), parameters=params
            )
        if marker_ids is None or len(marker_ids) < 2:
            return None

        try:
            count, charuco_corners, charuco_ids = aruco.interpolateCornersCharuco(
                marker_corners,
                marker_ids,
                gray,
                board,
                cameraMatrix=K,
                distCoeffs=dist_coeffs,
            )
        except TypeError:
            count, charuco_corners, charuco_ids = aruco.interpolateCornersCharuco(
                marker_corners, marker_ids, gray, board, K, dist_coeffs
            )

    if charuco_ids is None or int(count) < 6:
        return None

    if hasattr(aruco, "estimatePoseCharucoBoard"):
        rvec = np.zeros((3, 1), dtype=np.float64)
        tvec = np.zeros((3, 1), dtype=np.float64)
        ok, rvec, tvec = aruco.estimatePoseCharucoBoard(
            charuco_corners,
            charuco_ids,
            board,
            K,
            dist_coeffs,
            rvec,
            tvec,
        )
    else:
        obj_points, img_points = board.matchImagePoints(charuco_corners, charuco_ids)
        ok, rvec, tvec = cv2.solvePnP(
            obj_points,
            img_points,
            K,
            dist_coeffs,
            flags=cv2.SOLVEPNP_ITERATIVE,
        )
    if not ok:
        return None
    return {
        "T_cam_board": _pose_from_rvec_tvec(cv2, rvec, tvec),
        "n_markers": 0 if marker_ids is None else int(len(marker_ids)),
        "n_corners": int(count),
        "rvec": np.asarray(rvec, dtype=np.float64).reshape(3).tolist(),
        "tvec": np.asarray(tvec, dtype=np.float64).reshape(3).tolist(),
    }


def _draw_detection(cv2, image, K, dist_coeffs, board, out_path: Path) -> None:
    aruco = cv2.aruco
    canvas = np.asarray(image, dtype=np.uint8).copy()
    gray = cv2.cvtColor(canvas, cv2.COLOR_RGB2GRAY)
    charuco_corners = None
    charuco_ids = None
    if hasattr(aruco, "CharucoDetector"):
        detector = aruco.CharucoDetector(board)
        charuco_corners, charuco_ids, marker_corners, marker_ids = detector.detectBoard(gray)
    else:
        params = aruco.DetectorParameters()
        try:
            detector = aruco.ArucoDetector(board.getDictionary(), params)
            marker_corners, marker_ids, _ = detector.detectMarkers(gray)
        except Exception:
            marker_corners, marker_ids, _ = aruco.detectMarkers(
                gray, board.getDictionary(), parameters=params
            )
    if marker_ids is not None and len(marker_ids) > 0:
        aruco.drawDetectedMarkers(canvas, marker_corners, marker_ids)
    if charuco_ids is not None and len(charuco_ids) > 0:
        aruco.drawDetectedCornersCharuco(canvas, charuco_corners, charuco_ids)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    imageio.imwrite(out_path, canvas)


def _capture(client: SocketRpcClient, camera: str) -> tuple[dict, np.ndarray, dict]:
    obs = client.call("env.get_obs", timeout_s=30.0)
    frames = obs.get("frames") or {}
    meta = obs.get("camera_meta") or {}
    if camera not in frames:
        raise RuntimeError(f"camera {camera!r} missing from observation")
    if camera not in meta:
        raise RuntimeError(f"camera {camera!r} metadata missing")
    ee = client.call("env.get_ee_pose", timeout_s=15.0)
    return ee, np.asarray(frames[camera], dtype=np.uint8), dict(meta[camera])


def _check(args: argparse.Namespace) -> int:
    cv2 = _require_cv2_aruco()
    board = _make_board(cv2, _load_board_spec(args.board_spec))
    client = SocketRpcClient(args.host, args.port)
    ee, image, meta = _capture(client, args.camera)
    K = np.asarray(meta["K"], dtype=np.float64)
    dist = np.asarray(meta.get("dist_coeffs") or np.zeros(5), dtype=np.float64)
    det = _detect_charuco_pose(cv2, image, K, dist, board)
    out_path = Path(args.debug_dir) / f"{args.camera}_charuco_check.png"
    _draw_detection(cv2, image, K, dist, board, out_path)
    print(json.dumps({
        "camera": args.camera,
        "serial": meta.get("serial"),
        "debug_image": str(out_path),
        "detected": det is not None,
        "n_markers": None if det is None else det["n_markers"],
        "n_corners": None if det is None else det["n_corners"],
        "tcp_xyz": ee.get("xyz"),
    }, indent=2))
    return 0 if det is not None else 2


def _target_pose(start: dict, dxyz, drpy_deg) -> tuple[list[float], list[float]]:
    xyz = np.asarray(start["xyz"], dtype=np.float64) + np.asarray(dxyz, dtype=np.float64)
    euler = np.asarray(start["euler_xyz"], dtype=np.float64) + np.radians(np.asarray(drpy_deg, dtype=np.float64))
    return xyz.tolist(), euler.tolist()


def _collect_samples(
    args: argparse.Namespace,
) -> tuple[
    list[np.ndarray],
    list[np.ndarray],
    list[np.ndarray],
    list[np.ndarray],
    list[dict],
]:
    cv2 = _require_cv2_aruco()
    board = _make_board(cv2, _load_board_spec(args.board_spec))
    client = SocketRpcClient(args.host, args.port)
    start = client.call("env.get_ee_pose", timeout_s=15.0)
    start_xyz = start["xyz"]
    start_quat = start["quat_xyzw"]

    R_gripper2base: list[np.ndarray] = []
    t_gripper2base: list[np.ndarray] = []
    R_target2cam: list[np.ndarray] = []
    t_target2cam: list[np.ndarray] = []
    records: list[dict] = []

    try:
        for idx, (dxyz, drpy) in enumerate(_DEFAULT_OFFSETS, start=1):
            if len(records) >= args.n_samples:
                break
            target_xyz, target_euler = _target_pose(start, dxyz, drpy)
            print(f"\n[{idx}] target_xyz={np.round(target_xyz, 3).tolist()} drpy={drpy}")
            move = client.call(
                "env.move_to",
                args=(target_xyz,),
                kwargs={"euler_xyz": target_euler, "gripper": None},
                timeout_s=120.0,
            )
            print(f"  move: reached={move.get('reached')} err={move.get('pos_error_m')} final={move.get('final_xyz')}")
            time.sleep(args.settle_s)
            ee, image, meta = _capture(client, args.camera)
            K = np.asarray(meta["K"], dtype=np.float64)
            dist = np.asarray(meta.get("dist_coeffs") or np.zeros(5), dtype=np.float64)
            det = _detect_charuco_pose(cv2, image, K, dist, board)
            debug_path = Path(args.debug_dir) / f"{args.camera}_charuco_{idx:02d}.png"
            _draw_detection(cv2, image, K, dist, board, debug_path)
            if det is None:
                print(f"  detection failed (debug: {debug_path})")
                continue

            T_base_tcp = _pose_to_matrix(ee["xyz"], ee["quat_xyzw"])
            T_cam_board = det["T_cam_board"]
            R_gripper2base.append(T_base_tcp[:3, :3])
            t_gripper2base.append(T_base_tcp[:3, 3].reshape(3, 1))
            R_target2cam.append(T_cam_board[:3, :3])
            t_target2cam.append(T_cam_board[:3, 3].reshape(3, 1))
            records.append({
                "idx": idx,
                "target_xyz": target_xyz,
                "target_euler": target_euler,
                "tcp_xyz": ee["xyz"],
                "n_markers": det["n_markers"],
                "n_corners": det["n_corners"],
                "tvec": det["tvec"],
                "debug_image": str(debug_path),
                "move": move,
            })
            print(f"  captured: markers={det['n_markers']} corners={det['n_corners']} tvec={np.round(det['tvec'], 3).tolist()}")
    finally:
        try:
            client.call(
                "env.move_to",
                args=(start_xyz,),
                kwargs={"quat_xyzw": start_quat, "gripper": None},
                timeout_s=120.0,
            )
        except Exception as exc:
            print(f"warning: failed to return to start pose: {exc}")

    return (
        [np.asarray(r, dtype=np.float64) for r in R_gripper2base],
        [np.asarray(t, dtype=np.float64) for t in t_gripper2base],
        [np.asarray(r, dtype=np.float64) for r in R_target2cam],
        [np.asarray(t, dtype=np.float64) for t in t_target2cam],
        records,
    )


def _calibrate(args: argparse.Namespace) -> int:
    if not args.yes:
        print("Refusing to move the robot without --yes.")
        return 2
    cv2 = _require_cv2_aruco()
    board_spec = _load_board_spec(args.board_spec)
    client = SocketRpcClient(args.host, args.port)
    meta = client.call("env.get_camera_meta", timeout_s=15.0).get(args.camera)
    if not meta:
        print(f"camera {args.camera!r} not available")
        return 2
    serial = args.serial or meta.get("serial")
    if not serial:
        print("could not determine wrist camera serial; pass --serial")
        return 2

    R_g2b, t_g2b, R_t2c, t_t2c, records = _collect_samples(args)
    if len(records) < 5:
        print(f"Need at least 5 valid board detections; got {len(records)}")
        return 2

    R_cam2tcp, t_cam2tcp = cv2.calibrateHandEye(
        R_g2b,
        t_g2b,
        R_t2c,
        t_t2c,
        method=cv2.CALIB_HAND_EYE_TSAI,
    )
    T_tcp_cam = np.eye(4, dtype=np.float64)
    T_tcp_cam[:3, :3] = np.asarray(R_cam2tcp, dtype=np.float64)
    T_tcp_cam[:3, 3] = np.asarray(t_cam2tcp, dtype=np.float64).reshape(3)

    residuals = []
    target_points_base = []
    for Rb, tb, Rc, tc in zip(R_g2b, t_g2b, R_t2c, t_t2c):
        T_base_tcp = np.eye(4)
        T_base_tcp[:3, :3] = Rb
        T_base_tcp[:3, 3] = tb.reshape(3)
        T_cam_board = np.eye(4)
        T_cam_board[:3, :3] = Rc
        T_cam_board[:3, 3] = tc.reshape(3)
        T_base_board = T_base_tcp @ T_tcp_cam @ T_cam_board
        target_points_base.append(T_base_board[:3, 3])
    target_points_base = np.asarray(target_points_base)
    center = target_points_base.mean(axis=0)
    residuals = np.linalg.norm(target_points_base - center, axis=1)
    rmse = float(np.sqrt(np.mean(np.square(residuals))))

    print(f"\nHand-eye fit: samples={len(records)} board-position RMSE={rmse * 1000:.1f} mm")
    print("T_tcp_cam:")
    print(json.dumps(T_tcp_cam.tolist(), indent=2))

    if args.no_save:
        return 0

    path = franka_calib.save_wrist_extrinsic(
        serial,
        T_tcp_cam,
        K=np.asarray(meta["K"], dtype=np.float64),
        rmse_m=rmse,
        num_points=len(records),
        camera=args.camera,
        method="charuco_hand_eye_tsai",
        board_spec=board_spec,
        records=records,
        board_position_residuals_m=residuals.tolist(),
    )
    print(f"Saved T_tcp_cam -> {path}")
    reload_result = client.call("env.reload_camera_calibration", kwargs={"camera": args.camera}, timeout_s=15.0)
    print("Reload result:")
    print(json.dumps(reload_result, indent=2, default=str))
    return 0 if rmse <= franka_calib.MAX_ACCEPTABLE_RMSE_M else 1


def _build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--camera", default="wrist")
    parser.add_argument("--serial", default=None)
    parser.add_argument("--board-spec", type=Path, default=_DEFAULT_BOARD_SPEC)
    parser.add_argument("--debug-dir", type=Path, default=Path("/tmp/franka_charuco_wrist_debug"))
    sub = parser.add_subparsers(dest="cmd", required=True)

    sub.add_parser("check", help="Capture one wrist frame and report board detection.")

    calib = sub.add_parser("calibrate", help="Move through a small orbit and solve T_tcp_cam.")
    calib.add_argument("--yes", action="store_true")
    calib.add_argument("--no-save", action="store_true")
    calib.add_argument("--n-samples", type=int, default=8)
    calib.add_argument("--settle-s", type=float, default=0.5)
    return parser


def main() -> int:
    args = _build_argparser().parse_args()
    if args.cmd == "check":
        return _check(args)
    if args.cmd == "calibrate":
        return _calibrate(args)
    raise SystemExit(f"unknown command: {args.cmd}")


if __name__ == "__main__":
    raise SystemExit(main())
