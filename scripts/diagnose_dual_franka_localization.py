#!/usr/bin/env python3
"""Diagnose base-camera pixel back-projection for Dual-Franka.

The script marks selected base-image pixels, reports camera-frame and
right_base-frame coordinates, and saves a JSON report. It can either use an
existing episode snapshot or ask the running dual_franka env server for a fresh
read-only ``dump_state`` snapshot.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

from robots.dual_franka import perception as dual_perception
from rpent.tools.state import EnvState


def _parse_point(text: str) -> tuple[int, int]:
    raw = text.replace(":", ",").split(",")
    if len(raw) != 2:
        raise argparse.ArgumentTypeError("point must be ROW,COL")
    return int(raw[0]), int(raw[1])


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Mark base-camera pixels and compare their camera-frame and "
            "right_base back-projected coordinates."
        )
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Dual-Franka episode/bootstrap output directory with latest_state.json.",
    )
    parser.add_argument(
        "--point",
        action="append",
        type=_parse_point,
        default=[],
        metavar="ROW,COL",
        help="Base image pixel selected by the agent or operator. Repeatable.",
    )
    parser.add_argument(
        "--from-agent-log",
        action="store_true",
        help="Extract pixels from the episode's Codex stream back_project_base_pixel calls.",
    )
    parser.add_argument(
        "--step",
        type=int,
        default=None,
        help="Snapshot step to inspect. Default uses latest_state.json.",
    )
    parser.add_argument(
        "--window-radius",
        type=int,
        default=5,
        help="Median depth window radius around each selected pixel.",
    )
    parser.add_argument(
        "--capture-rpc",
        action="store_true",
        help="Ask the running env server for a fresh read-only dump_state first.",
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=5556)
    parser.add_argument("--timeout-s", type=float, default=120.0)
    parser.add_argument(
        "--out-dir",
        default=None,
        help="Directory for annotated image/report. Default: OUTPUT_DIR/localization_diagnostic.",
    )
    return parser


def _capture_snapshot(
    output_dir: Path,
    host: str,
    port: int,
    timeout_s: float,
) -> dict[str, Any]:
    from rpent.utils.socket_rpc import SocketRpcClient

    client = SocketRpcClient(host, port)
    observation = client.call("env.get_observation", timeout_s=timeout_s)
    robot_state = client.call("env.get_robot_state", timeout_s=timeout_s)
    camera_meta = client.call("env.get_camera_metadata", timeout_s=timeout_s)
    state = EnvState(output_dir)
    with state.record_step(state=robot_state) as step:
        for key, artifact in (
            ("d455_images", "d455.png"),
            ("d455_depths", "d455_depth.npy"),
        ):
            if observation.get(key) is not None:
                state.save(artifact, observation[key], step=step)
        extra_images = observation.get("extra_view_images")
        if extra_images is not None:
            images = np.asarray(extra_images)
            if images.ndim == 4 and images.shape[0] >= 1:
                state.save("base.png", images[0], step=step)
        extra_depths = observation.get("extra_view_depths")
        if extra_depths is not None:
            depths = np.asarray(extra_depths)
            if depths.ndim == 3 and depths.shape[0] >= 1:
                state.save("base_depth.npy", depths[0], step=step)
        if camera_meta is not None:
            state.save("camera_meta.json", camera_meta, step=step)
    return dual_perception.load_snapshot(output_dir)


def _load_json(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text(errors="replace"))
    if not isinstance(data, dict):
        raise ValueError(f"expected JSON object: {path}")
    return data


def _extract_points_from_agent_log(output_dir: Path) -> list[tuple[int, int]]:
    points: list[tuple[int, int]] = []
    stream_files = sorted(output_dir.glob("codex_*.stream.jsonl"))
    for path in stream_files:
        for line in path.read_text(errors="replace").splitlines():
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            item = row.get("item") if isinstance(row, dict) else None
            if not isinstance(item, dict):
                continue
            if item.get("type") != "mcp_tool_call":
                continue
            if item.get("tool") != "back_project_base_pixel":
                continue
            args = item.get("arguments") or {}
            if not isinstance(args, dict):
                continue
            if "row" in args and "col" in args:
                points.append((int(args["row"]), int(args["col"])))
    return _dedupe_points(points)


def _dedupe_points(points: list[tuple[int, int]]) -> list[tuple[int, int]]:
    seen: set[tuple[int, int]] = set()
    out: list[tuple[int, int]] = []
    for point in points:
        if point not in seen:
            out.append(point)
            seen.add(point)
    return out


def _base_image_path(snapshot: dict[str, Any]) -> Path:
    images = ((snapshot.get("artifacts") or {}).get("images") or {})
    path = images.get("base") or images.get("extra_0")
    if not path:
        raise ValueError("base image artifact not found in snapshot")
    return Path(path)


def _base_depth_path(snapshot: dict[str, Any]) -> Path:
    depths = ((snapshot.get("artifacts") or {}).get("depths") or {})
    path = depths.get("base") or depths.get("extra_0")
    if not path:
        raise ValueError("base depth artifact not found in snapshot")
    return Path(path)


def _depth_patch_stats(
    depth: np.ndarray,
    *,
    row: int,
    col: int,
    radius: int,
) -> dict[str, Any]:
    r0 = max(0, row - radius)
    r1 = min(depth.shape[0], row + radius + 1)
    c0 = max(0, col - radius)
    c1 = min(depth.shape[1], col + radius + 1)
    patch = np.asarray(depth[r0:r1, c0:c1], dtype=np.float32)
    valid = patch[np.isfinite(patch) & (patch > 0.0)]
    raw_depth = float(depth[row, col]) if np.isfinite(depth[row, col]) else math.nan
    stats: dict[str, Any] = {
        "window_bounds_rc": [int(r0), int(r1), int(c0), int(c1)],
        "window_shape": list(patch.shape),
        "raw_depth_at_pixel_m": _round_float(raw_depth),
        "valid_pixels": int(valid.size),
        "total_pixels": int(patch.size),
    }
    if valid.size:
        stats.update(
            {
                "min_m": _round_float(float(valid.min())),
                "max_m": _round_float(float(valid.max())),
                "mean_m": _round_float(float(valid.mean())),
                "median_m": _round_float(float(np.median(valid))),
                "std_m": _round_float(float(valid.std())),
            }
        )
    return stats


def _round_float(value: float, ndigits: int = 6) -> float | None:
    if not math.isfinite(float(value)):
        return None
    return round(float(value), ndigits)


def _annotate_image(
    *,
    image_path: Path,
    points: list[dict[str, Any]],
    output_path: Path,
) -> None:
    import cv2

    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"failed to read image: {image_path}")
    for index, item in enumerate(points, 1):
        row, col = item["pixel_rc"]
        ok = bool(item.get("ok"))
        color = (0, 255, 0) if ok else (0, 0, 255)
        cv2.drawMarker(
            image,
            (int(col), int(row)),
            color,
            markerType=cv2.MARKER_CROSS,
            markerSize=28,
            thickness=2,
            line_type=cv2.LINE_AA,
        )
        cv2.circle(image, (int(col), int(row)), 12, color, 2, lineType=cv2.LINE_AA)
        label = f"P{index} ({row},{col})"
        if ok:
            xyz = item.get("point_right_base_xyz")
            label += f" rb={_short_xyz(xyz)}"
        cv2.putText(
            image,
            label,
            (min(int(col) + 14, max(0, image.shape[1] - 240)), max(20, int(row) - 14)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            color,
            1,
            cv2.LINE_AA,
        )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(output_path), image):
        raise ValueError(f"failed to write annotated image: {output_path}")


def _short_xyz(value: Any) -> str:
    if not isinstance(value, list) or len(value) < 3:
        return "n/a"
    return f"{value[0]:.3f},{value[1]:.3f},{value[2]:.3f}"


def _diagnose_point(
    *,
    output_dir: Path,
    snapshot: dict[str, Any],
    depth: np.ndarray,
    row: int,
    col: int,
    step: int | None,
    window_radius: int,
) -> dict[str, Any]:
    item: dict[str, Any] = {
        "ok": False,
        "pixel_rc": [int(row), int(col)],
        "pixel_xy": [int(col), int(row)],
    }
    try:
        projection = dual_perception.back_project_base_pixel(
            output_dir,
            row=int(row),
            col=int(col),
            step=step,
            window_radius=window_radius,
        )
        item.update(
            {
                "ok": True,
                "depth_m": projection.get("depth_m"),
                "point_camera_xyz": projection.get("point_camera_xyz"),
                "point_right_base_xyz": projection.get("point_xyz"),
                "distance_camera_origin_m": _norm(projection.get("point_camera_xyz")),
                "distance_right_base_origin_m": _norm(projection.get("point_xyz")),
                "delta_right_tcp_to_point_xyz": projection.get("delta_right_tcp_to_point_xyz"),
                "delta_left_tcp_to_point_xyz": projection.get("delta_left_tcp_to_point_xyz"),
            }
        )
    except Exception as exc:
        item["error"] = str(exc)
        item["error_type"] = type(exc).__name__
    if 0 <= row < depth.shape[0] and 0 <= col < depth.shape[1]:
        item["depth_patch"] = _depth_patch_stats(
            depth,
            row=int(row),
            col=int(col),
            radius=window_radius,
        )
    else:
        item["depth_patch"] = {"error": f"pixel out of depth bounds {list(depth.shape)}"}
    item["snapshot_step"] = snapshot.get("step_idx")
    return item


def _norm(value: Any) -> float | None:
    if not isinstance(value, list) or len(value) < 3:
        return None
    arr = np.asarray(value[:3], dtype=np.float64)
    return round(float(np.linalg.norm(arr)), 6)


def main() -> int:
    args = _build_parser().parse_args()
    output_dir = Path(args.output_dir).expanduser().resolve()
    if args.capture_rpc:
        _capture_snapshot(output_dir, args.host, args.port, args.timeout_s)

    points = list(args.point)
    if args.from_agent_log:
        points.extend(_extract_points_from_agent_log(output_dir))
    points = _dedupe_points(points)
    if not points:
        raise SystemExit("provide --point ROW,COL or --from-agent-log")

    snapshot = dual_perception.load_snapshot(output_dir, step=args.step)
    image_path = _base_image_path(snapshot)
    depth_path = _base_depth_path(snapshot)
    depth = np.asarray(np.load(depth_path), dtype=np.float32).squeeze()
    if depth.ndim != 2:
        raise ValueError(f"expected 2D depth, got {depth.shape}: {depth_path}")

    out_dir = Path(args.out_dir).expanduser().resolve() if args.out_dir else output_dir / "localization_diagnostic"
    report_points = [
        _diagnose_point(
            output_dir=output_dir,
            snapshot=snapshot,
            depth=depth,
            row=row,
            col=col,
            step=args.step,
            window_radius=max(0, int(args.window_radius)),
        )
        for row, col in points
    ]

    step_label = snapshot.get("step_idx")
    annotated_path = out_dir / f"base_localization_step_{step_label}.png"
    report_path = out_dir / f"base_localization_step_{step_label}.json"
    _annotate_image(image_path=image_path, points=report_points, output_path=annotated_path)

    calibration = dual_perception.load_calibration_bundle()
    report = {
        "ok": True,
        "output_dir": str(output_dir),
        "snapshot_step": step_label,
        "image_path": str(image_path),
        "depth_path": str(depth_path),
        "annotated_image": str(annotated_path),
        "calibration_path": str(dual_perception.DEFAULT_CALIBRATION_BUNDLE_PATH),
        "coordinate_convention": (
            "point_camera_xyz is in the base RealSense color optical frame. "
            "point_right_base_xyz is transformed by T_right_base_base_camera "
            "and is the agent world coordinate."
        ),
        "T_right_base_base_camera": calibration.get("base_camera", {}).get("transformation"),
        "points": report_points,
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
