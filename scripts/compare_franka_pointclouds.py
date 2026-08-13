"""Compare two FR3 base-frame point clouds produced by manual back-projection."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np


def _load_points(path: Path) -> np.ndarray:
    points = np.asarray(np.load(path), dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError(f"expected Nx3 point cloud at {path}, got {points.shape}")
    valid = np.isfinite(points).all(axis=1)
    return points[valid]


def _filter_bounds(
    points: np.ndarray,
    *,
    min_xyz: tuple[float, float, float] | None,
    max_xyz: tuple[float, float, float] | None,
) -> np.ndarray:
    mask = np.ones(points.shape[0], dtype=bool)
    if min_xyz is not None:
        mask &= (points >= np.asarray(min_xyz, dtype=np.float64)).all(axis=1)
    if max_xyz is not None:
        mask &= (points <= np.asarray(max_xyz, dtype=np.float64)).all(axis=1)
    return points[mask]


def _voxel_downsample(points: np.ndarray, voxel: float) -> np.ndarray:
    if voxel <= 0 or points.size == 0:
        return points
    keys = np.floor(points / voxel).astype(np.int64)
    _, inverse = np.unique(keys, axis=0, return_inverse=True)
    sums = np.zeros((int(inverse.max()) + 1, 3), dtype=np.float64)
    counts = np.bincount(inverse)
    np.add.at(sums, inverse, points)
    return sums / counts[:, None]


def _sample(points: np.ndarray, max_points: int, seed: int) -> np.ndarray:
    if max_points <= 0 or points.shape[0] <= max_points:
        return points
    rng = np.random.default_rng(seed)
    idx = rng.choice(points.shape[0], size=max_points, replace=False)
    return points[np.sort(idx)]


def _nearest_distances(src: np.ndarray, dst: np.ndarray) -> np.ndarray:
    try:
        from scipy.spatial import cKDTree
    except Exception as exc:  # pragma: no cover - hardware helper
        raise RuntimeError(
            "scipy is required for nearest-neighbor metrics. Install scipy or "
            "use the generated PLY/PNG visualizations only."
        ) from exc
    tree = cKDTree(dst)
    dists, _ = tree.query(src, k=1, workers=-1)
    return np.asarray(dists, dtype=np.float64)


def _stats(dists: np.ndarray) -> dict[str, float]:
    return {
        "count": int(dists.size),
        "mean_m": float(np.mean(dists)),
        "median_m": float(np.median(dists)),
        "rmse_m": float(np.sqrt(np.mean(dists * dists))),
        "p90_m": float(np.percentile(dists, 90)),
        "p95_m": float(np.percentile(dists, 95)),
        "p99_m": float(np.percentile(dists, 99)),
        "max_m": float(np.max(dists)),
    }


def _write_ply(
    path: Path,
    first: np.ndarray,
    second: np.ndarray,
    *,
    first_color: tuple[int, int, int] = (255, 70, 40),
    second_color: tuple[int, int, int] = (40, 130, 255),
) -> None:
    total = first.shape[0] + second.shape[0]
    with path.open("w", encoding="ascii") as f:
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write(f"element vertex {total}\n")
        f.write("property float x\n")
        f.write("property float y\n")
        f.write("property float z\n")
        f.write("property uchar red\n")
        f.write("property uchar green\n")
        f.write("property uchar blue\n")
        f.write("end_header\n")
        for points, color in ((first, first_color), (second, second_color)):
            r, g, b = color
            for x, y, z in points:
                f.write(f"{x:.6f} {y:.6f} {z:.6f} {r} {g} {b}\n")


def _write_projection_pngs(path_prefix: Path, first: np.ndarray, second: np.ndarray) -> dict[str, str]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    views = {
        "xy_top": (0, 1, "base x (m)", "base y (m)", "top view"),
        "xz_front": (0, 2, "base x (m)", "base z (m)", "front view"),
        "yz_side": (1, 2, "base y (m)", "base z (m)", "side view"),
    }
    outputs: dict[str, str] = {}
    for name, (i, j, xlabel, ylabel, title) in views.items():
        fig, ax = plt.subplots(figsize=(8, 7), dpi=160)
        ax.scatter(first[:, i], first[:, j], s=0.35, c="#ff4628", label="first")
        ax.scatter(second[:, i], second[:, j], s=0.35, c="#2882ff", label="second")
        ax.set_title(f"base-frame point clouds, {title}")
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_aspect("equal", adjustable="box")
        ax.grid(True, linewidth=0.3, alpha=0.4)
        ax.legend(markerscale=8)
        fig.tight_layout()
        path = path_prefix.with_name(f"{path_prefix.name}_{name}.png")
        fig.savefig(path)
        plt.close(fig)
        outputs[name] = str(path)
    return outputs


def _bounds(points: np.ndarray) -> dict[str, list[float]]:
    return {
        "min": [float(x) for x in np.min(points, axis=0)],
        "max": [float(x) for x in np.max(points, axis=0)],
    }


def _parse_xyz(value: str | None) -> tuple[float, float, float] | None:
    if value is None or value == "":
        return None
    parts = [float(x.strip()) for x in value.split(",")]
    if len(parts) != 3:
        raise argparse.ArgumentTypeError("expected x,y,z")
    return parts[0], parts[1], parts[2]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare two manual back-projected point clouds in fr3_link0."
    )
    parser.add_argument("--first", required=True, help="First Nx3 .npy point cloud.")
    parser.add_argument("--second", required=True, help="Second Nx3 .npy point cloud.")
    parser.add_argument("--first-name", default="first")
    parser.add_argument("--second-name", default="second")
    parser.add_argument("--output-dir", default="pointcloud_compare")
    parser.add_argument("--voxel", type=float, default=0.005, help="Voxel size in meters.")
    parser.add_argument("--max-points", type=int, default=150000)
    parser.add_argument("--metric-max-points", type=int, default=200000)
    parser.add_argument("--min-xyz", type=_parse_xyz, default=None, help="Optional x,y,z crop.")
    parser.add_argument("--max-xyz", type=_parse_xyz, default=None, help="Optional x,y,z crop.")
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    first_raw = _load_points(Path(args.first).expanduser())
    second_raw = _load_points(Path(args.second).expanduser())
    first_crop = _filter_bounds(first_raw, min_xyz=args.min_xyz, max_xyz=args.max_xyz)
    second_crop = _filter_bounds(second_raw, min_xyz=args.min_xyz, max_xyz=args.max_xyz)
    first = _voxel_downsample(first_crop, args.voxel)
    second = _voxel_downsample(second_crop, args.voxel)

    first_metric = _sample(first, args.metric_max_points, seed=1)
    second_metric = _sample(second, args.metric_max_points, seed=2)
    first_to_second = _nearest_distances(first_metric, second)
    second_to_first = _nearest_distances(second_metric, first)

    first_vis = _sample(first, args.max_points, seed=3)
    second_vis = _sample(second, args.max_points, seed=4)
    ply_path = output_dir / "combined_base_clouds.ply"
    _write_ply(ply_path, first_vis, second_vis)
    pngs = _write_projection_pngs(output_dir / "combined_base_clouds", first_vis, second_vis)

    summary: dict[str, Any] = {
        "inputs": {
            "first": str(Path(args.first).expanduser().resolve()),
            "second": str(Path(args.second).expanduser().resolve()),
            "first_name": args.first_name,
            "second_name": args.second_name,
        },
        "processing": {
            "voxel_m": args.voxel,
            "crop_min_xyz": args.min_xyz,
            "crop_max_xyz": args.max_xyz,
            "max_points_visualized_per_cloud": args.max_points,
        },
        "counts": {
            "first_raw": int(first_raw.shape[0]),
            "second_raw": int(second_raw.shape[0]),
            "first_after_crop": int(first_crop.shape[0]),
            "second_after_crop": int(second_crop.shape[0]),
            "first_after_voxel": int(first.shape[0]),
            "second_after_voxel": int(second.shape[0]),
        },
        "bounds_base_m": {
            "first": _bounds(first),
            "second": _bounds(second),
        },
        "nearest_neighbor_error": {
            f"{args.first_name}_to_{args.second_name}": _stats(first_to_second),
            f"{args.second_name}_to_{args.first_name}": _stats(second_to_first),
            "symmetric_mean_m": float(
                (np.mean(first_to_second) + np.mean(second_to_first)) / 2.0
            ),
            "symmetric_rmse_m": float(
                np.sqrt(
                    (np.mean(first_to_second * first_to_second)
                    + np.mean(second_to_first * second_to_first))
                    / 2.0
                )
            ),
        },
        "outputs": {
            "combined_ply": str(ply_path),
            "projection_pngs": pngs,
        },
    }
    summary_path = output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
