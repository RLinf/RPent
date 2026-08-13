"""Render Franka saved metric depth arrays into false-color PNGs."""
from __future__ import annotations

import argparse
from pathlib import Path

import imageio.v2 as imageio
import numpy as np


def _depth_files(root: Path) -> list[Path]:
    if root.is_file():
        return [root] if root.suffix == ".npy" else []
    files = []
    for path in root.rglob("*.npy"):
        if path.parent.name == "depths" or "depth" in path.stem:
            files.append(path)
    return sorted(files)


def _output_path(path: Path, *, input_root: Path, output_dir: Path | None) -> Path:
    if output_dir is None:
        return path.with_name(f"{path.stem}_vis.png")
    if input_root.is_file():
        rel = Path(f"{path.stem}_vis.png")
    else:
        rel = path.relative_to(input_root).with_suffix(".png")
        rel = rel.with_name(f"{rel.stem}_vis.png")
    return output_dir / rel


def _depth_to_vis(depth: np.ndarray) -> np.ndarray:
    depth = np.asarray(depth, dtype=np.float32).squeeze()
    if depth.ndim != 2:
        raise ValueError(f"expected 2D depth array after squeeze, got {depth.shape}")

    valid = np.isfinite(depth) & (depth > 0)
    vis = np.zeros((*depth.shape, 3), dtype=np.uint8)
    if not valid.any():
        return vis

    valid_depth = depth[valid]
    p01, p99 = np.percentile(valid_depth, [1, 99])
    if p99 <= p01:
        p99 = p01 + 1e-6
    t = np.clip((depth - p01) / (p99 - p01), 0.0, 1.0)

    red = np.clip(1.5 - np.abs(4.0 * t - 3.0), 0.0, 1.0)
    green = np.clip(1.5 - np.abs(4.0 * t - 2.0), 0.0, 1.0)
    blue = np.clip(1.5 - np.abs(4.0 * t - 1.0), 0.0, 1.0)
    color = np.stack([red, green, blue], axis=-1)
    color[~valid] = 0.0
    return (color * 255).astype(np.uint8)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Render saved Franka depths/*.npy arrays into PNG visualizations.",
    )
    parser.add_argument(
        "path",
        help="Episode directory, depths directory, or one depth .npy file.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Optional output root. Defaults to writing *_vis.png next to each .npy.",
    )
    return parser


def main() -> int:
    args = _build_parser().parse_args()
    root = Path(args.path).expanduser().resolve()
    output_dir = (
        Path(args.output_dir).expanduser().resolve()
        if args.output_dir is not None
        else None
    )
    files = _depth_files(root)
    if not files:
        raise FileNotFoundError(f"no Franka depth .npy files found under {root}")

    for path in files:
        depth = np.load(path)
        vis = _depth_to_vis(depth)
        out = _output_path(path, input_root=root, output_dir=output_dir)
        out.parent.mkdir(parents=True, exist_ok=True)
        imageio.imwrite(out, vis)
        print(f"{path} -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
