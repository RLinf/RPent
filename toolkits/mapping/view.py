# Copyright 2026 The RPent Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Render a grounded map and its annotations as a portable report.

The grounder writes ``objects.json``, ``semantic_points.npz``, and per-frame
overlays next to the map. This module turns them into artifacts a person can
inspect without loading the reconstruction arrays:

1. a top-down semantic view, oriented by the map's gravity and floor when
   those were estimated and by the camera trajectory otherwise;
2. an isometric view of the same points with the camera path;
3. ``detections.html`` with the per-label evaluation, the object and detection
   tables, and every per-frame overlay;
4. ``objects.csv`` with one row per instance.

Run it on a map that has already been grounded::

    python -m toolkits.mapping.view --map-path /path/to/maps/workspace

The report is written to ``<map>/report`` unless ``--output-dir`` names
another directory. Plots are skipped for a map annotated in 2D only.
"""

from __future__ import annotations

import argparse
import csv
import html
import json
import logging
import math
import shutil
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from rpent.utils.logging import get_logger
from toolkits.grounding.ground_map import (
    LABEL_PALETTE,
    LocalMapSource,
)
from toolkits.map_geometry import gravity_up, meters_per_unit, rotation_to_z
from toolkits.mapping.server import (
    ANNOTATIONS_DIR,
    OBJECTS_FILE,
    SEMANTIC_POINTS_FILE,
)

logger = get_logger("map_view")

#: Report subdirectory used when the caller names none.
REPORT_DIR = "report"
TOPDOWN_FILE = "semantic_topdown.png"
ISOMETRIC_FILE = "semantic_3d.png"
HTML_FILE = "detections.html"
OBJECTS_CSV = "objects.csv"
#: Upper bound on the points drawn in one figure; the full artifact stays intact.
DEFAULT_MAX_POINTS = 200_000

_MATPLOTLIB_HINT = (
    "map_view needs matplotlib; install it with 'uv pip install -e \".[map-view]\"'"
)


def _pyplot() -> Any:
    """Import pyplot against the headless backend, or name the missing extra."""
    try:
        import matplotlib
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(_MATPLOTLIB_HINT) from exc
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


@dataclass(frozen=True)
class MapOrientation:
    """Rotation and translation from the map frame into the display frame.

    ``apply`` returns ``points @ rotation.T - offset``. The rotation maps the
    estimated up direction onto ``+z``; the offset centres the camera path in
    ``x``/``y`` and puts the estimated floor at ``z = 0``.
    """

    rotation: np.ndarray
    offset: np.ndarray
    up: np.ndarray | None
    source: str
    floor_aligned: bool

    def apply(self, points: np.ndarray) -> np.ndarray:
        """Transform ``[...,3]`` map-frame points into the display frame."""
        array = np.asarray(points, dtype=np.float64)
        return array @ self.rotation.T - self.offset


@dataclass(frozen=True)
class SemanticCloud:
    """The semantic point arrays written by the grounder."""

    points: np.ndarray
    labels: np.ndarray
    scores: np.ndarray
    object_ids: np.ndarray
    frame_indices: np.ndarray
    label_names: list[str]


def _load_orientation(
    map_dir: Path,
    metadata: dict[str, Any],
) -> tuple[MapOrientation, np.ndarray | None]:
    """Orient the map by gravity and floor, falling back to the camera path."""
    try:
        poses = LocalMapSource(map_dir).camera_to_world
    except Exception as exc:  # A map without camera files still renders flat.
        logger.warning("could not read camera poses for orientation: %s", exc)
        poses = None

    up = gravity_up(metadata)
    source = "gravity" if up is not None else "identity"
    if up is None and poses is not None and len(poses):
        down = poses[:, :3, 1].mean(axis=0)
        norm = float(np.linalg.norm(down))
        if math.isfinite(norm) and norm > 1e-9:
            up = -down / norm
            source = "camera"
    if up is not None:
        rotation = rotation_to_z(up)
    else:
        rotation = np.eye(3)

    offset = np.zeros(3)
    if poses is not None and len(poses):
        centres = poses[:, :3, 3] @ rotation.T
        offset[0] = float(centres[:, 0].mean())
        offset[1] = float(centres[:, 1].mean())

    floor_aligned = False
    floor = metadata.get("floor")
    if isinstance(floor, dict) and floor.get("status") == "estimated":
        normal = np.asarray(floor.get("normal", []), dtype=np.float64)
        floor_offset = floor.get("offset")
        if (
            normal.shape == (3,)
            and np.isfinite(normal).all()
            and isinstance(floor_offset, int | float)
            and math.isfinite(float(floor_offset))
            and up is not None
            and abs(float(normal @ up)) > 0.99
        ):
            # A floor point ``p`` satisfies ``dot(p, normal) == offset``.
            offset[2] = float(floor_offset)
            floor_aligned = True

    orientation = MapOrientation(
        rotation=rotation,
        offset=offset,
        up=None if up is None else np.asarray(up, dtype=np.float64),
        source=source,
        floor_aligned=floor_aligned,
    )
    return orientation, poses


def _load_cloud(path: Path) -> SemanticCloud:
    """Read the grounder's semantic point arrays."""
    with np.load(path, allow_pickle=False) as data:
        # Keep the stored dtypes: the cloud is subsampled before any
        # float64 transform, so upcasting millions of points here is wasted.
        points = np.asarray(data["points"])
        labels = np.asarray(data["labels"])
        scores = np.asarray(data["scores"])
        object_ids = np.asarray(data["object_ids"])
        frame_indices = np.asarray(data["frame_indices"])
        label_names = [str(name) for name in np.asarray(data["label_names"])]
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError(f"semantic points must be [N,3], got {points.shape}")
    count = points.shape[0]
    for name, array in (
        ("labels", labels),
        ("scores", scores),
        ("object_ids", object_ids),
        ("frame_indices", frame_indices),
    ):
        if array.shape != (count,):
            raise ValueError(
                f"semantic {name} has shape {array.shape}, expected ({count},)"
            )
    if not label_names:
        raise ValueError("semantic points declare no label names")
    if count and (labels.min() < 0 or labels.max() >= len(label_names)):
        raise ValueError("semantic labels index outside the label names")
    return SemanticCloud(
        points=points,
        labels=labels,
        scores=scores,
        object_ids=object_ids,
        frame_indices=frame_indices,
        label_names=label_names,
    )


def _label_colors(label_names: Sequence[str]) -> dict[str, np.ndarray]:
    """Assign each label the palette color used in the PLY and overlays."""
    return {
        label: LABEL_PALETTE[index % len(LABEL_PALETTE)] / 255.0
        for index, label in enumerate(label_names)
    }


def _subsample(count: int, max_points: int) -> np.ndarray:
    """Return deterministic indices for at most ``max_points`` samples."""
    if count <= max_points:
        return np.arange(count)
    generator = np.random.default_rng(0)
    return np.sort(generator.choice(count, size=max_points, replace=False))


def _centres_of(
    objects: Sequence[dict[str, Any]],
) -> list[tuple[dict[str, Any], np.ndarray]]:
    """Collect objects that carry a finite 3D centroid."""
    result: list[tuple[dict[str, Any], np.ndarray]] = []
    for instance in objects:
        centroid = np.asarray(instance.get("centroid", []), dtype=np.float64)
        if centroid.shape == (3,) and np.isfinite(centroid).all():
            result.append((instance, centroid))
    return result


def _draw_objects(
    axes: Any,
    objects: Sequence[tuple[dict[str, Any], np.ndarray]],
    colors: dict[str, np.ndarray],
    orientation: MapOrientation,
) -> None:
    """Draw instance centroids with their IDs."""
    if not objects:
        return
    from matplotlib.patheffects import withStroke

    placed = orientation.apply(np.stack([centroid for _, centroid in objects]))
    for (instance, _), centre in zip(objects, placed, strict=True):
        color = colors.get(str(instance.get("label")), np.asarray([0.1, 0.1, 0.1]))
        axes.scatter(
            centre[0],
            centre[1],
            color=color,
            edgecolor="black",
            linewidths=0.8,
            s=48,
            zorder=6,
        )
        axes.annotate(
            str(instance.get("id", "?")),
            (centre[0], centre[1]),
            xytext=(4, 4),
            textcoords="offset points",
            fontsize=7,
            zorder=7,
            path_effects=[withStroke(linewidth=2.5, foreground="white")],
        )


def _legend(axes: Any, colors: dict[str, np.ndarray], counts: Counter[str]) -> None:
    """Add one legend entry per label."""
    from matplotlib.lines import Line2D

    handles = [
        Line2D(
            [],
            [],
            marker="o",
            linestyle="",
            markersize=7,
            color=color,
            label=f"{label} ({counts.get(label, 0)})",
        )
        for label, color in colors.items()
        if counts.get(label, 0) > 0
    ]
    if handles:
        axes.legend(handles=handles, loc="best", fontsize=8, framealpha=0.85)


def _unit_label() -> str:
    """Name the plotted distance unit: the report plots the model frame."""
    return "map units"


def _scale_note(metadata: dict[str, Any]) -> str:
    """Qualify distances with the map's metric anchor when it has one."""
    scale = meters_per_unit(metadata)
    if scale is None:
        return " (arbitrary scale)"
    return f" (1 map unit = {scale:g} m)"


def _orientation_label(orientation: MapOrientation) -> str:
    if orientation.source == "camera":
        alignment = "camera-aligned"
    elif orientation.up is not None:
        alignment = "gravity-aligned"
    else:
        alignment = "map frame as recorded"
    if orientation.floor_aligned:
        return f"floor at z=0, {alignment}"
    return alignment


def _drawable_points(
    cloud: SemanticCloud,
    orientation: MapOrientation,
    max_points: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Return transformed points and labels, dropping non-finite coordinates."""
    indices = _subsample(cloud.points.shape[0], max_points)
    points = orientation.apply(cloud.points[indices])
    labels = cloud.labels[indices]
    finite = np.isfinite(points).all(axis=1)
    return points[finite], labels[finite]


def _point_colors(labels: np.ndarray) -> np.ndarray:
    """Return one palette color per point."""
    return LABEL_PALETTE[labels % len(LABEL_PALETTE)] / 255.0


def _plot_topdown(
    path: Path,
    map_name: str,
    cloud: SemanticCloud | None,
    document: dict[str, Any],
    orientation: MapOrientation,
    poses: np.ndarray | None,
    metadata: dict[str, Any],
    max_points: int,
) -> bool:
    """Render the top-down semantic view; return whether points were drawn."""
    if cloud is None or cloud.points.shape[0] == 0:
        return False
    plt = _pyplot()
    objects = _centres_of(document.get("objects", []))
    colours = _label_colors(cloud.label_names)
    figure, axes = plt.subplots(figsize=(12, 12))
    points, labels = _drawable_points(cloud, orientation, max_points)
    drawn = bool(labels.size)
    if drawn:
        counts = Counter(cloud.label_names[index] for index in cloud.labels)
        axes.scatter(
            points[:, 0],
            points[:, 1],
            c=_point_colors(labels),
            s=0.7,
            alpha=0.6,
            linewidths=0,
            rasterized=True,
        )
        _legend(axes, colours, counts)
    path_points = None
    if poses is not None and len(poses):
        transformed = orientation.apply(poses[:, :3, 3])
        transformed = transformed[np.isfinite(transformed).all(axis=1)]
        if transformed.shape[0]:
            path_points = transformed
            axes.plot(
                path_points[:, 0],
                path_points[:, 1],
                color="black",
                linewidth=1.6,
                alpha=0.85,
                zorder=5,
            )
            axes.scatter(
                path_points[0, 0],
                path_points[0, 1],
                color="lime",
                edgecolor="black",
                s=90,
                zorder=8,
            )
            axes.scatter(
                path_points[-1, 0],
                path_points[-1, 1],
                color="red",
                edgecolor="black",
                s=90,
                zorder=8,
            )
    _draw_objects(axes, objects, colours, orientation)
    if not drawn:
        plt.close(figure)
        return False

    coordinates: list[np.ndarray] = [points[:, :2]]
    if path_points is not None:
        coordinates.append(path_points[:, :2])
    if objects:
        transformed = orientation.apply(np.stack([centroid for _, centroid in objects]))
        coordinates.append(transformed[:, :2])
    if coordinates:
        merged = np.concatenate(coordinates, axis=0)
        low, high = np.percentile(merged, [1.0, 99.0], axis=0)
        span = float(max(high[0] - low[0], high[1] - low[1]))
        if not math.isfinite(span) or span <= 0.0:
            span = 1.0
        padding = 0.08 * span
        axes.set_xlim(low[0] - padding, low[0] + span + padding)
        axes.set_ylim(low[1] - padding, low[1] + span + padding)
        axes.set_aspect("equal", adjustable="box")

    unit = _unit_label()
    axes.set_xlabel(f"x ({unit})")
    axes.set_ylabel(f"y ({unit})")
    axes.grid(True, linewidth=0.3, alpha=0.4)
    axes.set_title(
        f"{map_name}: semantic map\n"
        f"{_orientation_label(orientation)}; "
        f"{cloud.points.shape[0]} points, "
        f"{len(document.get('objects', []))} objects"
        f"{_scale_note(metadata)}",
        fontsize=12,
    )
    figure.tight_layout()
    figure.savefig(path, dpi=160)
    plt.close(figure)
    return True


def _plot_isometric(
    path: Path,
    map_name: str,
    cloud: SemanticCloud,
    orientation: MapOrientation,
    poses: np.ndarray | None,
    metadata: dict[str, Any],
    max_points: int,
) -> bool:
    """Render an isometric view; return whether points were drawn."""
    plt = _pyplot()
    points, labels = _drawable_points(cloud, orientation, max_points)
    if not labels.size:
        return False
    counts = Counter(cloud.label_names[index] for index in cloud.labels)
    colours = _label_colors(cloud.label_names)

    figure = plt.figure(figsize=(14, 12))
    axes = figure.add_subplot(111, projection="3d")
    axes.scatter(
        points[:, 0],
        points[:, 1],
        points[:, 2],
        c=_point_colors(labels),
        s=0.4,
        alpha=0.55,
        linewidths=0,
        rasterized=True,
    )
    drawn_sets = [points]
    if poses is not None and len(poses):
        path_points = orientation.apply(poses[:, :3, 3])
        path_points = path_points[np.isfinite(path_points).all(axis=1)]
        if path_points.shape[0]:
            axes.plot(
                path_points[:, 0],
                path_points[:, 1],
                path_points[:, 2],
                color="black",
                linewidth=1.8,
                zorder=5,
            )
            axes.scatter(*path_points[0], color="lime", edgecolor="black", s=90)
            axes.scatter(*path_points[-1], color="red", edgecolor="black", s=90)
            drawn_sets.append(path_points)
    spans = np.ptp(np.vstack(drawn_sets), axis=0)
    largest = float(spans.max()) if spans.size else 0.0
    if largest > 0.0:
        axes.set_box_aspect([max(float(value) / largest, 0.05) for value in spans])
    axes.set_xlabel(f"x ({_unit_label()})")
    axes.set_ylabel(f"y ({_unit_label()})")
    axes.set_zlabel("up")
    axes.view_init(elev=30, azim=-60)
    axes.set_title(
        f"{map_name}: semantic point cloud\n"
        f"{labels.size} of {cloud.points.shape[0]} points{_scale_note(metadata)}",
        fontsize=12,
    )
    _legend(axes, colours, counts)
    figure.tight_layout()
    figure.savefig(path, dpi=160)
    plt.close(figure)
    return True


def _format_vector(values: Any, digits: int = 3) -> str:
    """Format a numeric triple for tables."""
    array = np.asarray(values, dtype=np.float64).reshape(-1)
    return ", ".join(f"{value:.{digits}f}" for value in array)


def _table(headers: Sequence[str], rows: Sequence[Sequence[str]]) -> str:
    """Render an escaped HTML table."""
    head = "".join(f"<th>{html.escape(header)}</th>" for header in headers)
    body = "".join(
        "<tr>" + "".join(f"<td>{cell}</td>" for cell in row) + "</tr>" for row in rows
    )
    return f"<table><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table>"


def _copy_overlays(
    map_dir: Path,
    output_dir: Path,
    detections: Sequence[dict[str, Any]],
) -> dict[int, str]:
    """Copy the overlays referenced by the detections into the report."""
    frames = sorted({int(detection.get("keyframe", -1)) for detection in detections})
    copied: dict[int, str] = {}
    for frame in frames:
        source = map_dir / ANNOTATIONS_DIR / f"overlay_{frame:06d}.png"
        if not source.is_file():
            continue
        target_dir = output_dir / ANNOTATIONS_DIR
        target_dir.mkdir(parents=True, exist_ok=True)
        target = target_dir / source.name
        shutil.copyfile(source, target)
        copied[frame] = f"{ANNOTATIONS_DIR}/{source.name}"
    return copied


def _write_html(
    path: Path,
    *,
    map_dir: Path,
    metadata: dict[str, Any],
    document: dict[str, Any],
    cloud: SemanticCloud | None,
    orientation: MapOrientation,
    topdown: bool,
    isometric: bool,
    overlays: dict[int, str],
) -> None:
    """Write the self-contained report page."""
    detections = document.get("detections", [])
    objects = document.get("objects", [])
    evaluation = document.get("evaluation", {})
    per_label = evaluation.get("labels", {}) if isinstance(evaluation, dict) else {}
    frames = sorted({int(detection.get("keyframe", -1)) for detection in detections})
    counts = Counter(str(detection.get("label")) for detection in detections)

    summary_rows = [
        ["Map", html.escape(str(map_dir))],
        ["Frame", html.escape(str(metadata.get("frame", "model")))],
        [
            "Scale",
            html.escape(str((metadata.get("scale") or {}).get("status", "unknown"))),
        ],
        ["Orientation", html.escape(_orientation_label(orientation))],
        ["Frames", html.escape(str(metadata.get("frames", 0)))],
        ["Grounded keyframes", html.escape(str(len(document.get("keyframes", []))))],
        ["Detections", html.escape(str(len(detections)))],
        ["Objects", html.escape(str(len(objects)))],
        [
            "Semantic points",
            html.escape("0" if cloud is None else str(cloud.points.shape[0])),
        ],
    ]

    label_rows = []
    for label in sorted(per_label):
        row = per_label[label]
        label_rows.append(
            [
                html.escape(label),
                html.escape(str(row.get("detections", 0))),
                html.escape(str(row.get("lifted_detections", 0))),
                html.escape(str(row.get("objects", 0))),
                _optional_float(row.get("best_score")),
                _optional_float(row.get("median_nearest_centroid_distance")),
            ]
        )

    object_rows = [
        [
            html.escape(str(instance.get("id", ""))),
            html.escape(str(instance.get("label", ""))),
            html.escape(str(instance.get("observation_count", 0))),
            html.escape(str(instance.get("point_count", 0))),
            html.escape(_format_vector(instance.get("centroid", []))),
            html.escape(_format_vector(instance.get("extent", []))),
            html.escape(f"{float(instance.get('radius', 0.0)):.3f}"),
        ]
        for instance in objects
    ]

    detection_rows = [
        [
            html.escape(str(detection.get("keyframe", ""))),
            html.escape(str(detection.get("label", ""))),
            html.escape(f"{float(detection.get('score', 0.0)):.3f}"),
            html.escape(str(detection.get("mask_pixels", 0))),
            html.escape("yes" if detection.get("lifted") else "no"),
            html.escape(str(detection.get("lift_reason") or "")),
            html.escape(str(detection.get("object_id") or "")),
            html.escape(_format_vector(detection.get("box", []), digits=1)),
        ]
        for detection in detections
    ]

    figures = []
    for frame in frames:
        overlay = overlays.get(frame)
        if overlay is None:
            continue
        frame_detections = [
            detection
            for detection in detections
            if int(detection.get("keyframe", -1)) == frame
        ]
        caption = ", ".join(
            f"{html.escape(str(detection.get('label')))!s} "
            f"{float(detection.get('score', 0.0)):.2f}"
            for detection in frame_detections
        )
        figures.append(
            f'<figure><img src="{html.escape(overlay)}" loading="lazy" '
            f'alt="frame {frame}"><figcaption>frame {frame}: {caption}</figcaption>'
            "</figure>"
        )

    label_text = "".join(
        f'<span class="chip">{html.escape(label)} ({count})</span>'
        for label, count in sorted(counts.items())
    )
    images = ""
    if topdown:
        images += (
            f'<figure class="wide"><img src="{TOPDOWN_FILE}" '
            'alt="top-down semantic map"><figcaption>Top-down semantic map'
            "</figcaption></figure>"
        )
    if isometric:
        images += (
            f'<figure class="wide"><img src="{ISOMETRIC_FILE}" '
            'alt="isometric semantic point cloud"><figcaption>Isometric view'
            "</figcaption></figure>"
        )
    if not images:
        images = (
            "<p>This map has no lifted semantic points, so the report contains "
            "2D annotations only.</p>"
        )

    page = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{html.escape(map_dir.name)} semantic report</title>
<style>
:root {{ color-scheme: light dark; }}
body {{ font-family: system-ui, -apple-system, "Segoe UI", sans-serif;
       margin: 2rem auto; max-width: 1100px; padding: 0 1rem; line-height: 1.45; }}
h1, h2 {{ font-weight: 600; }}
table {{ border-collapse: collapse; width: 100%; margin: 0.6rem 0 1.6rem; font-size: 0.88rem; }}
th, td {{ border: 1px solid rgba(128, 128, 128, 0.35); padding: 0.3rem 0.5rem; text-align: left; }}
th {{ background: rgba(128, 128, 128, 0.12); }}
figure {{ margin: 1rem 0; }}
figure.wide img {{ width: 100%; height: auto; }}
.gallery {{ display: grid; grid-template-columns: repeat(auto-fill, minmax(240px, 1fr)); gap: 0.8rem; }}
.gallery figure {{ margin: 0; }}
.gallery img {{ width: 100%; height: auto; border: 1px solid rgba(128, 128, 128, 0.3); }}
figcaption {{ font-size: 0.8rem; color: rgba(128, 128, 128, 0.95); }}
.chip {{ display: inline-block; padding: 0.1rem 0.45rem; margin: 0 0.3rem 0.3rem 0;
         border-radius: 999px; background: rgba(128, 128, 128, 0.18); font-size: 0.8rem; }}
details {{ margin: 0.6rem 0 1.6rem; }}
summary {{ cursor: pointer; font-weight: 600; }}
</style>
</head>
<body>
<h1>{html.escape(map_dir.name)} semantic report</h1>
<p>{label_text}</p>
{_table(["Field", "Value"], summary_rows)}
{images}
<h2>Per-label evaluation</h2>
<p><small>Precision and recall need independent ground-truth labels and are
not reported. The distance is the median nearest same-label centroid distance
between detections in different frames.</small></p>
{_table(["Label", "Detections", "Lifted", "Objects", "Best score", f"Median neighbour distance ({_unit_label()})"], label_rows)}
<h2>Objects</h2>
{_table(["ID", "Label", "Observations", "Points", f"Centroid ({_unit_label()})", f"Extent ({_unit_label()})", "Radius"], object_rows)}
<h2>Frame overlays</h2>
<div class="gallery">
{"".join(figures) if figures else "<p>No overlay images were found.</p>"}
</div>
<h2>Detections</h2>
<details>
<summary>{len(detections)} detections (click to expand)</summary>
{_table(["Keyframe", "Label", "Score", "Mask px", "Lifted", "Lift reason", "Object", "Box"], detection_rows)}
</details>
</body>
</html>
"""
    path.write_text(page, encoding="utf-8")


def _optional_float(value: Any) -> str:
    if value is None:
        return ""
    return f"{float(value):.3f}"


def _write_objects_csv(path: Path, objects: Sequence[dict[str, Any]]) -> None:
    """Write one row per instance."""
    has_robot_frame = any("robot_frame" in instance for instance in objects)
    headers = [
        "id",
        "label",
        "frame",
        "observation_count",
        "point_count",
        "centroid_x",
        "centroid_y",
        "centroid_z",
        "extent_x",
        "extent_y",
        "extent_z",
        "radius",
    ]
    if has_robot_frame:
        headers += ["robot_centroid_x", "robot_centroid_y", "robot_centroid_z"]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=headers)
        writer.writeheader()
        for instance in objects:
            centroid = np.asarray(instance.get("centroid", []), dtype=np.float64)
            extent = np.asarray(instance.get("extent", []), dtype=np.float64)
            row: dict[str, Any] = {
                "id": instance.get("id", ""),
                "label": instance.get("label", ""),
                "frame": instance.get("frame", "model"),
                "observation_count": instance.get("observation_count", 0),
                "point_count": instance.get("point_count", 0),
                "centroid_x": _round_or_blank(centroid, 0),
                "centroid_y": _round_or_blank(centroid, 1),
                "centroid_z": _round_or_blank(centroid, 2),
                "extent_x": _round_or_blank(extent, 0),
                "extent_y": _round_or_blank(extent, 1),
                "extent_z": _round_or_blank(extent, 2),
                "radius": round(float(instance.get("radius", 0.0)), 6),
            }
            robot = instance.get("robot_frame")
            if has_robot_frame:
                robot_centroid = np.asarray(
                    (robot or {}).get("centroid", []), dtype=np.float64
                )
                row["robot_centroid_x"] = _round_or_blank(robot_centroid, 0)
                row["robot_centroid_y"] = _round_or_blank(robot_centroid, 1)
                row["robot_centroid_z"] = _round_or_blank(robot_centroid, 2)
            writer.writerow(row)


def _round_or_blank(values: np.ndarray, index: int) -> Any:
    if values.shape != (3,):
        return ""
    return round(float(values[index]), 6)


def build_report(
    map_path: str | Path,
    output_dir: str | Path | None = None,
    *,
    max_points: int = DEFAULT_MAX_POINTS,
) -> dict[str, str]:
    """Write the plots, HTML page, and CSV for a grounded map.

    Args:
        map_path: Built map directory containing ``map.json`` and
            ``objects.json``.
        output_dir: Report directory; defaults to ``<map>/report``.
        max_points: Upper bound on points drawn in each figure.

    Returns:
        The written artifact paths, keyed by role.
    """
    if max_points < 1:
        raise ValueError("max_points must be positive")
    map_dir = Path(map_path)
    metadata_path = map_dir / "map.json"
    objects_path = map_dir / OBJECTS_FILE
    if not metadata_path.is_file():
        raise FileNotFoundError(f"map metadata not found: {metadata_path}")
    if not objects_path.is_file():
        raise FileNotFoundError(
            f"annotations not found: {objects_path}; run toolkits.grounding.ground_map first"
        )
    metadata = json.loads(metadata_path.read_text())
    document = json.loads(objects_path.read_text())
    if not isinstance(document, dict):
        raise ValueError(f"{objects_path} must contain a JSON object")

    points_path = map_dir / SEMANTIC_POINTS_FILE
    cloud = _load_cloud(points_path) if points_path.is_file() else None
    orientation, poses = _load_orientation(map_dir, metadata)

    report_dir = Path(output_dir) if output_dir is not None else map_dir / REPORT_DIR
    report_dir.mkdir(parents=True, exist_ok=True)
    written: dict[str, str] = {}

    topdown = _plot_topdown(
        report_dir / TOPDOWN_FILE,
        map_dir.name,
        cloud,
        document,
        orientation,
        poses,
        metadata,
        max_points,
    )
    if topdown:
        written["topdown"] = str(report_dir / TOPDOWN_FILE)
    isometric = False
    if cloud is not None and cloud.points.shape[0] > 0:
        isometric = _plot_isometric(
            report_dir / ISOMETRIC_FILE,
            map_dir.name,
            cloud,
            orientation,
            poses,
            metadata,
            max_points,
        )
        if isometric:
            written["isometric"] = str(report_dir / ISOMETRIC_FILE)

    overlays = _copy_overlays(map_dir, report_dir, document.get("detections", []))
    html_path = report_dir / HTML_FILE
    _write_html(
        html_path,
        map_dir=map_dir,
        metadata=metadata,
        document=document,
        cloud=cloud,
        orientation=orientation,
        topdown=topdown,
        isometric=isometric,
        overlays=overlays,
    )
    written["html"] = str(html_path)
    csv_path = report_dir / OBJECTS_CSV
    _write_objects_csv(csv_path, document.get("objects", []))
    written["csv"] = str(csv_path)
    return written


def _build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Render a grounded LingBot-Map map as a report"
    )
    parser.add_argument(
        "--map-path", required=True, help="built and grounded map directory"
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="report directory (default: <map>/report)",
    )
    parser.add_argument("--max-points", type=int, default=DEFAULT_MAX_POINTS)
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    """Write the report for one map."""
    args = _build_argparser().parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    try:
        written = build_report(
            args.map_path, args.output_dir, max_points=args.max_points
        )
    except Exception as exc:
        logger.error("map view failed: %s", exc)
        raise SystemExit(1) from exc
    logger.info("wrote %d report artifacts for %s", len(written), args.map_path)
    for name, path in written.items():
        logger.info("wrote %s: %s", name, path)


if __name__ == "__main__":
    main()
