"""Common physical agent tools."""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import numpy as np

from rpent.utils.config import get_repo_root
from rpent.utils.logging import get_output_dir

_BACKPROJECT_RADIUS = 6
_DEPTH_BAND_M = 0.02

TOOLS_SPEC: list[dict] = [
    {
        "name": "read_text_file",
        "description": (
            "Read a UTF-8 text file. Use for past recipe JSONLs, audit JSONs, "
            "and memory files. Large files are truncated."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "Absolute or repo-relative path"},
                "max_chars": {"type": "integer", "description": "Max chars (default 40000)"},
            },
            "required": ["path"],
        },
    },
    {
        "name": "write_text_file",
        "description": (
            "Write a UTF-8 text file (creates parent dirs). Use this to save "
            "the working recipe JSONL and the final audit JSON at the end of "
            "a successful run."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "path": {"type": "string"},
                "content": {"type": "string"},
            },
            "required": ["path", "content"],
        },
    },
    {
        "name": "list_dir",
        "description": (
            "List files in a directory (non-recursive). Default = {{output_dir}}. "
            "Use to inspect the working directory or to discover existing "
            "recipes in resources/libero/results_*_pert/."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "Default: {{output_dir}}"},
            },
        },
    },
    {
        "name": "finish",
        "description": (
            "Call when the task is complete or unrecoverable. Halts the agent "
            "loop. Save any artifacts (recipe, audit) BEFORE calling finish."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "status": {
                    "type": "string",
                    "description": "Outcome, e.g. 'success', 'failure', or 'stuck'.",
                },
                "summary": {
                    "type": "string",
                    "description": "Short natural-language summary of the run.",
                },
            },
            "required": ["status", "summary"],
        },
    },
]


def _resolve(path: str) -> Path:
    p = Path(path)
    if not p.is_absolute():
        p = get_repo_root() / p
    return p


def _truncate(text: str, max_chars: int) -> str:
    if len(text) <= max_chars:
        return text
    return (
        text[:max_chars]
        + f"\n\n[TRUNCATED — file is {len(text)} chars, showed first {max_chars}]"
    )


def read_text_file(path: str, max_chars: int = 40000) -> dict:
    p = _resolve(path)
    if not p.exists():
        return {"error": f"file not found: {p}"}
    if p.is_dir():
        return {"error": f"is a directory: {p}"}
    try:
        text = p.read_text(errors="replace")
    except Exception as e:
        return {"error": str(e)}
    return {"path": str(p), "size": len(text), "content": _truncate(text, max_chars)}


def write_text_file(path: str, content: str) -> dict:
    p = _resolve(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(content)
    return {"path": str(p), "bytes_written": len(content.encode("utf-8"))}


def list_dir(path: str = "") -> dict:
    # Default to the current output dir (so parallel agents see their own).
    p = _resolve(path) if path else get_output_dir()
    if not p.exists():
        return {"error": f"directory not found: {p}"}
    files = sorted(os.listdir(p))
    return {"path": str(p), "count": len(files), "files": files}


def finish(status: str, summary: str) -> dict:
    """Signal that the run is complete. Halts the agent loop.

    The ``_finish`` sentinel is what each planner detects to stop the
    tool-calling loop — see ``event.part.tool_name == "finish"`` in
    :meth:`rpent.planner.api_loop.ApiAgentLoop._solve` and the
    ``pending_finish`` bookkeeping in
    :class:`rpent.planner.claude_code._Recorder`.
    """
    return {"_finish": True, "status": status, "summary": summary}


def backproject_points(K, rows, cols, depths) -> np.ndarray:
    """Back-project pixel coords + depths to camera-frame XYZ, shape (N, 3)."""
    K = np.asarray(K, dtype=np.float64)
    rows = np.asarray(rows, dtype=np.float64)
    cols = np.asarray(cols, dtype=np.float64)
    depths = np.asarray(depths, dtype=np.float64)
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]
    return np.stack(
        [(cols - cx) * depths / fx, (rows - cy) * depths / fy, depths], axis=1
    )


def robust_surface_centroid(
    depth: np.ndarray,
    K,
    T_base_cam,
    row: int,
    col: int,
    *,
    radius: int = _BACKPROJECT_RADIUS,
    band: float = _DEPTH_BAND_M,
) -> dict:
    """Back-project a pixel neighbourhood to a robust 3D point.

    Back-projects every valid pixel in a ``(2*radius+1)`` window, keeps those on
    the dominant surface (depth within ``band`` of the window median, rejecting
    background / table / dropouts), and returns the median point + diagnostics.
    Returns world ``xyz`` when ``T_base_cam`` is given, otherwise camera-frame
    ``xyz_cam``.
    """
    row, col = int(row), int(col)
    radius = max(0, int(radius))
    height, width = depth.shape[:2]
    if not (0 <= row < height and 0 <= col < width):
        return {
            "error": (
                f"pixel ({row},{col}) out of bounds; image is {height}x{width}"
            )
        }
    row_start, row_end = max(0, row - radius), min(height, row + radius + 1)
    col_start, col_end = max(0, col - radius), min(width, col + radius + 1)
    rows, cols = np.mgrid[row_start:row_end, col_start:col_end]
    depths = depth[row_start:row_end, col_start:col_end].reshape(-1).astype(
        np.float64
    )
    rows = rows.reshape(-1).astype(np.float64)
    cols = cols.reshape(-1).astype(np.float64)
    valid = np.isfinite(depths) & (depths > 0)
    if not np.any(valid):
        return {"error": f"no valid depth near ({row},{col}); pick another pixel"}
    depths, rows, cols = depths[valid], rows[valid], cols[valid]
    median_depth = float(np.median(depths))
    surface = np.abs(depths - median_depth) <= band
    depths, rows, cols = depths[surface], rows[surface], cols[surface]
    if depths.size == 0:
        return {"error": f"no dominant surface depth near ({row},{col})"}

    camera_points = backproject_points(K, rows, cols, depths)
    camera_point = np.median(camera_points, axis=0)
    out: dict[str, Any] = {
        "pixel": [row, col],
        "radius": radius,
        "n_points": int(camera_points.shape[0]),
        "depth_m": round(median_depth, 4),
        "xyz_cam": [round(float(value), 4) for value in camera_point],
    }
    if T_base_cam is not None:
        transform = np.asarray(T_base_cam, dtype=np.float64)
        base_points = camera_points @ transform[:3, :3].T + transform[:3, 3]
        base_point = np.median(base_points, axis=0)
        out["xyz"] = [round(float(value), 4) for value in base_point]
        out["xy_spread_m"] = round(
            float(np.hypot(*base_points[:, :2].std(axis=0))), 4
        )
    else:
        out["xy_spread_m"] = round(
            float(np.hypot(*camera_points[:, :2].std(axis=0))), 4
        )
    return out


TOOL_HANDLERS: dict = {
    "read_text_file": read_text_file,
    "write_text_file": write_text_file,
    "list_dir": list_dir,
    "finish": finish,
}
