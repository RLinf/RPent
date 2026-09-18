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

"""RoboCasa tool schemas and handlers backed by the run's ``EnvState``.

The state trace (``states.json`` manifest + per-step artifact files under
``<step:02d>/``) is owned by :class:`rpent.session.EnvState`. Tool handlers
that need to read it take a ``state: EnvState`` keyword argument (bound by the
toolkit via :func:`functools.partial`); state-advancing primitive tools capture
state automatically through :meth:`RoboCasaToolkit.get_env_state`.
"""

from __future__ import annotations

from typing import Annotated, Literal

import numpy as np
from pydantic import Field

from robots.robocasa.primitives import RoboCasaPrimitives
from rpent.session import EnvState, StepRecord
from rpent.tools import ToolResult
from rpent.tools.base import tool

# ---- state persistence (dump_state writes through the run's EnvState) ----

# Heavy npy artifacts pruned after the ``_keep_heavy`` window elapses (the
# agent localizes from the latest frame; old world/depth maps are dead weight
# that once filled the 100GB root and deadlocked everything).
_HEAVY_ARTIFACTS = (
    "agentview_depth.npz",
    "agentview_world.npz",
    "wrist_depth.npz",
    "wrist_world.npz",
    "agentview_world_high.npz",
    "navview_world.npz",
)

# Artifact base names exposed to the agent for each camera (high-res first).
_CAMERA_IMAGE_ARTIFACTS = {
    "agentview": ("agentview_high.png", "agentview.png"),
    "navview": ("navview.png",),
    "wrist": ("wrist_high.png", "wrist.png"),
}

# Camera -> (low-res world map artifact, high-res world map artifact or None)
_CAMERA_WORLD_ARTIFACTS = {
    "agentview": ("agentview_world.npz", "agentview_world_high.npz"),
    "navview": ("navview_world.npz", None),
    "wrist": ("wrist_world.npz", None),
}


def dump_state(
    primitives: RoboCasaPrimitives,
    env_state: EnvState,
    log: dict | None = None,
) -> StepRecord:
    """Record one RoboCasa observation through its owned state record.

    Appends a new :class:`StepRecord` (proprio + task + success + vla_desync)
    via :meth:`EnvState.record_step`, saves the rendered RGB / depth / world
    artifacts for that step, and prunes heavy npy artifacts that fell out of
    the ``_keep_heavy`` window.
    """
    state_dict = primitives.current_state_dict()
    log = log or {}
    with env_state.record_step(
        state=state_dict["state"],
        terminated=state_dict["robocasa_terminated"],
        truncated=False,
        command=log.get("command"),
        result=log.get("result"),
        elapsed_s=log.get("elapsed_s"),
        extras={
            "task_language": state_dict["task_language"],
            "success": state_dict["success"],
            "task_progress": state_dict["task_progress"],
            "vla_desync": primitives._vla_desync,
        },
    ) as step_idx:
        _save_observation_artifacts(primitives, env_state, step_idx)
        env_state.prune_artifacts(
            _HEAVY_ARTIFACTS, step=step_idx, keep_last=primitives._keep_heavy
        )
    return env_state.get(step_idx)


def _save_observation_artifacts(
    primitives: RoboCasaPrimitives,
    env_state: EnvState,
    step_idx: int,
) -> None:
    """Render and save all per-step observation artifacts for ``step_idx``."""
    env = primitives.env
    hi_res = primitives.hi_res

    # ---- agentview + wrist: rgb, depth, world map, camera meta ----
    for cam, image_name, depth_name, world_name, meta_name in (
        (
            "agentview",
            "agentview.png",
            "agentview_depth.npz",
            "agentview_world.npz",
            "agentview_metadata.json",
        ),
        (
            "wrist",
            "wrist.png",
            "wrist_depth.npz",
            "wrist_world.npz",
            "wrist_metadata.json",
        ),
    ):
        rgb, depth = env.render_camera(cam, depth=True)
        env_state.save(image_name, rgb, step=step_idx)
        env_state.save(depth_name, depth.astype(np.float32), step=step_idx)
        env_state.save(world_name, env.world_map(cam).astype(np.float32), step=step_idx)
        if cam not in primitives._cam_meta_cache or cam == "wrist":
            primitives._cam_meta_cache[cam] = env.get_camera_meta(cam)
        env_state.save(meta_name, primitives._cam_meta_cache[cam], step=step_idx)

    # ---- hi-res agentview (SAM grounding / fine localize) ----
    if hi_res:
        hrgb, _ = env.render_camera("agentview", hi_res, hi_res, depth=True)
        env_state.save("agentview_high.png", hrgb, step=step_idx)
        env_state.save(
            "agentview_world_high.npz",
            env.world_map("agentview", hi_res, hi_res).astype(np.float16),
            step=step_idx,
        )

    # ---- navview: base-mounted forward-down floor camera (follows the base) ----
    nrgb, _ = env.render_camera("navview", depth=True)
    nworld = env.world_map("navview").astype(np.float32)
    env_state.save("navview.png", nrgb, step=step_idx)
    env_state.save("navview_world.npz", nworld, step=step_idx)
    floor = (nworld[:, :, 2] < 0.12) & (nworld[:, :, 2] > -0.2)
    overlay = nrgb.copy()
    overlay[floor] = [0, 255, 0]
    env_state.save("navview_floor.png", overlay, step=step_idx)


# ---- tool handlers (perception tools read the recorded EnvState) ----


@tool(readonly=True, exclude=("state",))
def view_env_state(step: int | None = None, *, state: EnvState) -> ToolResult:
    """Read step NN from states.json + the matching state images in the output dir. If step is null, returns the latest entry. Each entry contains the env state, robocasa_terminated flag, task_progress, vla_desync status, and log. Embeds available PNGs as multimodal image content blocks. Use calibration-frame agentview images for pixel back-projection; use navview for base navigation and floor walkability; use wrist for close-range details near the gripper.

    Args:
        step: Step number; 0 = initial. Null = latest.
    """
    images: list[bytes] = []
    try:
        record = state.get(step if step is not None else -1)
    except Exception as exc:
        return ToolResult(error=f"state step not available: {exc}")

    nn = record.step_idx
    extras = record.extras
    out: dict = {
        "step": nn,
        "task_progress": extras.get("task_progress", {}),
        "task_language": extras.get("task_language", ""),
        "state": record.state,
        "robocasa_terminated": record.terminated,
        "vla_desync": extras.get("vla_desync", False),
        "success": extras.get("success", False),
        "log": {
            "command": record.command,
            "result": record.result,
            "elapsed_s": record.elapsed_s,
        },
        "images": [],
    }

    for kind, candidates in (
        ("camera", _CAMERA_IMAGE_ARTIFACTS["agentview"]),
        ("navigation", _CAMERA_IMAGE_ARTIFACTS["navview"]),
        ("wrist", _CAMERA_IMAGE_ARTIFACTS["wrist"]),
    ):
        for name in candidates:
            if name not in record.artifacts:
                continue
            try:
                images.append(state.load_bytes(name, step=nn))
            except FileNotFoundError:
                continue
            label = {
                "camera": ("calibration_frame", "agentview"),
                "navigation": ("nav_view", "navview"),
                "wrist": ("calibration_frame", "wrist"),
            }[kind]
            out["images"].append(
                {
                    "role": label[0],
                    "camera": label[1],
                    "artifact": name,
                }
            )
            break

    return ToolResult(data=out, error=out.pop("error", None), images=images)


@tool(readonly=True, exclude=("state",))
def back_project_batch(
    pixels: Annotated[
        list[Annotated[list[int], Field(min_length=2, max_length=2)]],
        Field(min_length=1, max_length=50),
    ],
    step: int | None = None,
    camera: Literal["agentview", "navview", "wrist"] = "agentview",
    resolution: Literal["high", "low"] = "low",
    *,
    state: EnvState,
) -> ToolResult:
    """Back-project MULTIPLE pixels to world XYZ points in a single call. Loads the world map once and queries all pixels — replaces N separate back_project calls. Returns each pixel's world_xyz plus a summary with median_xyz across valid pixels.

    USE THIS for robust object localization: sample 3-8 pixels on the target object and read summary.median_xyz. Maximum 50 pixels per call.

    Args:
        pixels: List of [row, col] pixel coordinates (max 50)
        step: Depth / world-map step to use (default latest).
        camera: Camera to back-project from (default agentview).
        resolution: Coordinate system for pixels (default low). Use 'low' for the standard 256x256 world map.
    """
    camera = camera or "agentview"
    resolution = resolution or "low"
    if camera not in _CAMERA_WORLD_ARTIFACTS:
        return ToolResult(
            error=f"bad camera '{camera}' (use agentview, navview, or wrist)"
        )
    low_name, hi_name = _CAMERA_WORLD_ARTIFACTS[camera]
    source_artifact = hi_name if resolution == "high" else low_name
    if source_artifact is None:
        return ToolResult(error=f"{camera} has no {resolution}-resolution world map")

    try:
        record = state.get(step if step is not None else -1)
    except Exception as exc:
        return ToolResult(error=f"state step not available: {exc}")
    nn = record.step_idx
    if source_artifact not in record.artifacts:
        return ToolResult(
            error=f"{camera} {resolution}-resolution world map not recorded for step {nn}"
        )

    try:
        world_map = state.load(source_artifact, step=nn)
    except Exception as exc:
        return ToolResult(error=f"{source_artifact} not found for step {nn}: {exc}")

    results = []
    valid_xyzs = []
    for pixel in pixels:
        if not isinstance(pixel, (list, tuple)) or len(pixel) != 2:
            results.append(
                {
                    "pixel": pixel,
                    "world_xyz": None,
                    "valid": False,
                    "error": "pixel must be [row, col]",
                }
            )
            continue
        row, col = int(pixel[0]), int(pixel[1])
        h, w = world_map.shape[:2]
        if row < 0 or row >= h or col < 0 or col >= w:
            results.append(
                {
                    "pixel": pixel,
                    "world_xyz": None,
                    "valid": False,
                    "error": f"pixel ({row},{col}) out of bounds ({h}x{w})",
                }
            )
            continue
        xyz = world_map[row, col, :3]
        if not np.isfinite(xyz).all() or abs(float(xyz.sum())) <= 1e-6:
            results.append(
                {
                    "pixel": pixel,
                    "world_xyz": None,
                    "valid": False,
                    "error": "invalid world xyz at pixel",
                }
            )
            continue
        results.append(
            {
                "pixel": [row, col],
                "world_xyz": [
                    round(float(xyz[0]), 4),
                    round(float(xyz[1]), 4),
                    round(float(xyz[2]), 4),
                ],
                "valid": True,
                "error": None,
            }
        )
        valid_xyzs.append([float(xyz[0]), float(xyz[1]), float(xyz[2])])

    summary: dict = {
        "valid_count": len(valid_xyzs),
        "total_count": len(pixels),
    }
    if valid_xyzs:
        median = np.median(valid_xyzs, axis=0)
        summary["median_xyz"] = [
            round(float(median[0]), 4),
            round(float(median[1]), 4),
            round(float(median[2]), 4),
        ]

    return ToolResult(
        data={
            "results": results,
            "summary": summary,
            "step": nn,
            "camera": camera,
            "resolution": resolution,
        }
    )


@tool(readonly=True, exclude=("state",))
def query_world_map(
    z_min: float = 0.85,
    z_max: float = 0.95,
    x_range: Annotated[list[float], Field(min_length=2, max_length=2)] | None = None,
    y_range: Annotated[list[float], Field(min_length=2, max_length=2)] | None = None,
    camera: Literal["agentview", "navview", "wrist"] = "agentview",
    resolution: Literal["high", "low"] = "low",
    min_cluster_size: int = 10,
    *,
    state: EnvState,
) -> ToolResult:
    """Query the world map by Z-range / XY region to find objects at specific heights. Loads the world map once, filters pixels by z_min <= z <= z_max, optionally restricts to x_range / y_range, then clusters contiguous pixels into objects.

    TYPICAL USES:
    - z_min=0.85, z_max=0.95 -> countertop-height objects
    - z_min=0.0, z_max=0.12, camera='navview' -> walkable floor
    - z_min=0.85, z_max=0.95, x_range=[0,2], y_range=[-3,-1] -> counter objects in a specific quadrant

    Args:
        z_min: Minimum Z in meters (default 0.85 for counter height).
        z_max: Maximum Z in meters (default 0.95 for counter height).
        x_range: Optional X range [min, max] in meters; null = no filter.
        y_range: Optional Y range [min, max] in meters; null = no filter.
        camera: Camera world map to query (default agentview).
        resolution: World map resolution (default low).
        min_cluster_size: Minimum pixels per cluster to report (default 10).
    """
    if camera not in _CAMERA_WORLD_ARTIFACTS:
        return ToolResult(
            error=f"bad camera '{camera}' (use agentview, navview, or wrist)"
        )
    low_name, hi_name = _CAMERA_WORLD_ARTIFACTS[camera]
    source_artifact = hi_name if resolution == "high" else low_name
    if source_artifact is None:
        return ToolResult(error=f"{camera} has no {resolution}-resolution world map")

    try:
        record = state.get(-1)
    except Exception:
        return ToolResult(error="no state trace available")
    nn = record.step_idx
    if source_artifact not in record.artifacts:
        return ToolResult(
            error=f"{camera} {resolution}-resolution world map not found for step {nn}"
        )
    try:
        world_map = state.load(source_artifact, step=nn)
    except Exception:
        return ToolResult(
            error=f"{camera} {resolution}-resolution world map not found for step {nn}"
        )

    z = world_map[:, :, 2]
    mask = (z >= z_min) & (z <= z_max) & np.isfinite(z)
    if x_range:
        x = world_map[:, :, 0]
        mask &= (x >= x_range[0]) & (x <= x_range[1])
    if y_range:
        y = world_map[:, :, 1]
        mask &= (y >= y_range[0]) & (y <= y_range[1])

    ys, xs = np.where(mask)
    total_pixels = len(ys)
    if total_pixels < min_cluster_size:
        return ToolResult(
            data={
                "clusters": [],
                "summary": {"total_clusters": 0, "total_pixels_matched": 0},
            }
        )

    h, w = world_map.shape[:2]
    grid_cells = max(8, min(32, h // 32))
    cell_h = max(1, h // grid_cells)
    cell_w = max(1, w // grid_cells)
    cells: dict[tuple[int, int], dict] = {}
    for i in range(0, len(ys), 5):
        y, x = int(ys[i]), int(xs[i])
        gy, gx = y // cell_h, x // cell_w
        key = (gy, gx)
        if key not in cells:
            cells[key] = {"pixels": [], "world_pts": []}
        cells[key]["pixels"].append((y, x))
        cells[key]["world_pts"].append(world_map[y, x, :3])

    clusters = []
    for data in cells.values():
        if len(data["pixels"]) < min_cluster_size:
            continue
        pts = np.array(data["world_pts"])
        center = np.median(pts, axis=0)
        bbox_min = pts.min(axis=0)
        bbox_max = pts.max(axis=0)
        center_idx = len(data["pixels"]) // 2
        clusters.append(
            {
                "center_xyz": [
                    round(float(center[0]), 4),
                    round(float(center[1]), 4),
                    round(float(center[2]), 4),
                ],
                "pixel_count": len(data["pixels"]),
                "bbox_xyz": {
                    "min": [
                        round(float(bbox_min[0]), 4),
                        round(float(bbox_min[1]), 4),
                        round(float(bbox_min[2]), 4),
                    ],
                    "max": [
                        round(float(bbox_max[0]), 4),
                        round(float(bbox_max[1]), 4),
                        round(float(bbox_max[2]), 4),
                    ],
                },
                "sample_pixels": [list(data["pixels"][center_idx])],
            }
        )

    clusters.sort(key=lambda c: -c["pixel_count"])
    return ToolResult(
        data={
            "clusters": clusters[:20],
            "summary": {
                "total_clusters": len(clusters[:20]),
                "total_pixels_matched": total_pixels,
            },
        }
    )


@tool(readonly=True)
def finish(status: Literal["success", "failure", "stuck"], summary: str) -> ToolResult:
    """Declare the task finished. Call when robocasa_terminated becomes True (success detected), or when genuinely stuck after honest exploration. Provide a 1-3 sentence summary of what worked and what failed.

    Args:
        status: Task outcome classification.
        summary: 1-3 sentence summary of what worked / what failed.
    """
    return ToolResult(data={"_finish": True, "status": status, "summary": summary})


# ---- recipe export ----

_PRIMITIVE_ACTIONS = frozenset(
    {
        "move_to",
        "move_delta",
        "rotate_pitch",
        "set_gripper",
        "release",
        "scripted_grasp",
        "rldx_skill",
        "rldx_arm",
        "navigate_to",
        "move_base",
        "reset",
    }
)


def write_recipe_from_states(state: EnvState, recipe_tag: str) -> str:
    """Export non-error RoboCasa primitive commands from the state trace as JSONL."""
    commands = []
    for record in state.records():
        command = record.command
        if not isinstance(command, dict):
            continue
        if command.get("action") not in _PRIMITIVE_ACTIONS:
            continue
        result = record.result
        if isinstance(result, dict) and result.get("error"):
            continue
        commands.append(command)
    recipe_name = f"{recipe_tag}_recipe.jsonl"
    state.save(recipe_name, commands, step=None)
    return recipe_name
