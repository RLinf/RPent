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


"""Recorded segmentation and coordinate reads owned by GeometryAgent.

These are internal Python operations, not parent-planner tool declarations.
"""

from __future__ import annotations

import threading
from typing import TYPE_CHECKING, Literal

import numpy as np

from rpent.session import EnvState, StepRecord
from rpent.tools import ToolResult

if TYPE_CHECKING:
    from rpent.robots.components.sam3_client import Sam3Client


class GeometryPerception:
    """Own the SAM client and per-agent segmentation artifact sequence."""

    def __init__(self, sam3_client: Sam3Client | None) -> None:
        self._sam3_client = sam3_client
        self._segment_index_lock = threading.Lock()
        self._segment_index = 0

    def segment(
        self,
        prompt: str = "",
        camera: Literal["agentview", "wrist"] = "agentview",
        step: int = -1,
        point: list[int] | None = None,
        min_score: float = 0.2,
        *,
        state: EnvState,
    ) -> ToolResult:
        """Segment a recorded RGB/world-map pair and persist its mask and evidence.

        Args:
            prompt: Target description, mutually exclusive with point.
            camera: Recorded agentview or wrist camera.
            step: Observation step; -1 selects the latest record.
            point: Single positive prompt in row/column order.
            min_score: Minimum accepted SAM score.
            state: Shared observation and artifact store.
        """
        images: list[bytes] = []
        try:
            record = state.get(step)
        except Exception as exc:
            return ToolResult(data={"error": f"state step not available: {exc}"})
        nn = record.step_idx

        camera = camera or "agentview"
        prompt = prompt.strip()
        has_prompt = bool(prompt)
        has_point = point is not None
        if has_prompt == has_point:
            return ToolResult(
                data={"error": "segment needs exactly one of prompt or point"}
            )
        try:
            image_name, world_name, artifact_pairs = select_observation_artifacts(
                state, record, camera
            )
        except ValueError as e:
            return ToolResult(data={"error": str(e)})
        if image_name is None or world_name is None:
            return ToolResult(
                data={
                    "error": "complete segment artifacts not found",
                    "step": nn,
                    "camera": camera,
                    "checked_artifacts": [
                        name
                        for (image, world) in artifact_pairs
                        for name in (image, world)
                        if name
                    ],
                }
            )

        try:
            data = self._sam3_client.segment(
                state.load_bytes(image_name, step=nn),
                text_prompt=prompt if has_prompt else None,
                point=point,
                min_score=min_score,
            )
        except ValueError as e:
            return ToolResult(
                data={
                    "error": str(e),
                    "step": nn,
                    "camera": camera,
                    "image_artifact": image_name,
                }
            )
        except Exception as e:
            return ToolResult(
                data={
                    "error": f"segmentation service call failed: {e}",
                    "step": nn,
                    "camera": camera,
                    "image_artifact": image_name,
                    "fallback": "Inspect the geometry request or obtain another observation.",
                }
            )

        # Reserve an index before writing; failed saves leave gaps, never reuse.
        with self._segment_index_lock:
            segment_index = self._segment_index
            self._segment_index += 1
        segment_name = f"segment_{segment_index:02d}.json"
        overlay_name = f"segment_overlay_{segment_index:02d}.png"
        saved_overlay = None
        saved_mask = None
        mask = data.mask
        if data.found and isinstance(mask, np.ndarray):
            saved_mask = state.save(
                f"segment_mask_{segment_index:02d}.npz",
                mask.astype(bool),
                step=nn,
            )
            try:
                world_map = state.load(world_name, step=nn)
            except Exception as exc:
                world_result = {
                    "world_xyz": None,
                    "world_error": f"world map artifact not available: {exc}",
                    "expected_world_artifact": world_name,
                }
            else:
                world_result = _mask_to_world(mask, world_map)
                world_result["world_artifact"] = world_name
            overlay = _make_segment_overlay(state.load(image_name, step=nn), mask)
            if overlay is not None and state.save(
                overlay_name,
                overlay,
                step=nn,
            ):
                saved_overlay = overlay_name
        else:
            world_result = {
                "world_xyz": None,
                "world_error": data.reason or "segmentation did not find a mask",
            }

        segment_blob = {
            "found": data.found,
            "mode": "text" if has_prompt else "point",
            "camera": camera,
            "source_step": nn,
            "segment_index": segment_index,
            "image_artifact": image_name,
            "min_score": min_score,
            "score": round(float(data.score), 3) if data.score is not None else None,
            "box": data.box,
            "mask_shape": list(data.mask_shape) if data.mask_shape else None,
            "mask_artifact": saved_mask,
        }
        if has_prompt:
            segment_blob["prompt"] = prompt
        else:
            segment_blob["point"] = point
        if not data.found:
            segment_blob["error"] = data.reason or "SAM3 found no mask"
        elif saved_mask is None:
            segment_blob["error"] = "failed to persist segment mask"
        segment_blob.update(world_result)
        saved_segment = state.save(
            segment_name,
            segment_blob,
            step=nn,
        )

        result = {
            "found": data.found,
            "step": nn,
            "camera": camera,
            "image_artifact": image_name,
            "score": segment_blob["score"],
            "box": segment_blob["box"],
            "world_xyz": segment_blob["world_xyz"],
            "world_error": segment_blob.get("world_error"),
            "mask_artifact": saved_mask,
        }
        if saved_segment is None:
            result["error"] = f"failed to persist segment artifact {segment_name}"
            result["code"] = "segment_artifact_save_failed"
            result["attempted_segment_artifact"] = segment_name
            if "error" in segment_blob:
                result["segmentation_error"] = segment_blob["error"]
            result["fallback"] = (
                "Inspect the geometry request or obtain another observation."
            )
        else:
            result["segment_artifact"] = saved_segment
        if data.found and saved_mask is None and saved_segment is not None:
            result["error"] = "failed to persist segment mask"
            result["code"] = "segment_mask_save_failed"
        if saved_segment is not None and "error" in segment_blob:
            result["error"] = segment_blob["error"]
            result["fallback"] = (
                "Inspect the geometry request or obtain another observation."
            )
        if saved_overlay is not None:
            result["overlay_artifact"] = saved_overlay
            images.append(state.load_bytes(saved_overlay, step=nn))
        return ToolResult(data=result, images=images)


def select_observation_artifacts(
    state: EnvState,
    record: StepRecord,
    camera: str,
) -> tuple[str | None, str | None, list[tuple[str | None, str | None]]]:
    if camera not in ("agentview", "wrist"):
        raise ValueError(f"unknown segment camera: {camera}")
    pairs = [
        (f"{camera}_high.png", f"{camera}_world_high.npz"),
        (f"{camera}.png", f"{camera}_world.npz"),
    ]
    for image_name, world_name in pairs:
        if (
            image_name in record.artifacts
            and world_name in record.artifacts
            and state.exists(image_name, step=record.step_idx)
            and state.exists(world_name, step=record.step_idx)
        ):
            return image_name, world_name, pairs
    return None, None, pairs


def _mask_to_world(
    mask: np.ndarray, world_map: np.ndarray, min_valid: int = 10
) -> dict:
    if world_map.ndim != 3 or world_map.shape[2] < 3:
        return {
            "world_xyz": None,
            "world_error": f"invalid world map shape: {tuple(world_map.shape)}",
            "n_pixels": int(mask.sum()),
            "n_valid": 0,
            "mask_resized_to_world_shape": False,
        }

    if mask.shape != world_map.shape[:2]:
        return {
            "world_xyz": None,
            "world_error": (
                f"mask/world shape mismatch: mask={tuple(mask.shape)}, "
                f"world={tuple(world_map.shape[:2])}"
            ),
            "n_pixels": int(mask.sum()),
            "n_valid": 0,
            "mask_resized_to_world_shape": False,
        }

    ys, xs = np.where(mask)
    if ys.size == 0:
        return {"world_xyz": None, "world_error": "empty mask"}

    pts = world_map[ys, xs].astype(np.float64)
    valid = np.isfinite(pts).all(axis=1) & (np.abs(pts).sum(axis=1) > 1e-6)
    pts = pts[valid]
    result = {
        "centroid_pixel": [
            int(round(float(np.median(xs)))),
            int(round(float(np.median(ys)))),
        ],
        "n_pixels": int(mask.sum()),
        "n_valid": int(pts.shape[0]),
        "mask_resized_to_world_shape": False,
    }
    if pts.shape[0] < min_valid:
        result.update(
            {
                "world_xyz": None,
                "world_error": f"too few valid depth pixels ({int(pts.shape[0])})",
            }
        )
        return result

    result["world_xyz"] = [
        round(float(np.median(pts[:, 0])), 4),
        round(float(np.median(pts[:, 1])), 4),
        round(float(np.median(pts[:, 2])), 4),
    ]
    return result


def _make_segment_overlay(
    image: np.ndarray,
    mask: np.ndarray,
) -> np.ndarray | None:
    if image.ndim != 3 or image.shape[:2] != mask.shape:
        return None
    overlay = image.copy()
    red = np.zeros_like(overlay)
    red[..., 0] = 255
    overlay[mask] = (
        0.55 * overlay[mask].astype(np.float32) + 0.45 * red[mask].astype(np.float32)
    ).astype(np.uint8)
    return overlay


def back_project(
    row: int | None = None,
    col: int | None = None,
    step: int = -1,
    camera: Literal["agentview", "wrist"] = "agentview",
    resolution: Literal["high", "low"] = "high",
    row_range: list[int] | None = None,
    col_range: list[int] | None = None,
    z_min: float | None = None,
    z_max: float | None = None,
    *,
    state: EnvState,
) -> ToolResult:
    """Read a world point or region from the selected recorded world map.

    Args:
        row: Pixel row in the selected camera and resolution.
        col: Pixel column in the selected camera and resolution.
        step: Observation step; -1 selects the latest record.
        camera: Recorded agentview or wrist camera.
        resolution: High or low world-map resolution.
        row_range: Optional region row bounds.
        col_range: Optional region column bounds.
        z_min: Optional lower world-z bound for a region.
        z_max: Optional upper world-z bound for a region.
        state: Shared observation and artifact store.
    """
    if camera not in ("agentview", "wrist"):
        return ToolResult(
            data={"error": f"bad camera '{camera}' (use 'agentview' or 'wrist')"}
        )
    if resolution not in ("high", "low"):
        return ToolResult(
            data={"error": f"bad resolution '{resolution}' (use 'high' or 'low')"}
        )

    region_mode = row_range is not None or col_range is not None
    if not region_mode and (row is None or col is None):
        return ToolResult(
            data={
                "error": (
                    "provide either (row, col) for a single pixel, or "
                    "row_range=[r0,r1] and col_range=[c0,c1] for a region center"
                )
            }
        )

    try:
        record = state.get(step)
    except Exception as e:
        return ToolResult(data={"error": f"state step not available: {e}"})
    nn = record.step_idx

    hi_artifact = f"{camera}_world_high.npz"
    low_artifact = f"{camera}_world.npz"
    source_artifact = hi_artifact if resolution == "high" else low_artifact
    if source_artifact not in record.artifacts:
        return ToolResult(
            data={
                "error": (
                    f"{camera} {resolution}-resolution world map not recorded for step {nn}"
                )
            }
        )

    try:
        world_map = state.load(str(source_artifact), step=nn)
    except Exception as e:
        return ToolResult(
            data={
                "error": (
                    f"{camera} {resolution}-resolution artifact not found "
                    f"for step {nn}: {e}"
                )
            }
        )

    height, width = world_map.shape[:2]

    if region_mode:
        if row_range is None or col_range is None:
            return ToolResult(
                data={
                    "error": (
                        "region mode needs BOTH row_range=[r0,r1] and col_range=[c0,c1]"
                    )
                }
            )
        try:
            r0, r1 = int(row_range[0]), int(row_range[1])
            c0, c1 = int(col_range[0]), int(col_range[1])
        except Exception:
            return ToolResult(
                data={"error": "row_range/col_range must each be [min, max] integers"}
            )
        r0, r1 = sorted((max(0, r0), min(height, r1)))
        c0, c1 = sorted((max(0, c0), min(width, c1)))
        if r1 <= r0 or c1 <= c0:
            return ToolResult(
                data={
                    "error": (
                        f"empty region after clamping to image {height}x{width}: "
                        f"rows [{r0},{r1}] cols [{c0},{c1}]"
                    )
                }
            )
        window = (
            world_map[r0:r1, c0:c1].reshape(-1, world_map.shape[2]).astype(np.float64)
        )
        finite = np.isfinite(window).all(axis=1) & (
            np.abs(window[:, :3]).sum(axis=1) > 1e-6
        )
        pts = window[finite]
        n_total = int(pts.shape[0])
        if z_min is not None:
            pts = pts[pts[:, 2] >= float(z_min)]
        if z_max is not None:
            pts = pts[pts[:, 2] <= float(z_max)]
        if pts.shape[0] < 8:
            return ToolResult(
                data={
                    "error": (
                        f"too few valid pixels in region after z-filter "
                        f"({int(pts.shape[0])}); widen the window or the z band"
                    ),
                    "n_valid_before_zfilter": n_total,
                }
            )
        xs, ys, zs = pts[:, 0], pts[:, 1], pts[:, 2]
        center = [
            round(float((xs.min() + xs.max()) / 2.0), 4),
            round(float((ys.min() + ys.max()) / 2.0), 4),
            round(float(np.median(zs)), 4),
        ]
        return ToolResult(
            data={
                "camera": camera,
                "resolution": resolution,
                "mode": "region",
                "row_range": [r0, r1],
                "col_range": [c0, c1],
                "z_band": [z_min, z_max],
                "center_xyz": center,
                "median_xyz": [
                    round(float(np.median(xs)), 4),
                    round(float(np.median(ys)), 4),
                    round(float(np.median(zs)), 4),
                ],
                "n_valid": int(pts.shape[0]),
                "step": nn,
                "image_size": [height, width],
                "source_artifact": source_artifact,
            }
        )

    if row < 0 or row >= height or col < 0 or col >= width:
        return ToolResult(
            data={
                "error": (
                    f"pixel ({row},{col}) out of bounds; {camera} image is {height}x{width}"
                )
            }
        )

    depth_m = None
    if source_artifact == low_artifact:
        try:
            depth_artifact = f"{camera}_depth.npz"
            if depth_artifact not in record.artifacts:
                raise FileNotFoundError(depth_artifact)
            depth = state.load(depth_artifact, step=nn)
            if depth.ndim == 3:
                depth = depth[..., 0]
        except Exception as e:
            return ToolResult(
                data={"error": f"{camera} depth not found for step {nn}: {e}"}
            )
        depth_m = float(depth[row, col])
        if not np.isfinite(depth_m) or depth_m <= 0 or depth_m > 10:
            return ToolResult(
                data={
                    "error": (
                        f"invalid {camera} depth {depth_m:.3f}m at pixel "
                        f"({row},{col}); pick a different pixel"
                    )
                }
            )
    world_xyz_raw = world_map[row, col]
    if (
        not np.isfinite(world_xyz_raw).all()
        or float(np.abs(world_xyz_raw[:3]).sum()) <= 1e-6
    ):
        return ToolResult(
            data={"error": f"invalid {camera} world xyz at pixel ({row},{col})"}
        )
    world_xyz = [round(float(v), 4) for v in world_xyz_raw[:3]]

    out = {
        "camera": camera,
        "resolution": resolution,
        "pixel": [row, col],
        "world_xyz": world_xyz,
        "step": nn,
        "image_size": [height, width],
        "source_artifact": source_artifact,
    }
    if depth_m is not None:
        out["depth_m"] = round(depth_m, 4)
    return ToolResult(data=out)
