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

"""Shared plumbing for planner tools that read recorded camera images."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Annotated, Any, Literal, Protocol

import numpy as np
from pydantic import Field, create_model

from rpent.tools.base import Tool, ToolResult

if TYPE_CHECKING:
    from rpent.session import EnvState, StepRecord


class Localizer(Protocol):
    """Locate pixels of one recorded camera image in the robot's world frame."""

    @property
    def artifacts(self) -> tuple[str, ...]:
        """Return the artifacts that must be recorded alongside the image."""
        ...

    def locate_mask(
        self, mask: np.ndarray, state: EnvState, step: int
    ) -> dict[str, Any]:
        """Return ``world_xyz``, or ``None`` with ``world_error``, plus diagnostics."""
        ...


@dataclass(frozen=True)
class WorldMapLocalizer:
    """Localize masks through a pixel-aligned world-XYZ map in meters."""

    world_map: str

    @property
    def artifacts(self) -> tuple[str, ...]:
        return (self.world_map,)

    def locate_mask(
        self, mask: np.ndarray, state: EnvState, step: int
    ) -> dict[str, Any]:
        """Return the mask's median ``world_xyz``, or ``None`` with ``world_error``."""
        try:
            world_map = state.load(self.world_map, step=step)
        except Exception as exc:
            return {
                "world_xyz": None,
                "world_error": f"world map artifact not available: {exc}",
                "expected_world_artifact": self.world_map,
            }
        if world_map.ndim != 3 or world_map.shape[2] < 3:
            return {
                "world_xyz": None,
                "world_error": f"invalid world map shape: {tuple(world_map.shape)}",
                "n_pixels": int(mask.sum()),
                "n_valid": 0,
                "mask_resized_to_world_shape": False,
                "world_artifact": self.world_map,
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
                "world_artifact": self.world_map,
            }

        ys, xs = np.where(mask)
        if ys.size == 0:
            return {
                "world_xyz": None,
                "world_error": "empty mask",
                "world_artifact": self.world_map,
            }
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
        if pts.shape[0] < 10:
            result["world_xyz"] = None
            result["world_error"] = f"too few valid depth pixels ({int(pts.shape[0])})"
        else:
            result["world_xyz"] = [
                round(float(np.median(pts[:, 0])), 4),
                round(float(np.median(pts[:, 1])), 4),
                round(float(np.median(pts[:, 2])), 4),
            ]
        result["world_artifact"] = self.world_map
        return result


@dataclass(frozen=True)
class CameraView:
    """One recorded camera image and the localizer for its pixels."""

    image: str
    localizer: Localizer

    @property
    def artifacts(self) -> tuple[str, ...]:
        """Return the artifacts that must be recorded for this view."""
        return (self.image, *self.localizer.artifacts)


class PerceptionToolBase:
    """Base for planner tools that read a robot's recorded camera images.

    Subclasses declare class-level ``@tool`` methods that take a ``camera: str``
    input. Robot differences are constructor data rather than subclasses: a
    robot builds one instance per toolkit and registers each of its tools
    through :func:`with_cameras`, which publishes the robot's camera names.
    """

    #: Error reported when no view of the requested camera is fully recorded.
    missing_artifacts_error = "complete camera artifacts not found"

    def __init__(
        self,
        *,
        state: EnvState,
        cameras: dict[str, Sequence[CameraView]],
        fallback: str,
    ) -> None:
        """Bind the artifact store and the robot's camera configuration.

        Args:
            state: Run store used for both input and output artifacts.
            cameras: Camera names mapped to their views in preference order.
                The first camera is the default.
            fallback: Robot-specific guidance returned when a tool fails.
        """
        if not isinstance(cameras, dict) or not cameras:
            raise ValueError("cameras must be a non-empty dict of camera views")
        for camera, views in cameras.items():
            if not isinstance(camera, str) or not camera:
                raise ValueError(f"camera names must be non-empty strings: {camera!r}")
            if not views or not all(isinstance(view, CameraView) for view in views):
                raise ValueError(f"camera {camera!r} needs at least one CameraView")
        self._state = state
        self._cameras = {camera: tuple(views) for camera, views in cameras.items()}
        self._fallback = fallback

    def _select_view(self, record: StepRecord, camera: str) -> CameraView | ToolResult:
        """Return the first view of ``camera`` whose artifacts ``record`` holds."""
        views = self._cameras.get(camera)
        if views is None:
            return ToolResult(
                data={
                    "error": f"unknown camera {camera!r}",
                    "cameras": list(self._cameras),
                }
            )
        for view in views:
            if all(
                name in record.artifacts
                and self._state.exists(name, step=record.step_idx)
                for name in view.artifacts
            ):
                return view
        return ToolResult(
            data={
                "error": self.missing_artifacts_error,
                "step": record.step_idx,
                "camera": camera,
                "checked_artifacts": [
                    name for view in views for name in view.artifacts
                ],
            }
        )


def with_cameras(declaration: Tool, cameras: Iterable[str]) -> Tool:
    """Limit a declaration's ``camera`` input to a robot's cameras.

    The first camera is the published default. The handler is kept, so a bound
    declaration from ``iter_tools(instance)`` stays ready for registration.
    """
    names = tuple(cameras)
    if not names:
        raise ValueError("with_cameras needs at least one camera")
    schema = declaration.input_schema
    camera = schema["properties"].get("camera")
    if camera is None:
        raise ValueError(f"tool {declaration.name!r} has no camera input")
    description = f"{camera['description'].rstrip('.')} (default {names[0]})."
    schema["properties"]["camera"] = {
        "description": description,
        "enum": list(names),
        "type": "string",
    }
    args_schema = create_model(
        declaration.args_schema.__name__,
        __base__=declaration.args_schema,
        __module__=declaration.args_schema.__module__,
        camera=(
            Annotated[Literal[names], Field(description=description)],
            names[0],
        ),
    )
    return replace(declaration, args_schema=args_schema, _input_schema=schema)
