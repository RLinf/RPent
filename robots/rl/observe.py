# Copyright 2026 The RPent Authors.
# Licensed under the Apache License, Version 2.0.
"""Episode-only observation graphs over immutable recorded images and masks."""

from __future__ import annotations

import itertools
import math
import re
import time
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from dataclasses import dataclass
from typing import Annotated, Literal
from uuid import uuid4

from PIL import Image
from pydantic import BaseModel, ConfigDict, Field, model_validator

from rpent.tools import ToolResult, tool
from rpent.utils.serialization import write_json

NodeId = Annotated[str, Field(pattern=r"^[A-Za-z][A-Za-z0-9_]{0,31}$")]
SourceId = Annotated[str, Field(min_length=1, max_length=256)]
Coordinate = Annotated[int, Field(ge=0)]
Pixel = Annotated[list[Coordinate], Field(min_length=2, max_length=2)]
Box = Annotated[list[Coordinate], Field(min_length=4, max_length=4)]
Range = Annotated[list[float], Field(min_length=2, max_length=2)]


class _Call(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)
    id: NodeId = Field(
        description="Unique local name in this observe call; references use @id."
    )


class ViewCall(_Call):
    op: Literal["view_env"]
    camera: str = Field(
        default="all",
        description="Named camera or all. Reads the latest recorded observation; no source.",
    )


class SegmentCall(_Call):
    model_config = ConfigDict(
        json_schema_extra={
            "oneOf": [
                {
                    "required": ["prompt"],
                    "properties": {
                        "prompt": {"type": "string"},
                        "point": {"type": "null"},
                    },
                },
                {
                    "required": ["point"],
                    "properties": {
                        "point": {"type": "array"},
                        "prompt": {"type": "null"},
                    },
                },
            ]
        }
    )
    op: Literal["segment"]
    source: SourceId = Field(
        description="An issued image_id, or @view_id.camera from this call. Never a mask."
    )
    prompt: (
        Annotated[str, Field(min_length=1, pattern=r"^[\s\S]*\S[\s\S]*$")] | None
    ) = Field(
        default=None,
        description="Short object description; supply prompt or point, not both.",
    )
    point: Pixel | None = Field(
        default=None,
        description="[row,col] in the displayed source image, not world coordinates.",
    )

    @model_validator(mode="after")
    def query(self):
        if (self.prompt is None) == (self.point is None):
            raise ValueError("segment needs exactly one prompt or point")
        return self


class ProjectCall(_Call):
    model_config = ConfigDict(
        json_schema_extra={
            "oneOf": [
                {"properties": {"pixel": {"type": "null"}, "box": {"type": "null"}}},
                {
                    "required": ["pixel"],
                    "properties": {
                        "pixel": {"type": "array"},
                        "box": {"type": "null"},
                        "z_range_m": {"type": "null"},
                    },
                },
                {
                    "required": ["box"],
                    "properties": {"box": {"type": "array"}, "pixel": {"type": "null"}},
                },
            ]
        }
    )
    op: Literal["back_project"]
    source: SourceId = Field(
        description="Image_id plus pixel/box, or mask_id/@segment_id without pixel/box."
    )
    pixel: Pixel | None = Field(
        default=None, description="One [row,col] surface point; image source only."
    )
    box: Box | None = Field(
        default=None,
        description="Half-open [row_min,col_min,row_max,col_max]; image source only, includes background.",
    )
    z_range_m: Range | None = Field(
        default=None,
        description="Optional inclusive world-z [low,high] metres for box/mask, never pixel.",
    )

    @model_validator(mode="after")
    def region(self):
        if self.pixel is not None and self.box is not None:
            raise ValueError("back_project accepts pixel or box, not both")
        if self.pixel is not None and self.z_range_m is not None:
            raise ValueError("z_range_m does not apply to pixel")
        if self.box is not None and not (
            self.box[0] < self.box[2] and self.box[1] < self.box[3]
        ):
            raise ValueError("box must have positive height and width")
        if self.z_range_m is not None and self.z_range_m[0] > self.z_range_m[1]:
            raise ValueError("z_range_m must be [low,high]")
        return self


Call = Annotated[ViewCall | SegmentCall | ProjectCall, Field(discriminator="op")]


@dataclass(frozen=True)
class ImageSource:
    image_id: str
    step: int
    camera: str
    native_size: tuple[int, int]  # height,width
    size: tuple[int, int]  # displayed height,width

    def metadata(self):
        return {
            "image_id": self.image_id,
            "step": self.step,
            "camera": self.camera,
            "size": list(self.size),
        }


@dataclass(frozen=True)
class Binding:
    image: ImageSource
    kind: Literal["image", "mask"]
    mask_id: str | None = None
    parent: str | None = None


@dataclass
class Plan:
    calls: dict[str, Call]
    bindings: dict[str, Binding]
    dependencies: dict[str, set[str]]
    latest_step: int


@dataclass
class Outcome:
    result: ToolResult
    status: str
    timing: dict[str, float]


class Observe:
    """Reuse primitive perception under one toolkit read lease; no physics here."""

    def __init__(self, toolkit):
        self.tk = toolkit
        self.primitive = toolkit._primitives
        self.cameras = getattr(self.primitive, "cameras", ("agentview", "wrist"))
        self.namespace = uuid4().hex[:8]
        self.images: dict[str, ImageSource] = {}
        self.masks: dict[str, ImageSource] = {}
        self.sequence = itertools.count(1)
        self.publish_current()

    def image_ids(self, step: int) -> dict[str, str]:
        """Name a recorded frame without reading pixels or changing the registry."""
        return {
            camera: f"img_{self.namespace}_{step}_{camera}" for camera in self.cameras
        }

    def publish_current(self) -> dict[str, str]:
        """Issue current image handles even when an action returns no RGB."""
        step = self.tk.state.latest_step
        result = {}
        for camera, image_id in self.image_ids(step).items():
            if image_id not in self.images:
                with Image.open(
                    self.tk.state.artifact_path(f"{camera}.png", step=step)
                ) as image:
                    width, height = image.size
                from robots.rl.toolkit import uses_native_1024

                size = (
                    (self.tk.image_size,) * 2
                    if uses_native_1024(self.tk.environment, self.tk.world_cord)
                    else (height, width)
                )
                self.images[image_id] = ImageSource(
                    image_id, step, camera, (height, width), size
                )
            result[camera] = image_id
        self._save_sources()
        return result

    def _save_sources(self):
        write_json(
            self.tk.output_dir / "observe" / "sources.json",
            {
                "images": {key: value.metadata() for key, value in self.images.items()},
                "masks": {key: value.image_id for key, value in self.masks.items()},
            },
        )

    def _image(self, step, camera):
        return self.images[f"img_{self.namespace}_{step}_{camera}"]

    def compile(self, calls: list[Call]) -> Plan:
        """Validate the entire graph and every source before any subcall starts."""
        if not 1 <= len(calls) <= self.tk.max_observe_calls:
            raise ValueError(f"observe needs 1..{self.tk.max_observe_calls} calls")
        by_id = {call.id: call for call in calls}
        if len(by_id) != len(calls):
            raise ValueError("observe IDs must be unique")
        latest = self.tk.state.latest_step
        references, dependencies = {}, {call.id: set() for call in calls}
        for call in calls:
            if isinstance(call, ViewCall):
                if call.camera not in {*self.cameras, "all"}:
                    raise ValueError(
                        f"unknown camera {call.camera}; choose {self.cameras} or all"
                    )
            elif call.source.startswith("@"):
                match = re.fullmatch(
                    r"@([A-Za-z][A-Za-z0-9_]{0,31})(?:\.([A-Za-z][A-Za-z0-9_]*))?",
                    call.source,
                )
                if match is None or match[1] not in by_id:
                    raise ValueError(f"unknown local source {call.source}")
                references[call.id] = (match[1], match[2])
                dependencies[call.id].add(match[1])
        visited, visiting = set(), set()

        def visit(name):
            if name in visiting:
                raise ValueError(f"dependency cycle at {name}")
            if name not in visited:
                visiting.add(name)
                for parent in dependencies[name]:
                    visit(parent)
                visiting.remove(name)
                visited.add(name)

        for name in by_id:
            visit(name)
        bindings = {}

        def bind(name):
            if name in bindings:
                return bindings[name]
            call = by_id[name]
            if name in references:
                parent, camera = references[name]
                producer = by_id[parent]
                if isinstance(producer, ViewCall):
                    available = (
                        self.cameras if producer.camera == "all" else (producer.camera,)
                    )
                    if camera not in available:
                        raise ValueError(
                            f"{call.source} must select a camera returned by {parent}"
                        )
                    value = Binding(self._image(latest, camera), "image", parent=parent)
                elif isinstance(producer, SegmentCall) and camera is None:
                    value = Binding(bind(parent).image, "mask", parent=parent)
                else:
                    raise ValueError(
                        f"{call.source} is not an image/mask source; use @view.camera or @segment"
                    )
            elif call.source in self.images:
                value = Binding(self.images[call.source], "image")
            elif call.source in self.masks:
                if call.source not in self.primitive._masks:
                    raise ValueError(f"mask no longer available: {call.source}")
                value = Binding(self.masks[call.source], "mask", mask_id=call.source)
            else:
                raise ValueError(
                    f"unknown source {call.source}; copy an issued image_id/mask_id, or use @local_id"
                )
            if isinstance(call, SegmentCall) and value.kind != "image":
                raise ValueError(f"{name}: segment requires an image source")
            if isinstance(call, ProjectCall):
                selectors = int(call.pixel is not None) + int(call.box is not None)
                if selectors != (1 if value.kind == "image" else 0):
                    raise ValueError(
                        f"{name}: image sources require pixel or box; mask sources require neither"
                    )
            self._validate_coordinates(call, value.image)
            bindings[name] = value
            return value

        for call in calls:
            if not isinstance(call, ViewCall):
                bind(call.id)
        return Plan(by_id, bindings, dependencies, latest)

    @staticmethod
    def _validate_coordinates(call, image):
        height, width = image.size
        for field in ("point", "pixel"):
            point = getattr(call, field, None)
            if point is not None and not (
                0 <= point[0] < height and 0 <= point[1] < width
            ):
                raise ValueError(f"{field} outside source image {height}x{width}")
        box = getattr(call, "box", None)
        if box is not None and not (
            0 <= box[0] < box[2] <= height and 0 <= box[1] < box[3] <= width
        ):
            raise ValueError(f"box outside source image {height}x{width}")

    @staticmethod
    def _pixel(point, image):
        return [
            min(n - 1, math.floor((value + 0.5) * n / d))
            for value, n, d in zip(point, image.native_size, image.size)
        ]

    @staticmethod
    def _box(box, image):
        return [
            math.floor(value * image.native_size[i % 2] / image.size[i % 2])
            if i < 2
            else math.ceil(value * image.native_size[i % 2] / image.size[i % 2])
            for i, value in enumerate(box)
        ]

    def _run(self, call, binding, step, mask_id, submitted):
        started = time.perf_counter()
        try:
            self.tk.raise_if_cancelled()
            if isinstance(call, ViewCall):
                cameras = self.cameras if call.camera == "all" else (call.camera,)
                images = [self._image(step, camera) for camera in cameras]
                result = ToolResult(
                    data={
                        "step": step,
                        "images": [image.metadata() for image in images],
                    },
                    images=self.primitive.images(step, list(cameras)),
                )
            elif isinstance(call, SegmentCall):
                image = binding.image
                result = self.primitive.segment(
                    prompt=call.prompt,
                    point=self._pixel(call.point, image)
                    if call.point is not None
                    else None,
                    camera=image.camera,
                    step=image.step,
                )
                result.data["image_id"] = image.image_id
                if "box" in result.data and image.native_size != image.size:
                    result.data["box"] = [
                        math.floor(v * image.size[i % 2] / image.native_size[i % 2])
                        if i < 2
                        else math.ceil(v * image.size[i % 2] / image.native_size[i % 2])
                        for i, v in enumerate(result.data["box"])
                    ]
            else:
                image = binding.image
                result = self.primitive.back_project(
                    pixel=self._pixel(call.pixel, image)
                    if call.pixel is not None
                    else None,
                    box=self._box(call.box, image) if call.box is not None else None,
                    mask_id=mask_id,
                    camera=image.camera if binding.kind == "image" else None,
                    step=image.step if binding.kind == "image" else None,
                    z_range_m=call.z_range_m,
                )
                result.data["image_id"] = image.image_id
            status = (
                "error"
                if result.is_error
                else "not_found"
                if result.data.get("code") == "not_found"
                else "ok"
            )
        except Exception as exc:
            result = ToolResult(
                data={"error": str(exc)}, model_text=f"{call.op} failed: {exc}"
            )
            status = "error"
        return Outcome(
            result,
            status,
            {
                "worker_queue_seconds": started - submitted,
                "started_at": started,
                "finished_at": time.perf_counter(),
            },
        )

    @tool(readonly=True)
    def observe(self, calls: Annotated[list[Call], Field(min_length=1)]) -> ToolResult:
        """Read or measure recorded images with 1..K subcalls. One flat list; independent calls run concurrently, source references determine dependencies. Each call needs a unique id and op. view_env(camera="all" or a named camera) returns latest RGB and image_id per camera; it has no source. segment(source,prompt OR point) returns one mask_id, box and visible-surface world_xyz (median), top_z_m (95th-percentile z), size_xyz_m (5th–95th percentile extent), axis_yaw (major xy heading [-pi/2,pi/2), null if not elongated), not an object centre or grasp pose. Geometry can be null when depth is insufficient. back_project(source,pixel OR box) measures an image; source=mask_id or @segment measures that mask without pixel/box. z_range_m is a world-z filter for box/mask only. source is an issued image_id/mask_id from earlier feedback, @view_id.camera, or @segment_id from this list. No camera/step/frame_from/mask_from fields on measurement subcalls. Pixels use displayed [row,col]; boxes are half-open. All arguments/dependencies are checked before anything runs. Missing detections return text; dependent calls are skipped, independent calls continue. Results follow input order; only view_env attaches images, with zero-based image_index. No intermediate model request occurs: to inspect a new image before choosing a query, use another reply. Example: {"calls":[{"id":"v","op":"view_env","camera":"all"},{"id":"cup","op":"segment","source":"@v.agentview","prompt":"red cup"},{"id":"xyz","op":"back_project","source":"@cup"}]}.

        Args:
            calls: 1..K strict subcalls; forward references are allowed, cycles are not. Copy persistent source IDs; @ references are local to this observe call.
        """
        plan = self.compile(calls)
        begin = time.perf_counter()
        outcomes, running, pending = {}, {}, dict(plan.calls)
        audit = {
            "latest_step": plan.latest_step,
            "calls": [call.model_dump(exclude_none=True) for call in calls],
            "dependencies": {
                key: sorted(value) for key, value in plan.dependencies.items()
            },
            "nodes": {},
        }
        try:
            with ThreadPoolExecutor(
                max_workers=min(self.tk.observe_concurrency, len(calls))
            ) as executor:
                while pending or running:
                    self.tk.raise_if_cancelled()
                    for name, call in list(pending.items()):
                        if len(running) >= self.tk.observe_concurrency:
                            break
                        if not plan.dependencies[name] <= outcomes.keys():
                            continue
                        binding = plan.bindings.get(name)
                        mask_id = binding.mask_id if binding else None
                        parent = (
                            outcomes.get(binding.parent)
                            if binding and binding.parent
                            else None
                        )
                        if parent is not None:
                            if parent.status != "ok":
                                outcomes[name] = Outcome(
                                    ToolResult(
                                        model_text=f"Skipped: source {binding.parent} did not produce the required image/mask."
                                    ),
                                    "skipped",
                                    {},
                                )
                                del pending[name]
                                continue
                            if binding.kind == "mask":
                                mask_id = parent.result.data.get("mask_id")
                                if mask_id is None:
                                    outcomes[name] = Outcome(
                                        ToolResult(
                                            model_text=f"Skipped: {binding.parent} returned no mask."
                                        ),
                                        "skipped",
                                        {},
                                    )
                                    del pending[name]
                                    continue
                        submitted = time.perf_counter()
                        audit["nodes"][name] = {
                            "schedule_wait_seconds": submitted - begin,
                            "image_id": binding.image.image_id if binding else None,
                        }
                        running[
                            executor.submit(
                                self._run,
                                call,
                                binding,
                                plan.latest_step,
                                mask_id,
                                submitted,
                            )
                        ] = name
                        del pending[name]
                    if running:
                        completed, _ = wait(running, return_when=FIRST_COMPLETED)
                        for future in completed:
                            name = running.pop(future)
                            outcomes[name] = future.result()
                    elif pending and not any(
                        plan.dependencies[name] <= outcomes.keys() for name in pending
                    ):
                        raise RuntimeError(
                            "validated observation graph made no progress"
                        )
        finally:
            for name, outcome in outcomes.items():
                audit["nodes"].setdefault(name, {}).update(
                    status=outcome.status,
                    **outcome.timing,
                    result=outcome.result.to_text(),
                )
            audit["elapsed_seconds"] = time.perf_counter() - begin
            write_json(
                self.tk.output_dir / "observe" / f"{next(self.sequence):04d}.json",
                audit,
            )

        results, images, indices = [], [], {}
        for call in calls:
            outcome = outcomes[call.id]
            result = outcome.result
            if result.data.get("mask_id"):
                self.masks[result.data["mask_id"]] = plan.bindings[call.id].image
            if isinstance(call, ViewCall) and outcome.status == "ok":
                for meta, data in zip(result.data["images"], result.images):
                    image_id = meta["image_id"]
                    if image_id not in indices:
                        indices[image_id] = len(images)
                        images.append(data)
                    meta["image_index"] = indices[image_id]
            results.append(
                {
                    "id": call.id,
                    "op": call.op,
                    "result": result.to_text()
                    if result.model_text is not None or result.is_error
                    else result.data,
                }
            )
        self._save_sources()
        stats = self.tk.workflow_stats
        for call in calls:
            name = "observe_" + call.op + "_calls"
            stats[name] = stats.get(name, 0) + 1
        stats["observe_calls"] = stats.get("observe_calls", 0) + 1
        stats["observe_skipped"] = stats.get("observe_skipped", 0) + sum(
            value.status == "skipped" for value in outcomes.values()
        )
        return ToolResult(data={"results": results}, images=images)
