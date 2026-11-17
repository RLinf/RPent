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

"""Explicit bounded plan and exact-frame perception for snapshot requests."""

from __future__ import annotations

import io
from concurrent.futures import ThreadPoolExecutor
from typing import Annotated, Literal

import numpy as np
from PIL import Image, ImageDraw
from pydantic import BaseModel, Field, StringConstraints

from robots.rl import geometry
from robots.rl.images import scale_box
from rpent.tools import ToolResult, tool

Query = Annotated[str, StringConstraints(min_length=1, max_length=120, pattern=r"\S")]
TodoItem = Annotated[
    str, StringConstraints(min_length=1, max_length=320, pattern=r"\S")
]


class Snapshot:
    """Keep only the model's current queries, todo and last execution result."""

    def __init__(
        self,
        primitives,
        *,
        environment: str,
        task: str,
        max_episode_steps: int = 10000,
        image_size: int = 1024,
    ) -> None:
        self.primitives = primitives
        self.environment = environment
        self.task = task
        self.max_episode_steps = max_episode_steps
        self.image_size = image_size
        self.todo_revision = 0
        self.queries: list[str] = []
        self.todo: list[str] | list[dict] = []
        self.generation = 0
        self.last_result: dict | None = None
        self._cache_key: tuple[int, int] | None = None
        self._perception: tuple[list, list[bytes]] = ([], [])

    @tool(readonly=True)
    def change_query(
        self, queries: Annotated[list[Query], Field(min_length=1, max_length=8)]
    ) -> ToolResult:
        """Replace the perception targets with short visible-object descriptions. These queries run automatically on every current camera frame. Include destinations and relevant parts as well as manipulated objects. All matching instances are reported; a missing detection is not evidence of absence. This changes perception only and never moves the robot. Returns status, query_generation, image_size and the accepted queries. Example: {"queries":["red cup","plate"]}.

        Args:
            queries: Full replacement, 1 to 8 object or part descriptions; no coordinates or task scripts.
        """
        if any(not query.strip() for query in queries):
            return ToolResult(data={"error": "queries must not be blank"})
        self.queries = list(queries)
        self.generation += 1
        return ToolResult(
            data={
                "status": "updated",
                "query_generation": self.generation,
                "image_size": self.image_size,
                "queries": self.queries,
            }
        )

    @tool(readonly=True)
    def update_todo(
        self, items: Annotated[list[TodoItem], Field(min_length=1, max_length=8)]
    ) -> ToolResult:
        """Replace the entire persistent todo with a compact working plan. Earlier reasoning is absent from the next request: retain the goal, verified progress, immediate next subgoal and any uncertainty needed to continue. Separate observations from intentions; do not mark an intended action as completed. Prefer a few useful items over a motion transcript. This records your plan only; it does not execute it or determine task success. Returns status and the accepted todo. Example: {"items":["Place the cup on the plate; verify release"]}.

        Args:
            items: Complete replacement plan, 1 to 8 items of at most 320 characters each.
        """
        if any(not item.strip() for item in items):
            return ToolResult(data={"error": "todo items must not be blank"})
        self.todo = list(items)
        self.todo_revision += 1
        return ToolResult(data={"status": "updated", "todo": self.todo})

    def observe(self, robot_state: dict) -> ToolResult:
        """Use recorded RGB/depth from one step; never mix perception across steps."""
        primitive = self.primitives
        step = primitive._state.latest_step
        key = (step, self.generation)
        if key != self._cache_key:
            cameras = (
                ("agentview", "wrist")
                if self.environment == "libero"
                else primitive.cameras
            )
            detections, images = [], []
            raw_images = [primitive.images(step, [camera])[0] for camera in cameras]
            with ThreadPoolExecutor(max_workers=min(2, len(cameras))) as executor:
                pending = (
                    [
                        executor.submit(
                            primitive._sam3.segment_queries, raw, self.queries
                        )
                        for raw in raw_images
                    ]
                    if self.queries
                    else []
                )
                for index, camera in enumerate(cameras):
                    image = primitive._state.load(f"{camera}.png", step=step)
                    world = primitive._state.load(f"{camera}_world.npz", step=step)
                    view = Image.fromarray(np.asarray(image, dtype=np.uint8)).convert(
                        "RGB"
                    )
                    view = view.resize(
                        (self.image_size, self.image_size), Image.Resampling.LANCZOS
                    )
                    draw = ImageDraw.Draw(view)
                    try:
                        groups = pending[index].result() if self.queries else []
                    except Exception as exc:
                        primitive._audit_sam(
                            step=step,
                            camera=camera,
                            query=list(self.queries),
                            min_score=0.2,
                            error=exc,
                        )
                        raise
                    rows = []
                    for query, matches in zip(self.queries, groups, strict=True):
                        if not matches:
                            primitive._audit_sam(
                                step=step, camera=camera, query=query, min_score=0.2
                            )
                        for match in matches:
                            primitive._audit_sam(
                                step=step,
                                camera=camera,
                                query=query,
                                min_score=0.2,
                                result=match,
                            )
                            if not match.found or match.mask is None:
                                continue
                            if match.mask.shape != world.shape[:2]:
                                raise ValueError(
                                    "SAM3 mask and recorded depth dimensions differ"
                                )
                            box = geometry.box_from_mask(match.mask)
                            r0, c0, r1, c1 = box
                            display_box = scale_box(
                                box, world.shape[0], self.image_size
                            )
                            identifier = f"{camera}:{len(rows) + 1}"
                            rows.append(
                                {
                                    "id": identifier,
                                    "query": query,
                                    "box_rc": display_box,
                                    "surface": geometry.surface_geometry(
                                        primitive.surface_points(world[match.mask])
                                    ),
                                    "box_surface": geometry.surface_geometry(
                                        primitive.surface_points(world[r0:r1, c0:c1])
                                    ),
                                }
                            )
                            dr0, dc0, dr1, dc1 = display_box
                            draw.rectangle(
                                (dc0, dr0, dc1 - 1, dr1 - 1), outline="red", width=2
                            )
                            draw.text((dc0, max(0, dr0 - 12)), identifier, fill="red")
                    buffer = io.BytesIO()
                    view.save(buffer, format="PNG")
                    encoded = buffer.getvalue()
                    primitive._save(
                        f"zap-q{self.generation}-{camera}.png", np.asarray(view), step
                    )
                    detections.append(
                        {
                            "camera": camera,
                            "step": step,
                            "size": [self.image_size, self.image_size],
                            "objects": rows,
                        }
                    )
                    images.append(encoded)
            self._cache_key, self._perception = key, (detections, images)
        detections, images = self._perception
        return ToolResult(
            data={
                "task": self.task,
                "step": step,
                "state": robot_state,
                "env_steps": primitive.env_steps,
                "env_steps_remaining": max(
                    0, self.max_episode_steps - primitive.env_steps
                ),
                "query_generation": self.generation,
                "image_size": self.image_size,
                "queries": self.queries,
                "perception": detections,
                "todo": self.todo,
                "last_result": self.last_result,
            },
            images=list(images),
        )


class TodoEntry(BaseModel):
    """One model-visible subtask and its explicitly reported status."""

    text: TodoItem
    status: Literal["pending", "in_progress", "done"]


class SubtaskPlan:
    """Store the complete visible plan and signal a new conversation boundary."""

    def __init__(self, snapshot: Snapshot) -> None:
        self.snapshot = snapshot

    @tool(readonly=True)
    def update_todo(
        self, items: Annotated[list[TodoEntry], Field(min_length=1, max_length=8)]
    ) -> ToolResult:
        """Replace the entire subtask list and clear the conversation before the next request. Keep completed items marked done, mark the current meaningful goal in_progress, and retain remaining goals as pending. Mark progress only after verifying the current observation; intentions are not completed actions. Each item should describe an outcome, not one small motion. The next context contains only this list, the task, current queries and observation: include what is needed to continue. Calling this tool always creates a new context, including when revising an unfinished subtask. It does not move the robot or determine task success. Returns status, todo, todo_revision and context_reset. Example: {"items":[{"text":"Cup released on the plate","status":"done"},{"text":"Close the drawer","status":"in_progress"}]}.

        Args:
            items: Complete list of 1 to 8 subtasks, each with text and pending/in_progress/done status.
        """
        self.snapshot.todo = [item.model_dump() for item in items]
        self.snapshot.todo_revision += 1
        return ToolResult(
            data={
                "status": "updated",
                "todo": self.snapshot.todo,
                "todo_revision": self.snapshot.todo_revision,
                "context_reset": True,
            }
        )
