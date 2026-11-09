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

"""LeKiwi planner tools, primitives, and canonical state capture."""

from __future__ import annotations

import time
from collections.abc import Callable, Sequence
from typing import Annotated, Any

import numpy as np
from pydantic import Field

from rpent.session import EnvState, StepRecord
from rpent.tools import ToolResult, tool
from rpent.utils.logging import get_logger

logger = get_logger(__name__)


def coerce_vec3(value: Sequence[float], *, name: str) -> np.ndarray:
    """Return a finite float32 three-vector or raise a useful error."""
    array = np.asarray(value, dtype=np.float32)
    if array.shape != (3,):
        raise ValueError(f"{name} must contain exactly 3 values, got {array.shape}")
    if not np.isfinite(array).all():
        raise ValueError(f"{name} must contain only finite values")
    return array


class LeKiwiPrimitives:
    """Safe agent-facing operations over a remote LeKiwi environment."""

    def __init__(
        self,
        *,
        env: Any,
        check_cancelled: Callable[[], None],
    ) -> None:
        self.env = env
        self._check_cancelled = check_cancelled

    @tool
    def drive(
        self,
        velocity: Annotated[list[float], Field(min_length=3, max_length=3)],
        duration_s: Annotated[float, Field(gt=0.0, le=10.0)] = 1.0,
    ) -> ToolResult:
        """Drive the base with one body velocity [x, y, theta] held for a bounded duration.

        Units are m/s, m/s, and deg/s in the base frame. The command is
        clipped to the configured velocity limits, and the wheels stop
        when the duration ends.
        """
        self._check_cancelled()
        data = self.env.drive(
            coerce_vec3(velocity, name="velocity"), duration_s=duration_s
        )
        return ToolResult(data=data)

    @tool(readonly=True)
    def list_map_nodes(
        self,
        label: Annotated[str | None, Field(min_length=1, max_length=128)] = None,
    ) -> ToolResult:
        """List known navigation node IDs and their existing labels.

        Args:
            label: Exact existing label to filter by; omit to list all nodes.
                This is not semantic or object search.

        Returns node_id and label only; an empty label means the node is unnamed.
        Use inspect_map_node for a reference image and pose, or view_map for
        spatial context. Does not move the robot or start navigation.
        """
        self._check_cancelled()
        result = self.env.list_map_nodes(label=label)
        return ToolResult(
            data={
                "nodes": [
                    {"node_id": node["node_id"], "label": node["label"]}
                    for node in result["nodes"]
                ]
            }
        )

    @tool(readonly=True)
    def inspect_map_node(self, node_id: Annotated[int, Field(gt=0)]) -> ToolResult:
        """Inspect a known node's map pose and historical reference image.

        Args:
            node_id: Positive node ID returned by list_map_nodes or view_map.

        The node pose is the recorded robot viewpoint, not an object's position.
        The image and observed_at_s come from the database, not the live camera;
        missing images are reported by image_available=false. Use view_env_state
        for current visual confirmation.
        """
        data = dict(self.env.inspect_map_node(node_id))
        encoded = data.pop("image_png", None)
        return ToolResult(
            data=data, images=[] if encoded is None else [encoded.tobytes()]
        )

    @tool(readonly=True)
    def view_map(
        self,
        center_node_id: Annotated[int | None, Field(gt=0)] = None,
        radius_m: Annotated[float | None, Field(ge=0.1, le=100)] = None,
    ) -> ToolResult:
        """View the static 2D occupancy map, place labels and available robot pose.

        Args:
            center_node_id: Known node to highlight; required when radius_m is set.
            radius_m: Crop around that node in meters; omit to show the full map.

        Returns up to 100 nodes plus the selected node, retaining their IDs in
        the data. The image marks named places; only the selected node shows its
        ID, even when unnamed. nodes_truncated indicates more nodes in the region;
        use list_map_nodes to browse IDs. robot.valid=false means no current pose
        is drawn. The map excludes live obstacles and does not
        prove a route is safe or reachable. Navigate by node ID or existing label,
        not by estimating coordinates from image pixels.
        """
        data = dict(self.env.view_map(center_node_id=center_node_id, radius_m=radius_m))
        image = data.pop("image_png")
        return ToolResult(data=data, images=[image.tobytes()])

    @tool(readonly=True)
    def get_navigation_state(self) -> ToolResult:
        """Read navigation readiness, map pose and the current or last operation.

        Pose, when available, is [x_m, y_m, yaw_rad] in the map frame.
        After stop, wait until operation.state is no longer running before
        submitting another goal. An unavailable service is not restarted.
        """
        return ToolResult(data=self.env.get_navigation_state())

    @tool
    def navigate_to(
        self,
        target: int | str,
        timeout_s: Annotated[float, Field(gt=0.0, allow_inf_nan=False)] = 60.0,
    ) -> ToolResult:
        """Navigate to an RTAB-Map node ID or label and wait for the result.

        Args:
            target: Existing RTAB-Map node ID (integer) or label (string).
                The arrival position and heading come from that map node.
            timeout_s: Positive maximum wait in seconds after submission.

        Requires an already running navigation service with valid localization
        and perception. Returns succeeded, failed or cancelled; a Nav2 success
        does not establish physical clearance or visual confirmation of arrival.
        Interruption, lost status or timeout requests a stop; services are not
        restarted and goals are not retried.
        """
        self._check_cancelled()
        operation = self.env.navigate_to(target, timeout_s=timeout_s)
        deadline = time.monotonic() + timeout_s
        try:
            while True:
                self._check_cancelled()
                current = self.env.get_navigation_state()["operation"]
                if current["id"] != operation["id"]:
                    raise RuntimeError("Navigation operation changed while waiting")
                if current["state"] in {"succeeded", "failed", "cancelled"}:
                    data = dict(current)
                    if current["state"] != "succeeded":
                        data.setdefault("error", f"Navigation {current['state']}")
                    return ToolResult(data=data)
                if current["state"] != "running":
                    raise RuntimeError("Unknown navigation operation state")
                if time.monotonic() >= deadline:
                    raise TimeoutError("Timed out waiting for navigation result")
                time.sleep(0.1)
        except BaseException:
            try:
                self.env.stop()
            except Exception:
                logger.exception("Could not request stop after navigation interruption")
            raise

    @tool
    def stop(self) -> ToolResult:
        """Stop the base and cancel its current goal, keeping navigation running.

        Wait for the navigation operation to leave running before another goal.
        """
        self._check_cancelled()
        data = self.env.stop()
        return ToolResult(data=data)


def dump_state(
    primitives: LeKiwiPrimitives,
    state: EnvState,
    *,
    command: dict[str, Any] | None,
    result: dict[str, Any] | None,
    elapsed_s: float | None,
) -> StepRecord:
    """Capture robot state and the synchronized front-camera frame in ``EnvState``."""
    observation = primitives.env.get_observation()
    robot_state = primitives.env.get_robot_state()
    metadata = primitives.env.get_camera_meta()
    with state.record_step(
        state=robot_state,
        command=command,
        result=result,
        elapsed_s=elapsed_s,
    ) as step:
        main_image = observation.get("main_images")
        if main_image is not None:
            state.save("front.png", main_image, step=step)
        if metadata is not None:
            state.save("camera_meta.json", metadata, step=step)
    return state.get(step)


@tool(readonly=True, exclude=("state",))
def view_env_state(
    step: Annotated[int, Field(json_schema_extra={"default": -1})] = -1,
    *,
    state: EnvState,
) -> ToolResult:
    """Read a LeKiwi state snapshot and its synchronized front-camera image."""
    images: list[bytes] = []
    record = state.get(step)
    output = record.to_blob()
    if state.exists("front.png", step=record.step_idx):
        output["image_front_path"] = str(
            state.artifact_path("front.png", step=record.step_idx)
        )
        images.append(state.load_bytes("front.png", step=record.step_idx))
    return ToolResult(data=output, images=images)
