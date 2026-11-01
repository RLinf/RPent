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

"""Robot-agnostic workspace mapping tools.

:class:`MappingMixin` turns any :class:`~rpent.tools.toolkit.Toolkit` whose
robot can serve a synchronized RGB frame and a base pose into a mapping agent:

1. ``build_map`` opens a LingBot-Map session, asks the operator to teleoperate
   the robot around the workspace while a poller thread pushes frames and
   poses, finalizes the map when the operator replies, optionally labels
   requested objects on keyframes with Grounding DINO + SAM 2, and persists the
   map bundle as ``EnvState`` artifacts.
2. ``detect_objects`` runs Grounding DINO + SAM 2 standalone on any persisted
   camera frame, so the perception service stays usable outside mapping.
3. ``check_clearance`` localizes a live frame against a built map and reports
   whether the floor ahead is clear enough to drive on. It needs a
   localization service rather than a mapping session; the caller still
   decides how to move.

The robot-specific slice is two methods the mixin calls: ``_map_frame()``
returns ``(rgb, pose, timestamp)`` for the current sensor read, and the
toolkit's ``get_env_state`` keeps recording steps as usual while the operator
drives.
"""

from __future__ import annotations

import threading
import time
from functools import partial
from typing import TYPE_CHECKING, Annotated, Any, Literal, Protocol

from pydantic import Field

from rpent.tools import ToolResult, tool
from rpent.tools.clearance import (
    DEFAULT_HALF_WIDTH,
    DEFAULT_LENGTH,
    DEFAULT_UNIT,
)
from rpent.tools.ground_map import (
    ClientMapSource,
    ground_map,
    save_grounded_map_via_client,
)
from rpent.utils.logging import get_logger
from toolkits.grounding.client import GdinoSam2Client
from toolkits.localization.client import LocalizeClient
from toolkits.mapping.client import (
    DEFAULT_PUSH_TIMEOUT_S,
    LingBotMapClient,
)

if TYPE_CHECKING:
    from rpent.tools.base import Tool
    from rpent.tools.toolkit import Toolkit

logger = get_logger("mapping")

#: Default sampling cadence for frames pushed during teleoperation.
DEFAULT_FRAME_INTERVAL_S = 0.5


class MapFrameSource(Protocol):
    """The robot-specific sensor read the mapping mixin depends on."""

    def _map_frame(self) -> tuple[Any, list[float] | None, float]:
        """Return ``(rgb, base_pose, timestamp_s)`` for the current read.

        ``rgb`` is an RGB ``[H,W,3]`` array or PNG bytes. ``base_pose`` is the
        robot base pose in the map frame as a flat 6-vector
        ``[x, y, yaw, roll, pitch, yaw_head]``, or ``None`` when the robot
        does not expose odometry and the mapping service must estimate pose
        from frames alone. When every pushed frame carries a pose, the mapping
        service aligns the reconstruction to those poses, which fixes its
        metric scale and puts it in the robot map frame; without poses the map
        keeps the model's arbitrary scale.
        """


class MappingMixin:
    """Teleop-driven map construction and map-based clearance checks.

    Hosts provide ``self._lingbotmap_client``, ``self._gdino_sam2_client``,
    and/or ``self._localize_client`` plus ``_map_frame()``; ``build_map`` needs
    the first, ``detect_objects`` the second, and ``check_clearance`` the
    third. ``self._operator_input`` (a ``HumanInTheLoopInput`` or
    ``OperatorReader``) supplies the human side. Register the tools with
    :meth:`register_mapping_tools` after the toolkit holds its primitives.
    """

    _frame_interval_s: float = DEFAULT_FRAME_INTERVAL_S

    def register_mapping_tools(self) -> None:
        """Register the tools whose backing service is connected.

        ``build_map`` needs the LingBot-Map service, ``detect_objects`` the
        perception service, and ``check_clearance`` the localization service.
        A toolkit without a service stays a plain agent whose model is never
        offered a tool that would only fail at call time.
        """
        if getattr(self, "_lingbotmap_client", None) is not None:
            self.add_tool(self.build_map)  # type: ignore[attr-defined]
        if getattr(self, "_gdino_sam2_client", None) is not None:
            definition: Tool = self.detect_objects  # type: ignore[attr-defined]
            self.add_tool(
                definition.with_handler(
                    partial(definition, state=self.state)  # type: ignore[attr-defined]
                )
            )
        if getattr(self, "_localize_client", None) is not None:
            self.add_tool(self.check_clearance)  # type: ignore[attr-defined]

    # ---- robot-provided hooks --------------------------------------------

    def _map_frame(self) -> tuple[Any, list[float] | None, float]:
        """Return the current synchronized ``(rgb, base_pose, timestamp_s)``."""
        raise NotImplementedError("the robot toolkit must provide _map_frame")

    def _ask_operator(self, prompt: str, *, kind: str | None = None) -> str | None:
        """Wait for the operator's reply, or ``None`` when no channel exists."""
        operator_input = getattr(self, "_operator_input", None)  # type: ignore[attr-defined]
        if operator_input is not None:
            request = getattr(operator_input, "request", None)
            if callable(request):
                return request(prompt, self.raise_if_cancelled, kind=kind)  # type: ignore[attr-defined]
            return operator_input(prompt, self.raise_if_cancelled)  # type: ignore[attr-defined]
        # No operator channel was wired in (robots without human-interactive
        # exploration); a plain TTY still lets a terminal operator reply.
        import sys

        if sys.stdin is None or not sys.stdin.isatty():
            return None
        try:
            return input(f"{prompt}\nReply: ").strip()
        except EOFError:
            return None

    # ---- tools -------------------------------------------------------------

    @tool
    def build_map(
        self,
        *,
        map_name: str,
        labels: Annotated[
            list[str],
            Field(
                max_length=16,
                json_schema_extra={"default": []},
            ),
        ] = [],
    ) -> ToolResult:
        """Build a workspace map by having the operator teleoperate the robot.

        Opens a LingBot-Map mapping session, then asks the operator to
        teleoperate the robot around the workspace; frames and base poses are
        pushed while they drive. When the operator replies, the map is built
        and saved. When labels are given, Grounding DINO + SAM 2 runs on archived
        keyframes; on maps with depth each mask is lifted into the map frame,
        associated across frames, and written to the map as ``objects.json``,
        per-frame overlays, and a semantic point cloud. The frame-archive
        backend has no depth, so its annotations stay 2D.

        Args:
            map_name: Name for the saved map; also names the map artifacts.
            labels: Optional object labels (at most 16) to ground on archived
                keyframes with Grounding DINO + SAM 2.
        """
        map_client: LingBotMapClient = self._require_map_client()  # type: ignore[attr-defined]
        name = str(map_name).strip()
        if not name:
            return ToolResult(data={"error": "map_name must be a non-empty string"})
        client_labels = [str(label).strip() for label in labels or []]
        if any(not label for label in client_labels):
            return ToolResult(data={"error": "labels must be non-empty strings"})
        if client_labels and getattr(self, "_gdino_sam2_client", None) is None:
            return ToolResult(
                data={
                    "error": "labels require the Grounding DINO + SAM 2 service; "
                    "restart with --gdino-sam2-endpoint or omit labels"
                }
            )

        started = time.perf_counter()
        try:
            status = map_client.start_session(map_name=name)
        except Exception as exc:
            return ToolResult(data={"error": f"could not start mapping session: {exc}"})

        poller = _FramePoller(
            source=self,  # type: ignore[arg-type]
            map_client=map_client,
            interval_s=self._frame_interval_s,
        )
        try:
            poller.start()
            reply = self._ask_operator(
                "Mapping session is open. Teleoperate the robot slowly around the "
                "workspace now; reply 'done' when the sweep is complete.",
                kind="reset",
            )
        except BaseException:
            poller.stop()
            self._stop_mapping_session(map_client)
            raise
        pushed = poller.stop()
        if poller.error is not None:
            self._stop_mapping_session(map_client)
            return ToolResult(
                data={
                    "error": (
                        "the mapping sweep stopped early, so its map was "
                        f"discarded: {poller.error}"
                    ),
                    "frames_pushed": pushed,
                }
            )
        if pushed == 0:
            self._stop_mapping_session(map_client)
            return ToolResult(
                data={
                    "error": "no frames reached the mapping service",
                    "frames_pushed": 0,
                }
            )
        if reply is None:
            self._stop_mapping_session(map_client)
            return ToolResult(
                data={
                    "error": (
                        "no operator reply arrived; the teleoperation sweep was not "
                        "confirmed, so the map was discarded"
                    ),
                    "frames_pushed": pushed,
                }
            )

        try:
            build = map_client.build()
        except Exception as exc:
            self._stop_mapping_session(map_client)
            return ToolResult(
                data={
                    "error": f"map build failed and the session was discarded: {exc}",
                    "frames_pushed": pushed,
                }
            )

        annotations: dict[str, Any] = {}
        annotation_error: str | None = None
        if client_labels:
            annotations, annotation_error = self._label_keyframes(
                map_client, build.map_path, build.metadata, client_labels
            )
        elapsed_s = round(time.perf_counter() - started, 2)
        data: dict[str, Any] = {
            "success": True,
            "map_name": name,
            "session_id": status.session_id,
            "frames_pushed": pushed,
            "operator_reply": reply,
            "map_path": build.map_path,
            "map_metadata": build.metadata,
            "labels": annotations,
            "elapsed_s": elapsed_s,
        }
        if annotation_error is not None:
            data["annotation_error"] = annotation_error
        return ToolResult(data=data)

    @tool(readonly=True, exclude=("state",))
    def detect_objects(
        self,
        *,
        labels: Annotated[list[str], Field(min_length=1, max_length=16)],
        step: Annotated[int, Field(json_schema_extra={"default": -1})] = -1,
        min_score: Annotated[
            float, Field(ge=0.0, le=1.0, json_schema_extra={"default": 0.3})
        ] = 0.3,
        state: Any = None,
    ) -> ToolResult:
        """Detect and segment named objects on one persisted camera frame.

        Runs Grounding DINO + SAM 2 on the saved RGB frame at ``step``
        (default: latest). Returns each detection's label, score, box, pixel
        count, and the artifact path of its rendered mask overlay.
        """
        return _detect_objects_impl(
            self._require_perception_client(),  # type: ignore[attr-defined]
            state,
            labels=labels,
            step=step,
            min_score=min_score,
        )

    @tool(readonly=True)
    def check_clearance(
        self,
        *,
        heading_deg: Annotated[float, Field(json_schema_extra={"default": 0.0})] = 0.0,
        length: Annotated[
            float, Field(gt=0.0, json_schema_extra={"default": DEFAULT_LENGTH})
        ] = DEFAULT_LENGTH,
        half_width: Annotated[
            float,
            Field(gt=0.0, json_schema_extra={"default": DEFAULT_HALF_WIDTH}),
        ] = DEFAULT_HALF_WIDTH,
        unit: Annotated[
            Literal["camera_heights", "map_units", "m"],
            Field(json_schema_extra={"default": DEFAULT_UNIT}),
        ] = DEFAULT_UNIT,
    ) -> ToolResult:
        """Check whether the floor just ahead is clear before driving forward.

        Localizes the current camera frame against the built map and measures a
        corridor of ``length`` by ``2 * half_width`` starting at the camera,
        following the view direction, and turned by ``heading_deg``
        counter-clockwise. Distances are in ``unit``: camera heights by default
        because an unaligned map has arbitrary scale, ``map_units``, or ``m``
        when the map records a metric anchor.

        The status is ``clear`` (every part of the corridor has mapped
        geometry and nothing blocks it), ``blocked`` (obstacles reach into the
        corridor; ``nearest`` gives the closest one's distance, side offset,
        height, and, when the map was grounded, its object label),
        ``unmapped`` (the map has no geometry partway along the corridor; do
        not treat this as clear), ``unlocalized`` (no pose yet), or
        ``unusable`` (the map lacks gravity, floor, or a usable unit choice).
        Stop and turn when the status is not ``clear``.
        """
        client = self._require_localize_client()
        prior_index = getattr(self, "_clearance_prior_index", None)
        prior_pose = getattr(self, "_clearance_prior_pose", None)
        rgb, _, _ = self._map_frame()
        try:
            result = client.check_path(
                rgb,
                prior_index=prior_index,
                prior_pose=prior_pose,
                exhaustive=prior_index is None and prior_pose is None,
                heading_deg=heading_deg,
                length=length,
                half_width=half_width,
                unit=unit,
            )
        except Exception as exc:
            return ToolResult(data={"error": f"clearance check failed: {exc}"})
        data = result.to_dict()
        data["success"] = result.ok
        data["used_prior"] = prior_index is not None or prior_pose is not None
        if result.status == "unlocalized":
            localization = data.get("localization") or {}
            reason = localization.get("reason") or "the frame did not match the map"
            self._clearance_prior_index = None
            self._clearance_prior_pose = None
            data["error"] = (
                "could not localize the current frame against the map "
                f"({reason}). Stop, rotate the camera toward mapped space, and "
                "call check_clearance again; do not drive forward on an "
                "unlocalized check."
            )
            return ToolResult(data=data)
        if result.ok and result.pose is not None:
            # Reuse this pose as the next localization prior: it narrows the
            # keyframe search and keeps consecutive checks on one estimate.
            self._clearance_prior_pose = result.pose
            self._clearance_prior_index = result.keyframe
        elif result.status == "unusable":
            data["error"] = result.reason or ("the map cannot answer clearance checks")
        return ToolResult(data=data)

    # ---- helpers -----------------------------------------------------------

    def _require_map_client(self) -> LingBotMapClient:
        client = getattr(self, "_lingbotmap_client", None)
        if client is None:
            raise RuntimeError("no LingBot-Map service is connected")
        return client

    def _require_perception_client(self) -> GdinoSam2Client:
        client = getattr(self, "_gdino_sam2_client", None)
        if client is None:
            raise RuntimeError("no Grounding DINO + SAM 2 service is connected")
        return client

    def _require_localize_client(self) -> LocalizeClient:
        client = getattr(self, "_localize_client", None)
        if client is None:
            raise RuntimeError("no map localization service is connected")
        return client

    def _stop_mapping_session(self, map_client: LingBotMapClient) -> None:
        try:
            map_client.stop_session()
        except Exception as exc:
            logger.warning("failed to stop mapping session: %s", exc)

    def _label_keyframes(
        self,
        map_client: LingBotMapClient,
        map_path: str,
        metadata: dict[str, Any],
        labels: list[str],
    ) -> tuple[dict[str, Any], str | None]:
        """Ground labels on archived keyframes and persist the annotations.

        Frames are read back from the built map, so labeling sees the same
        source images and poses the map recorded rather than frames buffered
        during teleoperation. Maps with depth lift every mask into the map
        frame and report 3D centroids; the frame-archive backend has no depth,
        so its detections stay 2D.

        Returns the per-label best observation and an error string when
        labeling or persistence failed. The map itself is already built at
        this point, so a labeling failure is reported without discarding it.
        """
        perception = self._require_perception_client()
        try:
            source = ClientMapSource(map_client, map_path, metadata)
            result = ground_map(source, perception, labels, allow_2d=True)
        except Exception as exc:
            return {}, f"labeling failed: {exc}"

        annotations: dict[str, Any] = {}
        for label in labels:
            rows = [
                observation
                for observation in result.observations
                if observation.label == label
            ]
            if not rows:
                continue
            best = max(rows, key=lambda observation: observation.score)
            record: dict[str, Any] = {
                "keyframe": best.keyframe,
                "score": round(best.score, 4),
                "box": [round(value, 1) for value in best.box],
                "pixel_count": best.mask_pixels,
                "base_pose": best.pushed_base_pose,
                "timestamp_s": best.timestamp_s,
                "lifted": best.lifted,
                "instances": sum(
                    1 for instance in result.objects if instance.label == label
                ),
            }
            if best.summary is not None:
                record.update(
                    {
                        "point_count": best.summary["point_count"],
                        "centroid": [
                            round(value, 4) for value in best.summary["centroid"]
                        ],
                        "extent": [round(value, 4) for value in best.summary["extent"]],
                        "object_id": best.object_id,
                    }
                )
            annotations[label] = record

        try:
            save_grounded_map_via_client(result, map_client)
        except Exception as exc:
            return annotations, f"could not persist annotations: {exc}"
        return annotations, None


class _FramePoller:
    """Push frames and poses to a mapping session from a background thread."""

    #: A push may still be in flight when the operator ends the sweep, so the
    #: join waits past the client's push timeout before giving up on the thread.
    STOP_JOIN_TIMEOUT_S = DEFAULT_PUSH_TIMEOUT_S + 15.0

    def __init__(
        self,
        *,
        source: "Toolkit | MapFrameSource",
        map_client: LingBotMapClient,
        interval_s: float,
    ) -> None:
        self._source = source
        self._map_client = map_client
        self._interval_s = max(0.1, float(interval_s))
        self._thread: threading.Thread | None = None
        self._stop_event = threading.Event()
        self._pushed = 0
        self._error: str | None = None
        self._lock = threading.Lock()

    @property
    def error(self) -> str | None:
        """Return why the poller gave up, or ``None`` if it ran normally."""
        with self._lock:
            return self._error

    def start(self) -> None:
        self._thread = threading.Thread(
            target=self._run, name="map-frame-poller", daemon=True
        )
        self._thread.start()

    def stop(self) -> int:
        """Stop the poller, join it, and return the number of frames pushed.

        Check :attr:`error` after this returns: frames pushed before a failure
        do not make the session complete.
        """
        self._stop_event.set()
        thread = self._thread
        self._thread = None
        if thread is not None:
            thread.join(timeout=self.STOP_JOIN_TIMEOUT_S)
            if thread.is_alive():
                with self._lock:
                    if self._error is None:
                        self._error = (
                            "the frame poller did not stop within "
                            f"{self.STOP_JOIN_TIMEOUT_S:.0f} s"
                        )
        error = self.error
        if error is not None:
            logger.warning("frame poller stopped early: %s", error)
        return self._pushed

    def _run(self) -> None:
        next_due = 0.0
        while not self._stop_event.is_set():
            now = time.monotonic()
            if now < next_due:
                self._stop_event.wait(min(next_due - now, self._interval_s))
                continue
            next_due = now + self._interval_s
            try:
                rgb, pose, timestamp = self._source._map_frame()
                status = self._map_client.push_frame(
                    rgb, pose=pose, timestamp_s=timestamp
                )
            except Exception as exc:
                with self._lock:
                    self._error = str(exc)
                return
            with self._lock:
                self._pushed = status.frames


def _detect_objects_impl(
    perception: GdinoSam2Client,
    state: Any,
    *,
    labels: list[str],
    step: int,
    min_score: float,
) -> ToolResult:
    """Shared body for the ``detect_objects`` tool; kept out of the mixin MRO."""
    prompts = [str(label).strip() for label in labels]
    if not prompts or any(not label for label in prompts):
        return ToolResult(data={"error": "labels must be non-empty strings"})
    try:
        record = state.get(step)
    except Exception as exc:
        return ToolResult(data={"error": f"state step not available: {exc}"})
    image_name = next(
        (name for name in sorted(record.artifacts) if name.endswith(".png")),
        None,
    )
    if image_name is None:
        return ToolResult(
            data={
                "error": "no persisted RGB frame at this step",
                "step": record.step_idx,
                "artifacts": sorted(record.artifacts),
            }
        )
    image_bytes = state.load_bytes(image_name, step=record.step_idx)
    try:
        result = perception.detect_segment(
            image_bytes, labels=prompts, min_score=min_score
        )
    except Exception as exc:
        return ToolResult(data={"error": f"detection failed: {exc}"})
    if not result.found:
        return ToolResult(
            data={
                "success": True,
                "found": False,
                "step": record.step_idx,
                "labels": prompts,
                "reason": result.reason,
            }
        )
    return ToolResult(
        data={
            "success": True,
            "found": True,
            "step": record.step_idx,
            "image": image_name,
            "image_shape": list(result.image_shape),
            "detections": [
                {
                    "label": detection.label,
                    "score": round(detection.score, 4),
                    "box": [round(value, 1) for value in detection.box],
                    "pixel_count": int(detection.mask.sum()),
                }
                for detection in result.detections
            ],
        }
    )
