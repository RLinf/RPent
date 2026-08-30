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

"""BEHAVIOR-only Dashboard launcher and manual-control adapter.

This module intentionally lives outside :mod:`rpent.dashboard`.  It reuses the
main Dashboard server/state contracts, adds BEHAVIOR-only static controls and
HTTP routes, and leaves the shared dashboard implementation untouched.

Leader integration point:
    ``python -m robots.behavior.dashboard`` is a single-task BEHAVIOR launcher:
    it parses the standard robot spec configuration, initializes the BEHAVIOR
    runtime, constructs the toolkit, and binds ``toolkit.primitives.env`` as the
    manual-control backend.  The explicit ``--ui-only`` mode keeps a fake/static
    UI path for frontend debugging.
"""

from __future__ import annotations

import argparse
import hashlib
import hmac
import inspect
import json
import os
import socket
import threading
import time
import uuid
from pathlib import Path
from typing import Any, Mapping, Protocol

from fastapi import Body
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

from rpent.dashboard.events import (
    DashboardEvent,
    RunStartedEvent,
    RuntimeStatusEvent,
    ToolResultEvent,
)
from rpent.dashboard.server import DashboardServer as CoreDashboardServer
from rpent.dashboard.state import DashboardState

BEHAVIOR_CAMERAS = ("head", "left_wrist", "right_wrist")
BEHAVIOR_TARGETS = ("chassis", "left_arm", "right_arm")
BEHAVIOR_ACTIONS = (
    "forward",
    "backward",
    "turn_left",
    "turn_right",
    "up",
    "down",
    "rotate_left",
    "rotate_right",
    "open",
    "close",
    "observe",
)
_CHASSIS_ACTIONS = {
    "forward",
    "backward",
    "turn_left",
    "turn_right",
    "up",
    "down",
    "observe",
}
_ARM_ACTIONS = {
    "up",
    "down",
    "rotate_left",
    "rotate_right",
    "open",
    "close",
    "observe",
}
_FRAME_PATH_KEYS = (
    "path",
    "rgb_path",
    "image_path",
    "image_cam_path",
    "overlay_path",
)
_RUNTIME_STATES = {"pending", "starting", "ready", "failed"}

BEHAVIOR_DASHBOARD_SPEC: dict[str, Any] = {
    "task": {
        "command": "/rpent-task",
        "usage": "/rpent-task <task_name> <public_seed>",
        "fields": (
            {
                "name": "task_name",
                "suggestions": ("turning_on_radio", "picking_up_trash"),
            },
            {"name": "public_seed", "kind": "integer", "minimum": 0},
        ),
        "display": "{task_name} / s{public_seed}",
        "output_slug": "{task_name}_s{public_seed}",
    },
    "runtime_components": (
        {"name": "env", "label": "ENV", "scope": "unique"},
        {"name": "vla", "label": "VLA", "scope": "shared"},
        {"name": "dino", "label": "DINO", "scope": "shared"},
        {"name": "memory", "label": "MEM", "scope": "unique"},
    ),
    "frame_channels": (
        {"name": "head", "label": "head"},
        {"name": "left_wrist", "label": "left wrist"},
        {"name": "right_wrist", "label": "right wrist"},
    ),
    "behavior_control": {
        "targets": BEHAVIOR_TARGETS,
        "actions": BEHAVIOR_ACTIONS,
        "cameras": BEHAVIOR_CAMERAS,
        "pipeline": ("prepare", "execute", "discard", "capture", "stop"),
        "official_success_source": (
            'backend raw info["done"]["success"] or info_done.success only'
        ),
    },
}


class BehaviorControlBackend(Protocol):
    """Environment-owned manual-control surface consumed by this adapter."""

    def dashboard_control_capabilities(self) -> Mapping[str, Any]:
        """Return simulator-validated manual-control capabilities."""
        ...

    def dashboard_prepare_manual_command(
        self,
        *,
        target: str,
        action: str,
        camera: str,
        predecessor_plan_id: str | None = None,
        permit_command_id: str,
        background: bool = False,
        planning_only_probe: bool = False,
    ) -> Mapping[str, Any]:
        """Prepare one command without executing simulator state changes."""
        ...

    def dashboard_execute_prepared_command(
        self,
        *,
        command_id: str,
        plan_id: str | None = None,
    ) -> Mapping[str, Any]:
        """Execute one previously prepared command."""
        ...

    def dashboard_discard_prepared_command(
        self,
        *,
        plan_id: str | None = None,
        command_id: str | None = None,
    ) -> Mapping[str, Any]:
        """Discard one prepared command."""
        ...

    def dashboard_capture_views(
        self,
        *,
        command_id: str | None = None,
        camera: str = "head",
    ) -> Mapping[str, Any]:
        """Capture one atomic head/left_wrist/right_wrist frame group."""
        ...


class ControlRequestError(RuntimeError):
    """Stable HTTP-facing control rejection."""

    def __init__(
        self,
        status_code: int,
        code: str,
        message: str,
        *,
        extra: Mapping[str, Any] | None = None,
    ) -> None:
        super().__init__(message)
        self.status_code = int(status_code)
        self.code = str(code)
        self.message = str(message)
        self.extra = dict(extra or {})

    def payload(self) -> dict[str, Any]:
        return {"code": self.code, "error": self.message, **self.extra}


class OfficialSuccessLatch:
    """Latch only backend-sourced raw official BEHAVIOR success evidence."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._latched = False
        self._binding: dict[str, Any] | None = None

    def observe(self, result: Any) -> tuple[bool, dict[str, Any] | None]:
        binding = _raw_success_binding(result)
        with self._lock:
            if binding is not None:
                self._latched = True
                if self._binding is None:
                    self._binding = dict(binding)
            return self._latched, (
                dict(self._binding) if self._binding is not None else None
            )

    def is_latched(self) -> bool:
        with self._lock:
            return self._latched

    def binding(self) -> dict[str, Any] | None:
        with self._lock:
            return dict(self._binding) if self._binding is not None else None


class BehaviorDashboardState(DashboardState):
    """Dashboard state with BEHAVIOR cameras, control, and success receipt."""

    environment = "behavior"

    def __init__(
        self,
        *,
        run_id: str,
        output_dir: str | Path,
        dashboard_spec: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(
            run_id=run_id,
            output_dir=output_dir,
            dashboard_spec=dashboard_spec or BEHAVIOR_DASHBOARD_SPEC,
        )
        self._control_controller: BehaviorControlController | None = None
        self._selected_camera = "head"
        self._control_snapshot: dict[str, Any] = _initial_control_snapshot()
        self._success_latch = OfficialSuccessLatch()
        self._manual_terminal_receipt: dict[str, Any] | None = None
        self._progress: dict[str, Any] = {
            "official_task_success": False,
            "terminal_receipt_complete": False,
            "workflow_complete": False,
            "publication_complete": False,
        }

    @property
    def success_latch(self) -> OfficialSuccessLatch:
        return self._success_latch

    def bind_controller(self, controller: "BehaviorControlController") -> None:
        snapshot = controller.snapshot()
        if not isinstance(snapshot, Mapping):
            raise TypeError("controller snapshot must be a mapping")
        with self._lock:
            if self._control_controller not in (None, controller):
                raise RuntimeError("a different BEHAVIOR controller is already bound")
            self._control_controller = controller
            self._control_snapshot = dict(_json_safe(snapshot))

    def unbind_controller(
        self,
        controller: "BehaviorControlController | None" = None,
    ) -> None:
        with self._lock:
            if controller is not None and self._control_controller is not controller:
                return
            previous = dict(self._control_snapshot)
            self._control_controller = None
            self._control_snapshot = {
                **_initial_control_snapshot(),
                "control_revision": int(previous.get("control_revision") or 0) + 1,
                "selected_camera": self._selected_camera,
                "last_terminal": previous.get("last_terminal"),
                "success_latched": self._success_latch.is_latched(),
                "success_binding": self._success_latch.binding(),
                "unavailable_reason": "controller_not_bound",
            }

    def control_controller(self) -> "BehaviorControlController | None":
        with self._lock:
            return self._control_controller

    def bind_runtime_backend(self, primitives_kwargs: Mapping[str, Any]) -> None:
        """Bind the task-owned env client supplied by the shared Dashboard runner."""

        backend = primitives_kwargs.get("env")
        if backend is None:
            return
        controller = self.control_controller()
        if controller is None:
            self.bind_controller(BehaviorControlController(state=self, backend=backend))
            return
        controller.bind_backend(backend)

    def unbind_runtime_backend(self) -> None:
        """Release the task-owned backend without changing shared components."""

        controller = self.control_controller()
        if controller is not None:
            controller.unbind_backend()
        self.unbind_controller(controller)

    def update_control_snapshot(
        self,
        snapshot: Mapping[str, Any],
        *,
        controller: "BehaviorControlController | None" = None,
    ) -> bool:
        safe = _json_safe(snapshot)
        if not isinstance(safe, dict):
            return False
        with self._lock:
            if controller is not None and self._control_controller is not controller:
                return False
            current_revision = self._control_snapshot.get("control_revision")
            incoming_revision = safe.get("control_revision")
            if (
                isinstance(current_revision, int)
                and isinstance(incoming_revision, int)
                and incoming_revision < current_revision
            ):
                return False
            safe["selected_camera"] = self._selected_camera
            safe["success_latched"] = self._success_latch.is_latched()
            safe["success_binding"] = self._success_latch.binding()
            self._control_snapshot = safe
            return True

    def control_admission_snapshot(self) -> dict[str, Any]:
        with self._lock:
            return {
                "state": self._visible_state_locked(),
                "official_task_success": self._success_latch.is_latched(),
            }

    def set_selected_camera(self, camera: str) -> None:
        camera = str(camera or "").strip()
        if camera not in BEHAVIOR_CAMERAS:
            raise ValueError("invalid BEHAVIOR camera")
        with self._lock:
            self._selected_camera = camera
            self._control_snapshot["selected_camera"] = camera

    def selected_camera(self) -> str:
        with self._lock:
            return self._selected_camera

    def set_component_status(
        self,
        component: str,
        status: str,
        error: BaseException | str | None = None,
    ) -> None:
        component = str(component or "").strip()
        status = str(status or "").strip()
        if component not in {"env", "vla", "dino", "memory"}:
            raise ValueError(f"unknown BEHAVIOR runtime component: {component!r}")
        if status not in _RUNTIME_STATES:
            raise ValueError(f"unknown runtime status: {status!r}")
        self.emit(RuntimeStatusEvent(component=component, status=status, error=error))

    def publish_frame(self, kind: str, image: bytes, *, env_step: Any = None) -> bool:
        kind = _physical_camera(kind)
        if kind not in BEHAVIOR_CAMERAS or not isinstance(image, bytes):
            return False
        try:
            frame_idx = int(env_step)
        except (TypeError, ValueError):
            frame_idx = None
        with self._lock:
            if frame_idx is not None and frame_idx < self._frame_idx:
                return False
            self._frames[kind] = bytes(image)
            if frame_idx is not None:
                self._frame_idx = frame_idx
            return True

    def publish_frame_group(
        self,
        frames: Mapping[str, Any],
        *,
        capture_group_id: str | int,
        simulator_step: int,
    ) -> bool:
        if (
            set(frames) != set(BEHAVIOR_CAMERAS)
            or not all(isinstance(frames[camera], bytes) for camera in BEHAVIOR_CAMERAS)
            or not isinstance(capture_group_id, (str, int))
            or isinstance(capture_group_id, bool)
            or capture_group_id == ""
            or not isinstance(simulator_step, int)
            or isinstance(simulator_step, bool)
            or simulator_step < 0
        ):
            return False
        with self._lock:
            if simulator_step < self._frame_idx:
                return False
            for camera in BEHAVIOR_CAMERAS:
                self._frames[camera] = bytes(frames[camera])
            self._frame_idx = simulator_step
            return True

    def emit(self, event: DashboardEvent) -> None:
        if isinstance(event, ToolResultEvent):
            self._apply_behavior_tool_result(event.name, event.result)
            return
        super().emit(event)

    def begin_manual_command(self, command: Mapping[str, Any]) -> None:
        command_id = str(command.get("command_id") or "")
        if not command_id:
            raise ValueError("manual command_id is required")
        target = str(command.get("target") or "")
        action = str(command.get("action") or "")
        with self._lock:
            step = len(self._timeline) + 1
            self._timeline.append(
                {
                    "step": step,
                    "source": "behavior_dashboard",
                    "action": action,
                    "target": target,
                    "command_id": command_id,
                    "lease_id": str(command.get("lease_id") or ""),
                    "sequence": command.get("sequence"),
                    "args": {
                        "target": target,
                        "action": action,
                        "camera": str(command.get("camera") or ""),
                    },
                    "result": {},
                    "elapsed_s": None,
                    "terminated": self._success_latch.is_latched(),
                    "truncated": False,
                    "has_action_video": False,
                    "status": "prepared",
                    "_started_at": time.monotonic(),
                }
            )

    def finish_manual_command(
        self,
        command: Mapping[str, Any],
        result: Mapping[str, Any],
    ) -> dict[str, Any]:
        if not isinstance(result, Mapping):
            raise TypeError("manual command result must be a mapping")
        command_id = str(command.get("command_id") or "")
        safe_result = _public_result(result)
        self._ingest_frames_from_result(result)
        success_latched, success_binding = self._success_latch.observe(result)
        terminal_receipt = {
            **dict(_json_safe(command)),
            "phase": "failed" if _result_failed(result) else "completed",
            "result": safe_result,
            "primitive_success": result.get("primitive_success"),
            "task_success": bool(success_latched),
            "stop_reason": (
                result.get("stop_reason")
                or ("official_task_success" if success_latched else None)
            ),
            "official_success_binding": success_binding,
        }
        with self._lock:
            item = next(
                (
                    candidate
                    for candidate in reversed(self._timeline)
                    if candidate.get("source") == "behavior_dashboard"
                    and candidate.get("command_id") == command_id
                ),
                None,
            )
            if item is None:
                step = len(self._timeline) + 1
                item = {
                    "step": step,
                    "source": "behavior_dashboard",
                    "action": str(command.get("action") or ""),
                    "args": {},
                    "_started_at": time.monotonic(),
                }
                self._timeline.append(item)
            item["result"] = safe_result
            item["elapsed_s"] = _elapsed_s(result, item.get("_started_at"))
            item["status"] = terminal_receipt["phase"]
            item["terminated"] = success_latched
            item["truncated"] = bool(result.get("truncated"))
            item["primitive_success"] = result.get("primitive_success")
            item["task_success"] = bool(success_latched)
            self._terminated = self._terminated or success_latched
            self._truncated = self._truncated or bool(result.get("truncated"))
            if success_latched:
                self._progress["official_task_success"] = True
                self._progress["terminal_receipt_complete"] = True
                self._manual_terminal_receipt = dict(terminal_receipt)
                self._control_snapshot.update(
                    {
                        "available": False,
                        "motion_available": False,
                        "observe_available": False,
                        "phase": terminal_receipt["phase"],
                        "command_id": command_id,
                        "lease_id": str(command.get("lease_id") or ""),
                        "last_terminal": dict(terminal_receipt),
                        "success_latched": True,
                        "success_binding": success_binding,
                        "unavailable_reason": "official_success_latched",
                    }
                )
        return terminal_receipt

    def seal_safe_stop_receipt(
        self,
        *,
        lease_id: str,
        reason: str,
        stop_mode: str,
        prepared: Mapping[str, Any] | None,
        backend_result: Mapping[str, Any],
    ) -> tuple[dict[str, Any], Path]:
        """Seal a non-motion Dashboard stop without inventing task success."""

        safe_result = _public_result(backend_result)
        success_latched, success_binding = self._success_latch.observe(backend_result)
        official_receipt = backend_result.get("official_success_receipt")
        if not isinstance(official_receipt, Mapping):
            official_receipt = None
        terminal_receipt = {
            "schema_version": 1,
            "kind": "behavior_dashboard_safe_stop_terminal_receipt",
            "source": "behavior_dashboard.control.stop",
            "run_id": self.run_id,
            "phase": "stopped",
            "status": "stopped",
            "lease_id": str(lease_id),
            "command_id": str((prepared or {}).get("command_id") or ""),
            "plan_id": str((prepared or {}).get("plan_id") or ""),
            "reason": str(reason),
            "stop_mode": str(stop_mode),
            "had_prepared_command": bool(prepared),
            "motion_command_issued": bool(
                backend_result.get("motion_command_issued", False)
            ),
            "primitive_success": bool(
                backend_result.get("primitive_success") is True
                and not _result_failed(backend_result)
            ),
            "task_success": bool(success_latched),
            "official_success_source": 'info["done"]["success"]',
            "official_success_binding": success_binding,
            "official_success_receipt": (
                dict(_json_safe(official_receipt))
                if official_receipt is not None
                else None
            ),
            "raw_success_observed": bool(success_latched),
            "total_env_steps": backend_result.get("total_env_steps"),
            "backend_result": safe_result,
        }
        receipt_path = self._write_safe_stop_receipt(terminal_receipt)
        with self._lock:
            self._manual_terminal_receipt = dict(terminal_receipt)
            self._progress["terminal_receipt_complete"] = True
            self._progress["official_task_success"] = bool(success_latched)
            self._control_snapshot.update(
                {
                    "available": False,
                    "motion_available": False,
                    "observe_available": False,
                    "phase": "stopped",
                    "command_id": terminal_receipt["command_id"],
                    "lease_id": str(lease_id),
                    "last_terminal": dict(terminal_receipt),
                    "success_latched": bool(success_latched),
                    "success_binding": success_binding,
                    "unavailable_reason": "safe_stop_sealed",
                }
            )
        return terminal_receipt, receipt_path

    def _write_safe_stop_receipt(self, receipt: Mapping[str, Any]) -> Path:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        primary = self.output_dir / "terminal_receipt.json"
        target = (
            self.output_dir / "dashboard_safe_stop_terminal_receipt.json"
            if primary.exists()
            else primary
        )
        temporary = target.with_name(f".{target.name}.{uuid.uuid4().hex}.tmp")
        try:
            with temporary.open("x", encoding="utf-8") as stream:
                json.dump(
                    dict(_json_safe(receipt)),
                    stream,
                    indent=2,
                    sort_keys=True,
                    ensure_ascii=False,
                )
                stream.write("\n")
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, target)
        finally:
            temporary.unlink(missing_ok=True)
        return target

    def publish_capture_result(
        self,
        result: Mapping[str, Any],
    ) -> bool:
        if not isinstance(result, Mapping):
            return False
        frames = result.get("_frames_bytes")
        if not isinstance(frames, Mapping):
            return False
        group_id = result.get("capture_group_id")
        simulator_step = result.get("simulator_step", result.get("env_step"))
        if isinstance(simulator_step, bool) or not isinstance(simulator_step, int):
            return False
        return self.publish_frame_group(
            frames,
            capture_group_id=group_id,
            simulator_step=simulator_step,
        )

    def ingest_child_event(self, event: Mapping[str, Any]) -> None:
        """Relay a child event without trusting child lifecycle claims."""

        if not isinstance(event, Mapping):
            return
        event_type = str(event.get("type") or "")
        if event_type in {
            "official_success",
            "workflow_complete",
            "publication_complete",
        }:
            return
        with self._lock:
            self._events.append(dict(_json_safe(event)))

    def snapshot(self) -> dict[str, Any]:
        value = super().snapshot()
        with self._lock:
            value["control"] = dict(self._control_snapshot)
            value["progress"] = dict(self._progress)
        return value

    def run_detail(self) -> dict[str, Any]:
        value = super().run_detail()
        with self._lock:
            value["control"] = dict(self._control_snapshot)
            value["progress"] = dict(self._progress)
        return value

    def _apply_behavior_tool_result(self, name: str, result: Any) -> None:
        if not isinstance(result, Mapping):
            return
        self._ingest_frames_from_result(result)
        safe_result = _public_result(result)
        log = result.get("log")
        command = log.get("command") if isinstance(log, Mapping) else None
        if not isinstance(command, Mapping):
            return
        action = str(command.get("action") or name)
        try:
            step = int(result.get("step", len(self._timeline) + 1))
        except (TypeError, ValueError):
            step = len(self._timeline) + 1
        with self._lock:
            self._timeline.append(
                {
                    "step": step,
                    "action": action,
                    "args": {
                        key: _json_safe(value)
                        for key, value in command.items()
                        if key != "action"
                    },
                    "result": safe_result,
                    "elapsed_s": _elapsed_s(result, None),
                    "terminated": bool(result.get("terminated")),
                    "truncated": bool(result.get("truncated")),
                    "has_action_video": False,
                    "status": "failed" if _result_failed(result) else "completed",
                }
            )
            self._terminated = self._terminated or bool(result.get("terminated"))
            self._truncated = self._truncated or bool(result.get("truncated"))

    def _ingest_frames_from_result(self, result: Mapping[str, Any]) -> None:
        frames = result.get("_frames_bytes")
        if isinstance(frames, Mapping):
            for camera, image in frames.items():
                self.publish_frame(str(camera), image, env_step=result.get("env_step"))

        frame_paths = result.get("frames")
        if isinstance(frame_paths, Mapping):
            for camera, path in frame_paths.items():
                image = _read_contained_image(self.output_dir, {"path": path})
                if isinstance(image, bytes):
                    self.publish_frame(
                        str(camera),
                        image,
                        env_step=result.get("env_step") or result.get("step"),
                    )

        direct = result.get("_image_bytes")
        if isinstance(direct, bytes):
            self.publish_frame(
                result.get("resolved_camera") or result.get("camera") or "head",
                direct,
                env_step=result.get("env_step"),
            )

        containers: list[Mapping[str, Any]] = []
        for key in ("views", "images"):
            value = result.get(key)
            if isinstance(value, Mapping):
                containers.append(value)
        review = result.get("visual_review")
        if isinstance(review, Mapping):
            for key in ("views", "images"):
                value = review.get(key)
                if isinstance(value, Mapping):
                    containers.append(value)
        for views in containers:
            for camera, view in views.items():
                if not isinstance(view, Mapping):
                    continue
                image = view.get("_image_bytes")
                if not isinstance(image, bytes):
                    image = _read_contained_image(self.output_dir, view)
                if isinstance(image, bytes):
                    self.publish_frame(
                        str(camera),
                        image,
                        env_step=result.get("env_step") or view.get("env_step"),
                    )


class BehaviorControlController:
    """BEHAVIOR prepare/execute/discard/capture/stop controller."""

    def __init__(
        self,
        *,
        state: BehaviorDashboardState,
        backend: BehaviorControlBackend | None = None,
    ) -> None:
        self._state = state
        self._backend = backend
        self._lock = threading.RLock()
        self._control_revision = 0
        self._selected_camera = "head"
        self._prepared: dict[str, Any] | None = None
        self._last_terminal: dict[str, Any] | None = None
        self._last_error: str | None = None
        self._stop_requested = False
        self._capabilities: dict[str, Any] = {
            "motion_available": False,
            "observe_available": False,
            "unavailable_reason": "backend_not_bound",
        }

    def bind_backend(self, backend: BehaviorControlBackend) -> None:
        with self._lock:
            self._backend = backend
            self._stop_requested = False
            self._refresh_capabilities_locked()
            self._touch_locked()
        self._publish_snapshot()

    def unbind_backend(self) -> None:
        """Detach the per-task backend and discard any prepared command first."""

        backend: BehaviorControlBackend | None
        prepared: dict[str, Any]
        with self._lock:
            backend = self._backend
            prepared = dict(self._prepared or {})
            self._prepared = None
        if prepared and backend is not None:
            discard = getattr(backend, "dashboard_discard_prepared_command", None)
            if callable(discard):
                try:
                    _call_backend(
                        discard,
                        command_id=str(prepared.get("command_id") or ""),
                        plan_id=str(prepared.get("plan_id") or ""),
                    )
                except Exception as exc:
                    with self._lock:
                        self._last_error = (
                            f"unbind_discard_failed: {type(exc).__name__}: {exc}"
                        )
        with self._lock:
            self._backend = None
            self._capabilities = {
                "motion_available": False,
                "observe_available": False,
                "unavailable_reason": "backend_not_bound",
            }
            self._touch_locked()
            snapshot = self._snapshot_locked(phase="offline")
        self._publish_snapshot(snapshot)

    def configure_capabilities(
        self,
        *,
        motion_available: bool,
        observe_available: bool,
        unavailable_reason: str = "",
    ) -> None:
        with self._lock:
            self._capabilities.update(
                {
                    "motion_available": bool(motion_available),
                    "observe_available": bool(observe_available),
                    "unavailable_reason": str(unavailable_reason or ""),
                }
            )
            self._touch_locked()
        self._publish_snapshot()

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            return self._snapshot_locked()

    def state(self) -> dict[str, Any]:
        with self._lock:
            self._refresh_capabilities_locked()
            self._touch_locked()
            snapshot = self._snapshot_locked()
        self._publish_snapshot(snapshot)
        return snapshot

    def select_camera(self, camera: str) -> dict[str, Any]:
        camera = _validate_camera(camera)
        self._state.set_selected_camera(camera)
        with self._lock:
            self._selected_camera = camera
            self._touch_locked()
            snapshot = self._snapshot_locked()
        self._publish_snapshot(snapshot)
        return snapshot

    def prepare(
        self,
        *,
        lease_id: str,
        sequence: int,
        target: str,
        action: str,
        camera: str,
    ) -> dict[str, Any]:
        lease_id = _validate_token(lease_id, "lease_id")
        sequence = _validate_sequence(sequence)
        target, action, camera = _validate_target_action_camera(target, action, camera)
        self._ensure_running()
        backend = self._require_backend()
        motion_needed = action != "observe"
        self._ensure_capability(motion=motion_needed, observe=not motion_needed)

        command_id = uuid.uuid4().hex
        try:
            prepared = _call_backend(
                backend.dashboard_prepare_manual_command,
                target=target,
                action=action,
                camera=camera,
                predecessor_plan_id=(
                    self._prepared.get("plan_id") if self._prepared else None
                ),
                permit_command_id=command_id,
                background=False,
                planning_only_probe=False,
            )
        except Exception as exc:
            raise ControlRequestError(
                409,
                "prepare_failed",
                f"{type(exc).__name__}: {exc}",
            ) from exc
        if not isinstance(prepared, Mapping):
            raise ControlRequestError(
                502, "invalid_prepare", "prepare returned non-object"
            )
        if prepared.get("status") == "failed":
            raise ControlRequestError(
                409,
                str(prepared.get("stop_reason") or "prepare_failed"),
                str(prepared.get("error") or "manual command prepare failed"),
                extra={"prepare_result": _public_result(prepared)},
            )
        plan_id = str(prepared.get("plan_id") or "").strip()
        if not plan_id:
            raise ControlRequestError(502, "missing_plan_id", "prepare omitted plan_id")

        command = {
            "command_id": command_id,
            "lease_id": lease_id,
            "sequence": sequence,
            "target": target,
            "action": action,
            "camera": camera,
            "plan_id": plan_id,
        }
        with self._lock:
            self._prepared = {
                **command,
                "prepare_result": _public_result(prepared),
                "accepted_at": time.monotonic(),
            }
            self._selected_camera = camera
            self._last_error = None
            self._touch_locked()
            snapshot = self._snapshot_locked()
        self._state.begin_manual_command(command)
        self._publish_snapshot(snapshot)
        return {
            **snapshot,
            "accepted": True,
            "command_id": command_id,
            "plan_id": plan_id,
            "prepare_result": _public_result(prepared),
        }

    def execute(
        self,
        *,
        lease_id: str,
        command_id: str | None = None,
        plan_id: str | None = None,
    ) -> dict[str, Any]:
        lease_id = _validate_token(lease_id, "lease_id")
        self._ensure_running()
        backend = self._require_backend()
        with self._lock:
            prepared = dict(self._prepared or {})
        if not prepared:
            raise ControlRequestError(409, "nothing_prepared", "no prepared command")
        if prepared.get("lease_id") != lease_id:
            raise ControlRequestError(409, "lease_mismatch", "prepared lease mismatch")
        if command_id is not None and str(prepared.get("command_id")) != str(
            command_id
        ):
            raise ControlRequestError(
                409,
                "command_mismatch",
                "prepared command_id mismatch",
            )
        if plan_id is not None and str(prepared.get("plan_id")) != str(plan_id):
            raise ControlRequestError(409, "plan_mismatch", "prepared plan_id mismatch")

        try:
            result = _call_backend(
                backend.dashboard_execute_prepared_command,
                plan_id=str(prepared["plan_id"]),
                command_id=str(prepared["command_id"]),
            )
        except Exception as exc:
            raise ControlRequestError(
                409,
                "execute_failed",
                f"{type(exc).__name__}: {exc}",
            ) from exc
        if not isinstance(result, Mapping):
            raise ControlRequestError(
                502,
                "invalid_execute",
                "execute returned non-object",
            )
        terminal = self._state.finish_manual_command(prepared, result)
        with self._lock:
            self._last_terminal = dict(terminal)
            self._prepared = None
            self._last_error = None
            self._touch_locked()
            snapshot = self._snapshot_locked()
        self._publish_snapshot(snapshot)
        return {
            **snapshot,
            "executed": True,
            "command_id": prepared["command_id"],
            "plan_id": prepared["plan_id"],
            "terminal_receipt": terminal,
        }

    def discard(
        self,
        *,
        lease_id: str,
        command_id: str | None = None,
        plan_id: str | None = None,
    ) -> dict[str, Any]:
        lease_id = _validate_token(lease_id, "lease_id")
        backend = self._require_backend()
        with self._lock:
            prepared = dict(self._prepared or {})
        if not prepared:
            raise ControlRequestError(409, "nothing_prepared", "no prepared command")
        if prepared.get("lease_id") != lease_id:
            raise ControlRequestError(409, "lease_mismatch", "prepared lease mismatch")
        if command_id is not None and str(prepared.get("command_id")) != str(
            command_id
        ):
            raise ControlRequestError(
                409,
                "command_mismatch",
                "prepared command_id mismatch",
            )
        if plan_id is not None and str(prepared.get("plan_id")) != str(plan_id):
            raise ControlRequestError(409, "plan_mismatch", "prepared plan_id mismatch")
        try:
            result = _call_backend(
                backend.dashboard_discard_prepared_command,
                command_id=str(prepared["command_id"]),
                plan_id=str(prepared["plan_id"]),
            )
        except Exception as exc:
            raise ControlRequestError(
                409,
                "discard_failed",
                f"{type(exc).__name__}: {exc}",
            ) from exc
        with self._lock:
            self._prepared = None
            self._last_error = None
            self._touch_locked()
            snapshot = self._snapshot_locked(phase="discarded")
        self._publish_snapshot(snapshot)
        return {
            **snapshot,
            "discarded": True,
            "command_id": prepared["command_id"],
            "plan_id": prepared["plan_id"],
            "discard_result": _public_result(result),
        }

    def capture(self, *, lease_id: str) -> dict[str, Any]:
        _validate_token(lease_id, "lease_id")
        self._ensure_running()
        backend = self._require_backend()
        command_id = f"capture_{uuid.uuid4().hex}"
        try:
            result = _call_backend(
                backend.dashboard_capture_views,
                command_id=command_id,
                camera=self._selected_camera,
            )
        except Exception as exc:
            raise ControlRequestError(
                409,
                "capture_failed",
                f"{type(exc).__name__}: {exc}",
            ) from exc
        if not isinstance(result, Mapping):
            raise ControlRequestError(
                502, "invalid_capture", "capture returned non-object"
            )
        if not self._state.publish_capture_result(result):
            raise ControlRequestError(
                502,
                "invalid_capture",
                "capture omitted atomic BEHAVIOR three-camera frames",
            )
        with self._lock:
            self._last_error = None
            self._touch_locked()
            snapshot = self._snapshot_locked(phase="captured")
        self._publish_snapshot(snapshot)
        return {
            **snapshot,
            "captured": True,
            "command_id": command_id,
            "capture_result": _public_result(result),
        }

    def stop(
        self,
        *,
        lease_id: str,
        reason: str = "client_stop",
        stop_mode: str = "safe_stop",
    ) -> dict[str, Any]:
        _validate_token(lease_id, "lease_id")
        reason = str(reason or "client_stop")
        stop_mode = str(stop_mode or "safe_stop")
        backend_result: Mapping[str, Any] | None = None
        with self._lock:
            prepared = dict(self._prepared or {})
            self._prepared = None
            self._stop_requested = True
            self._touch_locked()
        backend = self._backend
        handler = getattr(backend, "dashboard_safe_stop", None)
        if callable(handler):
            try:
                backend_result = _call_backend(
                    handler,
                    reason=reason,
                    stop_mode=stop_mode,
                )
            except Exception as exc:
                raise ControlRequestError(
                    409,
                    "safe_stop_failed",
                    f"{type(exc).__name__}: {exc}",
                ) from exc
        elif prepared and backend is not None:
            discard = getattr(backend, "dashboard_discard_prepared_command", None)
            if callable(discard):
                try:
                    backend_result = _call_backend(
                        discard,
                        command_id=str(prepared.get("command_id") or ""),
                        plan_id=str(prepared.get("plan_id") or ""),
                    )
                except Exception:
                    backend_result = None
        safe_backend_result = (
            dict(backend_result) if isinstance(backend_result, Mapping) else {}
        )
        terminal_receipt, receipt_path = self._state.seal_safe_stop_receipt(
            lease_id=lease_id,
            reason=reason,
            stop_mode=stop_mode,
            prepared=prepared,
            backend_result=safe_backend_result,
        )
        with self._lock:
            self._last_terminal = dict(terminal_receipt)
            self._last_error = None
            self._capabilities = {
                "motion_available": False,
                "observe_available": False,
                "unavailable_reason": "safe_stop_sealed",
            }
            self._touch_locked()
            snapshot = self._snapshot_locked(phase="stopped")
        self._publish_snapshot(snapshot)
        return {
            **snapshot,
            "stopped": True,
            "stop_mode": stop_mode,
            "reason": reason,
            "backend_result": _public_result(safe_backend_result),
            "terminal_receipt": terminal_receipt,
            "terminal_receipt_path": str(receipt_path),
        }

    def command(
        self,
        *,
        lease_id: str,
        sequence: int,
        target: str,
        action: str,
        camera: str,
    ) -> dict[str, Any]:
        if str(action or "").strip() == "observe":
            _validate_sequence(sequence)
            _validate_target_action_camera(target, action, camera)
            return self.capture(lease_id=lease_id)
        prepared = self.prepare(
            lease_id=lease_id,
            sequence=sequence,
            target=target,
            action=action,
            camera=camera,
        )
        if action == "observe":
            return self.capture(lease_id=lease_id)
        return self.execute(
            lease_id=lease_id,
            command_id=str(prepared["command_id"]),
            plan_id=str(prepared["plan_id"]),
        )

    def _require_backend(self) -> BehaviorControlBackend:
        with self._lock:
            backend = self._backend
        if backend is None:
            raise ControlRequestError(
                409,
                "backend_not_bound",
                "BEHAVIOR control backend is not bound",
            )
        return backend

    def _ensure_running(self) -> None:
        lifecycle = self._state.control_admission_snapshot()
        if lifecycle["official_task_success"]:
            raise ControlRequestError(410, "run_finished", "official success latched")
        if lifecycle["state"] != "running":
            raise ControlRequestError(410, "run_not_running", "run is not running")

    def _ensure_capability(self, *, motion: bool, observe: bool) -> None:
        with self._lock:
            self._refresh_capabilities_locked()
            motion_available = bool(self._capabilities.get("motion_available"))
            observe_available = bool(self._capabilities.get("observe_available"))
            reason = str(
                self._capabilities.get("unavailable_reason")
                or self._capabilities.get("motion_unavailable_reason")
                or self._capabilities.get("observe_unavailable_reason")
                or "manual control unavailable"
            )
        if motion and not motion_available:
            raise ControlRequestError(409, "motion_unavailable", reason)
        if observe and not observe_available:
            raise ControlRequestError(409, "observe_unavailable", reason)

    def _refresh_capabilities_locked(self) -> None:
        backend = self._backend
        if backend is None:
            self._capabilities = {
                "motion_available": False,
                "observe_available": False,
                "unavailable_reason": "backend_not_bound",
            }
            return
        callback = getattr(backend, "dashboard_control_capabilities", None)
        if not callable(callback):
            self._capabilities = {
                "motion_available": False,
                "observe_available": False,
                "unavailable_reason": "capabilities_unavailable",
            }
            return
        try:
            reported = callback()
        except Exception as exc:
            self._capabilities = {
                "motion_available": False,
                "observe_available": False,
                "unavailable_reason": f"{type(exc).__name__}: {exc}",
            }
            return
        if not isinstance(reported, Mapping):
            self._capabilities = {
                "motion_available": False,
                "observe_available": False,
                "unavailable_reason": "capabilities_not_mapping",
            }
            return
        self._capabilities = dict(_json_safe(reported))

    def _snapshot_locked(self, *, phase: str | None = None) -> dict[str, Any]:
        prepared = dict(self._prepared or {})
        capabilities = dict(self._capabilities)
        motion_available = bool(capabilities.get("motion_available"))
        observe_available = bool(capabilities.get("observe_available"))
        available = bool(motion_available or observe_available)
        current_phase = phase or ("prepared" if prepared else "idle")
        if self._stop_requested and not prepared and phase is None:
            current_phase = "stopped"
        return {
            "control_revision": self._control_revision,
            "available": available,
            "motion_available": motion_available,
            "observe_available": observe_available,
            "phase": current_phase,
            "selected_camera": self._selected_camera,
            "prepared": bool(prepared),
            "prepared_plan_id": prepared.get("plan_id"),
            "command_id": prepared.get("command_id"),
            "lease_id": prepared.get("lease_id"),
            "sequence": prepared.get("sequence"),
            "target": prepared.get("target"),
            "action": prepared.get("action"),
            "prepare_result": prepared.get("prepare_result"),
            "last_terminal": self._last_terminal,
            "last_error": self._last_error,
            "success_latched": self._state.success_latch.is_latched(),
            "success_binding": self._state.success_latch.binding(),
            "stop_requested": self._stop_requested,
            "unavailable_reason": str(
                capabilities.get("unavailable_reason")
                or capabilities.get("motion_unavailable_reason")
                or capabilities.get("observe_unavailable_reason")
                or ""
            ),
            "capabilities": capabilities,
        }

    def _touch_locked(self) -> None:
        self._control_revision += 1

    def _publish_snapshot(self, snapshot: Mapping[str, Any] | None = None) -> None:
        self._state.update_control_snapshot(
            snapshot or self.snapshot(),
            controller=self,
        )


class BehaviorDashboardServer(CoreDashboardServer):
    """Main Dashboard server with BEHAVIOR-only control routes."""

    def __init__(
        self,
        *,
        host: str = "127.0.0.1",
        port: int = 0,
        runs_dir: str = "",
        language: str = "en",
        dashboard_spec: dict[str, Any] | None = None,
        control_backend: BehaviorControlBackend | None = None,
    ) -> None:
        super().__init__(
            host=host,
            port=port,
            runs_dir=runs_dir,
            language=language,
            dashboard_spec=dashboard_spec or BEHAVIOR_DASHBOARD_SPEC,
        )
        self._behavior_control_backend = control_backend
        self._install_static_wrapper()
        self._install_control_routes()

    def register(self, state: DashboardState) -> None:
        super().register(state)
        if isinstance(state, BehaviorDashboardState):
            controller = state.control_controller()
            if controller is None:
                controller = BehaviorControlController(
                    state=state,
                    backend=self._behavior_control_backend,
                )
                state.bind_controller(controller)
            elif self._behavior_control_backend is not None:
                controller.bind_backend(self._behavior_control_backend)

    def bind_control_backend(self, backend: BehaviorControlBackend) -> None:
        """Bind a runtime-owned backend without importing BEHAVIOR runtime here."""

        self._behavior_control_backend = backend
        state = getattr(self, "_state", None)
        if isinstance(state, BehaviorDashboardState):
            controller = state.control_controller()
            if controller is None:
                controller = BehaviorControlController(state=state, backend=backend)
                state.bind_controller(controller)
            else:
                controller.bind_backend(backend)

    def unbind_control_backend(
        self,
        backend: BehaviorControlBackend | None = None,
    ) -> None:
        """Detach the current task backend before the env runtime is stopped."""

        if backend is None or backend is self._behavior_control_backend:
            self._behavior_control_backend = None
        state = getattr(self, "_state", None)
        if isinstance(state, BehaviorDashboardState):
            controller = state.control_controller()
            if controller is not None:
                controller.unbind_backend()

    def arm_auto_start(self, defaults: dict[str, Any]) -> None:
        """Attach-only launcher mode for an already-started parent run."""

        self._launch_defaults = dict(defaults)
        self._launch_config = dict(defaults)
        self._launch_enabled = False
        self._launch_event.set()

    def stop(self, timeout_s: float = 10.0) -> None:
        """Stop only this in-process uvicorn server, with a bounded probe."""

        server = self._server
        if server is None:
            return
        server.should_exit = True
        deadline = time.monotonic() + max(0.0, float(timeout_s))
        probe_host = "127.0.0.1" if self.host in {"0.0.0.0", "::"} else self.host
        while time.monotonic() < deadline:
            try:
                with socket.create_connection(
                    (probe_host, int(self.port)), timeout=0.1
                ):
                    pass
            except OSError:
                self._server = None
                return
            time.sleep(0.02)
        raise RuntimeError(f"dashboard server did not stop on {self.host}:{self.port}")

    def _install_static_wrapper(self) -> None:
        static_dir = Path(__file__).with_name("dashboard") / "static"
        self._app.mount(
            "/behavior-static",
            StaticFiles(directory=static_dir),
            name="behavior-dashboard-static",
        )
        self._index_html = _inject_behavior_controls(self._index_html)

    def _install_control_routes(self) -> None:
        def lookup_state(run_id: Any) -> BehaviorDashboardState:
            run = str(run_id or "").strip()
            if not run:
                raise ControlRequestError(422, "invalid_run", "run is required")
            state = self._resolve(run)
            if not isinstance(state, BehaviorDashboardState):
                raise ControlRequestError(
                    404,
                    "unknown_behavior_run",
                    "unknown BEHAVIOR run",
                )
            return state

        def controller_for_run(run_id: Any) -> BehaviorControlController:
            state = lookup_state(run_id)
            controller = state.control_controller()
            if controller is None:
                raise ControlRequestError(
                    409,
                    "controller_not_bound",
                    "BEHAVIOR control controller is not bound",
                )
            return controller

        @self._app.get("/api/run/control/state")
        def api_control_state(run: str) -> JSONResponse:
            try:
                return JSONResponse(controller_for_run(run).state())
            except ControlRequestError as exc:
                return _error_response(exc)

        @self._app.post("/api/run/control/camera")
        def api_control_camera(
            payload: dict[str, Any] = Body(default={}),
        ) -> JSONResponse:
            try:
                body = _validate_payload(payload, required={"run", "camera"})
                return JSONResponse(
                    controller_for_run(body["run"]).select_camera(body["camera"])
                )
            except ControlRequestError as exc:
                return _error_response(exc)
            except ValueError as exc:
                return _error_response(
                    ControlRequestError(422, "invalid_camera", str(exc))
                )

        @self._app.post("/api/run/control/prepare")
        def api_control_prepare(
            payload: dict[str, Any] = Body(default={}),
        ) -> JSONResponse:
            try:
                body = _validate_payload(
                    payload,
                    required={
                        "run",
                        "lease_id",
                        "sequence",
                        "target",
                        "action",
                        "camera",
                    },
                )
                response = controller_for_run(body["run"]).prepare(
                    lease_id=body["lease_id"],
                    sequence=body["sequence"],
                    target=body["target"],
                    action=body["action"],
                    camera=body["camera"],
                )
                return JSONResponse(response, status_code=202)
            except ControlRequestError as exc:
                return _error_response(exc)

        @self._app.post("/api/run/control/execute")
        def api_control_execute(
            payload: dict[str, Any] = Body(default={}),
        ) -> JSONResponse:
            try:
                body = _validate_payload(
                    payload,
                    required={"run", "lease_id"},
                    optional={"command_id", "plan_id"},
                )
                return JSONResponse(
                    controller_for_run(body["run"]).execute(
                        lease_id=body["lease_id"],
                        command_id=body.get("command_id"),
                        plan_id=body.get("plan_id"),
                    )
                )
            except ControlRequestError as exc:
                return _error_response(exc)

        @self._app.post("/api/run/control/discard")
        def api_control_discard(
            payload: dict[str, Any] = Body(default={}),
        ) -> JSONResponse:
            try:
                body = _validate_payload(
                    payload,
                    required={"run", "lease_id"},
                    optional={"command_id", "plan_id"},
                )
                return JSONResponse(
                    controller_for_run(body["run"]).discard(
                        lease_id=body["lease_id"],
                        command_id=body.get("command_id"),
                        plan_id=body.get("plan_id"),
                    )
                )
            except ControlRequestError as exc:
                return _error_response(exc)

        @self._app.post("/api/run/control/capture")
        def api_control_capture(
            payload: dict[str, Any] = Body(default={}),
        ) -> JSONResponse:
            try:
                body = _validate_payload(payload, required={"run", "lease_id"})
                return JSONResponse(
                    controller_for_run(body["run"]).capture(
                        lease_id=body["lease_id"],
                    )
                )
            except ControlRequestError as exc:
                return _error_response(exc)

        @self._app.post("/api/run/control/stop")
        def api_control_stop(
            payload: dict[str, Any] = Body(default={}),
        ) -> JSONResponse:
            try:
                body = _validate_payload(
                    payload,
                    required={"run", "lease_id"},
                    optional={"reason", "stop_mode"},
                )
                return JSONResponse(
                    controller_for_run(body["run"]).stop(
                        lease_id=body["lease_id"],
                        reason=str(body.get("reason") or "client_stop"),
                        stop_mode=str(body.get("stop_mode") or "safe_stop"),
                    )
                )
            except ControlRequestError as exc:
                return _error_response(exc)

        @self._app.post("/api/run/control/command")
        def api_control_command(
            payload: dict[str, Any] = Body(default={}),
        ) -> JSONResponse:
            try:
                body = _validate_payload(
                    payload,
                    required={
                        "run",
                        "lease_id",
                        "sequence",
                        "target",
                        "action",
                        "camera",
                    },
                )
                response = controller_for_run(body["run"]).command(
                    lease_id=body["lease_id"],
                    sequence=body["sequence"],
                    target=body["target"],
                    action=body["action"],
                    camera=body["camera"],
                )
                return JSONResponse(response, status_code=202)
            except ControlRequestError as exc:
                return _error_response(exc)


def create_server(
    *,
    host: str = "127.0.0.1",
    port: int = 0,
    output_dir: str | Path,
    run_id: str = "behavior-dashboard/manual",
    language: str = "en",
    control_backend: BehaviorControlBackend | None = None,
) -> tuple[BehaviorDashboardServer, BehaviorDashboardState]:
    """Create a BEHAVIOR Dashboard server/state pair without starting runtime."""

    server = BehaviorDashboardServer(
        host=host,
        port=port,
        language=language,
        control_backend=control_backend,
    )
    state = BehaviorDashboardState(
        run_id=run_id,
        output_dir=output_dir,
        dashboard_spec=BEHAVIOR_DASHBOARD_SPEC,
    )
    server.register(state)
    return server, state


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Start a BEHAVIOR-only RPent Dashboard launcher. By default this "
            "launches/connects the single-task env, VLA, DINO, and memory "
            "components through robots.behavior.robot_spec. Use --ui-only only "
            "for fake/static frontend debugging."
        )
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--language", choices=("en", "zh-cn"), default="en")
    parser.add_argument("--run-id", default=None)
    parser.add_argument(
        "--ui-only",
        action="store_true",
        help="Serve BEHAVIOR Dashboard UI/control routes without env/VLA/DINO/memory.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help=(
            "Run output directory. Runtime mode defaults to the BEHAVIOR "
            "runtime log path; --ui-only defaults to logs/behavior_dashboard_manual."
        ),
    )
    _add_behavior_runtime_args(parser)
    _add_optional_arg(
        parser,
        "--memory-dir",
        default=None,
        help="Explicit BEHAVIOR episode-memory directory.",
    )
    _add_optional_arg(
        parser,
        "--dino-source-archive",
        default=None,
        help=(
            "DINOv2 source archive path. Forwarded via "
            "RPENT_BEHAVIOR_DINOV2_SOURCE_ARCHIVE for the spawned DINO service."
        ),
    )
    _add_optional_arg(
        parser,
        "--dino-weights",
        default=None,
        help=(
            "DINOv2 weights path. Forwarded via RPENT_BEHAVIOR_DINOV2_WEIGHTS "
            "for the spawned DINO service."
        ),
    )
    _add_optional_arg(
        parser,
        "--dino-cache-dir",
        default=None,
        help=(
            "DINOv2 cache directory metadata for launcher integrations. The "
            "current runtime-side DINO spawner does not yet consume this flag."
        ),
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _build_arg_parser()
    args = parser.parse_args(argv)
    if bool(getattr(args, "ui_only", False)):
        return _run_ui_only_dashboard(args)
    return _run_runtime_bound_dashboard(args, parser)


def _run_ui_only_dashboard(args: argparse.Namespace) -> int:
    output_dir = (
        Path(args.output_dir)
        if args.output_dir is not None
        else Path.cwd() / "logs" / "behavior_dashboard_manual"
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    server, state = create_server(
        host=args.host,
        port=args.port,
        output_dir=output_dir,
        run_id=args.run_id or "behavior-dashboard/manual",
        language=args.language,
    )
    state.shared_services_ready()
    url = server.start()
    print(
        f"BEHAVIOR Dashboard: {url}. UI/control routes are serving in "
        "--ui-only mode; no simulator or model service was started.",
        flush=True,
    )
    try:
        threading.Event().wait()
    except KeyboardInterrupt:
        state.request_shutdown()
    finally:
        server.stop(timeout_s=5.0)
    return 0


def _run_runtime_bound_dashboard(
    args: argparse.Namespace,
    parser: argparse.ArgumentParser,
) -> int:
    _require_runtime_task_args(args, parser)
    _apply_memory_dir_alias(args, parser)
    _apply_dino_asset_env(args)

    from robots.behavior.robot_spec import get_robot_spec, get_toolkit
    from rpent.utils.logging import init_output_dir

    robot_spec = get_robot_spec()
    run_config = robot_spec.parse_config(args)
    output_dir = init_output_dir(run_config.output_dir, verbose=False)
    run_id = args.run_id or f"behavior-dashboard/{run_config.recipe_tag}"
    server, state = create_server(
        host=args.host,
        port=args.port,
        output_dir=output_dir,
        run_id=run_id,
        language=args.language,
    )
    url = server.start()
    print(
        f"BEHAVIOR Dashboard: {url}. Starting runtime-bound single task "
        f"{run_config.recipe_tag} with env/VLA/DINO/memory components.",
        flush=True,
    )

    daemons: list[Any] = []
    toolkit: Any = None
    backend: Any = None
    try:
        daemons, primitives_kwargs = robot_spec.init_runtime(
            args,
            output_dir,
            state,
            {"env", "vla", "dino", "memory"},
        )
        toolkit = get_toolkit(
            primitives_kwargs=primitives_kwargs,
            dashboard_events=state,
            config=run_config,
        )
        primitives = getattr(toolkit, "primitives", None)
        backend = getattr(primitives, "env", None)
        if backend is None:
            raise RuntimeError("BEHAVIOR toolkit did not expose primitives.env")
        server.bind_control_backend(backend)
        state.emit(RunStartedEvent())
        print(
            "BEHAVIOR runtime is bound; Dashboard controls now use "
            "toolkit.primitives.env as backend.",
            flush=True,
        )
        threading.Event().wait()
    except KeyboardInterrupt:
        state.request_shutdown()
    except Exception as exc:
        state.fail_session(exc)
        raise
    finally:
        _safe_stop_runtime_backend(backend)
        _close_toolkit(toolkit)
        _stop_daemons(daemons)
        server.stop(timeout_s=5.0)
    return 0


def _add_behavior_runtime_args(parser: argparse.ArgumentParser) -> None:
    from robots.behavior import runtime

    runtime.add_cli_args(parser, use_dashboard=True)


def _parser_has_option(parser: argparse.ArgumentParser, option: str) -> bool:
    return any(option in action.option_strings for action in parser._actions)


def _add_optional_arg(
    parser: argparse.ArgumentParser,
    option: str,
    **kwargs: Any,
) -> None:
    if not _parser_has_option(parser, option):
        parser.add_argument(option, **kwargs)


def _require_runtime_task_args(
    args: argparse.Namespace,
    parser: argparse.ArgumentParser,
) -> None:
    if not (getattr(args, "task_name", None) or getattr(args, "task", None)):
        parser.error("runtime-bound mode requires --task-name or --task")
    if (
        getattr(args, "public_seed", None) is None
        and getattr(args, "seed", None) is None
    ):
        parser.error("runtime-bound mode requires --public-seed or --seed")


def _apply_memory_dir_alias(
    args: argparse.Namespace,
    parser: argparse.ArgumentParser,
) -> None:
    memory_dir = getattr(args, "memory_dir", None)
    behavior_memory_dir = getattr(args, "behavior_memory_dir", None)
    if not memory_dir:
        return
    memory_dir_path = Path(memory_dir).expanduser().resolve()
    if behavior_memory_dir:
        behavior_memory_dir_path = Path(behavior_memory_dir).expanduser().resolve()
        if memory_dir_path != behavior_memory_dir_path:
            parser.error("--memory-dir and --behavior-memory-dir disagree")
    setattr(args, "memory_dir", str(memory_dir_path))
    setattr(args, "behavior_memory_dir", str(memory_dir_path))


def _apply_dino_asset_env(args: argparse.Namespace) -> None:
    for attr, env_name in (
        ("dino_source_archive", "RPENT_BEHAVIOR_DINOV2_SOURCE_ARCHIVE"),
        ("dino_weights", "RPENT_BEHAVIOR_DINOV2_WEIGHTS"),
        ("dino_cache_dir", "RPENT_BEHAVIOR_DINOV2_CACHE_DIR"),
    ):
        value = getattr(args, attr, None)
        if not value:
            continue
        resolved = Path(value).expanduser().resolve()
        setattr(args, attr, str(resolved))
        os.environ[env_name] = str(resolved)


def _safe_stop_runtime_backend(backend: Any) -> None:
    if backend is None:
        return
    handler = getattr(backend, "dashboard_safe_stop", None)
    if callable(handler):
        try:
            _call_backend(
                handler,
                reason="dashboard_launcher_exit",
                stop_mode="safe_stop",
            )
            return
        except Exception:
            pass
    finalize = getattr(backend, "finalize_paused_runtime", None)
    if callable(finalize):
        try:
            _call_backend(finalize, vla_status=None)
        except Exception:
            pass


def _close_toolkit(toolkit: Any) -> None:
    closer = getattr(toolkit, "close", None)
    if callable(closer):
        try:
            closer()
        except Exception:
            pass


def _stop_daemons(daemons: list[Any]) -> None:
    for daemon in reversed(list(daemons or [])):
        if hasattr(daemon, "stop"):
            try:
                daemon.stop()
            except Exception:
                pass


def _initial_control_snapshot() -> dict[str, Any]:
    return {
        "control_revision": 0,
        "available": False,
        "motion_available": False,
        "observe_available": False,
        "phase": "idle",
        "selected_camera": "head",
        "prepared": False,
        "prepared_plan_id": None,
        "command_id": None,
        "lease_id": None,
        "sequence": None,
        "target": None,
        "action": None,
        "prepare_result": None,
        "last_terminal": None,
        "last_error": None,
        "success_latched": False,
        "success_binding": None,
        "stop_requested": False,
        "unavailable_reason": "controller_not_bound",
        "capabilities": {},
    }


def _inject_behavior_controls(html: str) -> str:
    panel = """\
        <button class="controls-toggle collapsed-toggle" type="button" aria-expanded="false"
                aria-controls="interactiveControls" title="Show or hide manual robot controls.">
          <span>Interactive Controls</span><span class="chevron" aria-hidden="true">⌃</span>
        </button>
        <div class="frame-tabs behavior-frame-tabs" aria-label="Camera view">
          <button type="button" data-kind="head" data-camera="head" class="active"
                  aria-pressed="true">head</button>
          <button type="button" data-kind="left_wrist" data-camera="left_wrist"
                  aria-pressed="false">left wrist</button>
          <button type="button" data-kind="right_wrist" data-camera="right_wrist"
                  aria-pressed="false">right wrist</button>
        </div>
        <section class="control-rail control-left" id="interactiveControls"
                 aria-label="Interactive robot controls">
          <button class="controls-toggle" type="button" aria-expanded="true"
                  aria-controls="interactiveControls" title="Show or hide manual robot controls.">
            <span>Interactive Controls</span><span class="chevron" aria-hidden="true">⌃</span>
          </button>
          <div class="target-row">
            <button class="target-button selected active" type="button" data-target="chassis"
                    data-tooltip="Control the chassis."
                    aria-pressed="true">Chassis</button>
          </div>
          <div class="control-section">
            <div class="dpad-wrap">
              <span class="dpad-label label-forward">Forward</span>
              <span class="dpad-label label-left">Turn<br>left</span>
              <span class="dpad-label label-right">Turn<br>right</span>
              <span class="dpad-label label-backward">Backward</span>
              <div class="dpad" aria-label="Directional controls">
                <button class="control-button dpad-up" type="button" data-action="forward"
                        data-repeat="true" aria-label="Forward"
                        data-tooltip="Move the chassis forward by 5 cm. Hold to continue.">
                  <svg class="control-icon dpad-icon" viewBox="0 0 16 16" aria-hidden="true">
                    <path fill="currentColor" d="M8 3 13 11H3Z"/>
                  </svg>
                </button>
                <button class="control-button dpad-left" type="button" data-action="turn_left"
                        data-repeat="true" aria-label="Turn left"
                        data-tooltip="Rotate the chassis left by 5°. Hold to continue.">
                  <svg class="control-icon dpad-icon" viewBox="0 0 16 16" aria-hidden="true">
                    <path fill="currentColor" d="m3 8 8-5v10Z"/>
                  </svg>
                </button>
                <button class="control-button dpad-right" type="button" data-action="turn_right"
                        data-repeat="true" aria-label="Turn right"
                        data-tooltip="Rotate the chassis right by 5°. Hold to continue.">
                  <svg class="control-icon dpad-icon" viewBox="0 0 16 16" aria-hidden="true">
                    <path fill="currentColor" d="m13 8-8 5V3Z"/>
                  </svg>
                </button>
                <button class="control-button dpad-down" type="button" data-action="backward"
                        data-repeat="true" aria-label="Backward"
                        data-tooltip="Move the chassis backward by 5 cm. Hold to continue.">
                  <svg class="control-icon dpad-icon" viewBox="0 0 16 16" aria-hidden="true">
                    <path fill="currentColor" d="M8 13 3 5h10Z"/>
                  </svg>
                </button>
              </div>
            </div>
            <div class="observe-wrap">
              <button class="control-button round-button" type="button" data-action="observe"
                      data-repeat="false" aria-label="Observe"
                      data-tooltip="Refresh the currently selected camera view.">
                <svg class="control-icon observe-icon" viewBox="0 0 26 18" aria-hidden="true">
                  <path d="M1.8 9s4.1-6.1 11.2-6.1S24.2 9 24.2 9 20.1 15.1 13 15.1 1.8 9 1.8 9Z"
                        fill="none" stroke="currentColor" stroke-width="1.8" stroke-linejoin="round"/>
                  <circle cx="13" cy="9" r="3.2" fill="currentColor"/>
                </svg>
              </button>
              <span class="button-caption">Observe</span>
            </div>
          </div>
          <span id="behaviorManualControlState" class="control-status" aria-live="polite">offline</span>
        </section>
"""
    right_panel = """\
        <section class="control-rail control-right" aria-label="Arm and body controls">
          <div class="target-row">
            <button class="target-button" type="button" data-target="left_arm"
                    data-tooltip="Control the left arm."
                    aria-pressed="false">Left arm</button>
            <button class="target-button" type="button" data-target="right_arm"
                    data-tooltip="Control the right arm."
                    aria-pressed="false">Right arm</button>
          </div>
          <div class="function-grid">
            <div class="function-key">
              <button class="control-button round-button" type="button" data-action="up"
                      data-repeat="true" aria-label="Up"
                      data-tooltip="Raise the R1Pro torso by 3 cm. Hold to continue.">
                <svg class="control-icon" viewBox="0 0 24 24" aria-hidden="true">
                  <path d="M12 20V4M5.8 10.2 12 4l6.2 6.2" fill="none"
                        stroke="currentColor" stroke-width="2.3" stroke-linecap="round"
                        stroke-linejoin="round"/>
                </svg>
              </button>
              <span class="button-caption">Up</span>
            </div>
            <div class="function-key">
              <button class="control-button round-button" type="button" data-action="down"
                      data-repeat="true" aria-label="Down"
                      data-tooltip="Lower the R1Pro torso by 3 cm. Hold to continue.">
                <svg class="control-icon" viewBox="0 0 24 24" aria-hidden="true">
                  <path d="M12 4v16m6.2-6.2L12 20l-6.2-6.2" fill="none"
                        stroke="currentColor" stroke-width="2.3" stroke-linecap="round"
                        stroke-linejoin="round"/>
                </svg>
              </button>
              <span class="button-caption">Down</span>
            </div>
            <div class="function-key">
              <button class="control-button round-button" type="button" data-action="rotate_left"
                      data-repeat="true" aria-label="Rotate left"
                      data-tooltip="Available for arm control only.">
                <svg class="control-icon rotate-icon" viewBox="0 0 24 24" aria-hidden="true">
                  <path d="M15.8 5.5A7.7 7.7 0 1 1 10.1 3.9" fill="none"
                        stroke="currentColor" stroke-width="2.2" stroke-linecap="round"/>
                  <path d="M10.1 3.9h5.7v5.7" fill="none" stroke="currentColor"
                        stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round"/>
                </svg>
              </button>
              <span class="button-caption">Rotate left</span>
            </div>
            <div class="function-key">
              <button class="control-button round-button" type="button" data-action="rotate_right"
                      data-repeat="true" aria-label="Rotate right"
                      data-tooltip="Available for arm control only.">
                <svg class="control-icon rotate-icon" viewBox="0 0 24 24" aria-hidden="true">
                  <path d="M8.2 5.5a7.7 7.7 0 1 0 5.7-1.6" fill="none"
                        stroke="currentColor" stroke-width="2.2" stroke-linecap="round"/>
                  <path d="M13.9 3.9H8.2v5.7" fill="none" stroke="currentColor"
                        stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round"/>
                </svg>
              </button>
              <span class="button-caption">Rotate right</span>
            </div>
            <div class="function-key gripper">
              <button class="control-button round-button" type="button" data-action="open"
                      data-repeat="false" aria-label="Open"
                      data-tooltip="Available for arm control only.">
                <svg class="control-icon gripper-icon" viewBox="0 0 24 24" aria-hidden="true">
                  <path class="grip-accent" d="M9.2 3.1 5.5 5.5 2.7 11.5l2.8 6.6 3.7 2.6v-5l-1.4-.9-1.5-3.2 1.5-3.1 1.4-.9Z"/>
                  <path class="grip-dark" d="m14.8 3.1 3.7 2.4 2.8 6-2.8 6.6-3.7 2.6v-5l1.4-.9 1.5-3.2-1.5-3.1-1.4-.9Z"/>
                  <path class="grip-mid" d="M9.2 9.2h2.1v8.7H9.2Zm3.5 0h2.1v8.7h-2.1Z"/>
                </svg>
              </button>
              <span class="button-caption">Open</span>
            </div>
            <div class="function-key gripper">
              <button class="control-button round-button" type="button" data-action="close"
                      data-repeat="false" aria-label="Close"
                      data-tooltip="Available for arm control only.">
                <svg class="control-icon gripper-icon" viewBox="0 0 24 24" aria-hidden="true">
                  <path class="grip-dark" d="M9.2 3.1 5.5 5.5 2.7 11.5l2.8 6.6 3.7 2.6v-5l-1.4-.9-1.5-3.2 1.5-3.1 1.4-.9Z"/>
                  <path class="grip-accent" d="m14.8 3.1 3.7 2.4 2.8 6-2.8 6.6-3.7 2.6v-5l1.4-.9 1.5-3.2-1.5-3.1-1.4-.9Z"/>
                  <path class="grip-mid" d="M9.2 9.2h2.1v8.7H9.2Zm3.5 0h2.1v8.7h-2.1Z"/>
                </svg>
              </button>
              <span class="button-caption">Close</span>
            </div>
          </div>
          <span class="control-status" aria-hidden="true"></span>
        </section>
"""
    if (
        "/behavior-static/behavior_controls.js" in html
        or 'id="interactiveControls"' in html
    ):
        return html
    html = html.replace(
        "</head>",
        '<link rel="stylesheet" href="/behavior-static/behavior_controls.css" />\n</head>',
    )
    html = html.replace(
        '<div class="framewrap">',
        '<div class="framewrap behavior-mode" id="framewrap">\n'
        + panel
        + '        <div class="frame-stage">',
        1,
    )
    html = html.replace(
        '<div class="frame-tabs">',
        '<div class="frame-tabs legacy-frame-tabs">',
        1,
    )
    html = html.replace(
        '        <div class="frame-cap" id="frameCap" data-i18n="waitingFrame">waiting for first frame…</div>\n'
        "      </div>",
        "        </div>\n"
        '        <div class="frame-cap" id="frameCap" data-i18n="waitingFrame">waiting for first frame…</div>\n'
        + right_panel
        + "      </div>",
        1,
    )
    html = html.replace(
        '<div class="col right">',
        '<div class="col right behavior-dashboard">',
        1,
    )
    html = html.replace(
        "</body>",
        '<script type="module" src="/behavior-static/behavior_controls.js"></script>\n</body>',
    )
    return html


def _validate_payload(
    payload: Any,
    *,
    required: set[str],
    optional: set[str] | None = None,
) -> dict[str, Any]:
    if not isinstance(payload, dict):
        raise ControlRequestError(422, "invalid_payload", "request body must be object")
    allowed = required | (optional or set())
    extra = sorted(set(payload) - allowed)
    missing = sorted(required - set(payload))
    if extra:
        raise ControlRequestError(
            422,
            "unexpected_fields",
            f"unexpected fields: {', '.join(extra)}",
        )
    if missing:
        raise ControlRequestError(
            422,
            "missing_fields",
            f"missing fields: {', '.join(missing)}",
        )
    return payload


def _error_response(exc: ControlRequestError) -> JSONResponse:
    return JSONResponse(exc.payload(), status_code=exc.status_code)


def _validate_token(value: Any, name: str) -> str:
    token = str(value or "").strip()
    if not token or len(token) > 128:
        raise ControlRequestError(422, f"invalid_{name}", f"{name} is invalid")
    return token


def _validate_sequence(value: Any) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ControlRequestError(
            422,
            "invalid_sequence",
            "sequence must be a positive integer",
        )
    return int(value)


def _validate_camera(value: Any) -> str:
    camera = _physical_camera(value)
    if camera not in BEHAVIOR_CAMERAS:
        raise ControlRequestError(422, "invalid_camera", "invalid camera")
    return camera


def _validate_target_action_camera(
    target: Any,
    action: Any,
    camera: Any,
) -> tuple[str, str, str]:
    target = str(target or "").strip()
    action = str(action or "").strip()
    camera = _validate_camera(camera)
    if target not in BEHAVIOR_TARGETS:
        raise ControlRequestError(422, "invalid_target", "invalid control target")
    if action not in BEHAVIOR_ACTIONS:
        raise ControlRequestError(422, "invalid_action", "invalid control action")
    allowed = _CHASSIS_ACTIONS if target == "chassis" else _ARM_ACTIONS
    if action not in allowed:
        raise ControlRequestError(
            422,
            "invalid_target_action",
            f"{action} is not available for {target}",
        )
    return target, action, camera


def _call_backend(method: Any, **kwargs: Any) -> Any:
    signature = inspect.signature(method)
    if not any(
        parameter.kind == inspect.Parameter.VAR_KEYWORD
        for parameter in signature.parameters.values()
    ):
        kwargs = {
            key: value for key, value in kwargs.items() if key in signature.parameters
        }
    return method(**kwargs)


def _physical_camera(value: Any) -> str:
    camera = str(value or "").strip()
    aliases = {
        "main": "head",
        "agent": "head",
        "head": "head",
        "left": "left_wrist",
        "left_wrist": "left_wrist",
        "right": "right_wrist",
        "right_wrist": "right_wrist",
    }
    return aliases.get(camera, camera)


def _raw_success_binding(result: Any) -> dict[str, Any] | None:
    if not isinstance(result, Mapping):
        return None
    for info_key in ("info", "last_info"):
        info = result.get(info_key)
        done = info.get("done") if isinstance(info, Mapping) else None
        if isinstance(done, Mapping) and done.get("success") is True:
            return {
                "source": f'{info_key}["done"]["success"]',
                **_env_step_field(result),
            }
    info_done = result.get("info_done")
    if isinstance(info_done, Mapping) and info_done.get("success") is True:
        return {"source": 'info_done["success"]', **_env_step_field(result)}
    receipt = _validated_success_receipt(result.get("official_success_receipt"))
    if receipt is None:
        return None
    return {
        "source": str(receipt["source"]),
        "env_step": int(receipt["env_step"]),
        "receipt": receipt,
    }


def _validated_success_receipt(value: Any) -> dict[str, Any] | None:
    if not isinstance(value, Mapping):
        return None
    receipt = dict(value)
    raw_done = receipt.get("raw_done")
    if (
        receipt.get("source") != 'info["done"]["success"]'
        or not isinstance(raw_done, Mapping)
        or raw_done.get("success") is not True
        or not isinstance(receipt.get("env_step"), int)
        or isinstance(receipt.get("env_step"), bool)
        or receipt.get("env_step") < 0
    ):
        return None
    claimed = receipt.get("receipt_sha256")
    if claimed is None:
        return dict(_json_safe(receipt))
    if not isinstance(claimed, str) or len(claimed) != 64:
        return None
    unsigned = dict(receipt)
    unsigned.pop("receipt_sha256", None)
    canonical = json.dumps(
        unsigned,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("utf-8")
    expected = hashlib.sha256(canonical).hexdigest()
    if not hmac.compare_digest(claimed, expected):
        return None
    return dict(_json_safe(receipt))


def _env_step_field(result: Mapping[str, Any]) -> dict[str, int]:
    step = result.get("env_step", result.get("step"))
    if isinstance(step, int) and not isinstance(step, bool) and step >= 0:
        return {"env_step": int(step)}
    return {}


def _public_result(value: Any) -> dict[str, Any]:
    safe = _json_safe(value)
    return safe if isinstance(safe, dict) else {"result": safe}


def _json_safe(value: Any) -> Any:
    if isinstance(value, bytes):
        return f"<{len(value)} bytes>"
    if isinstance(value, Mapping):
        public: dict[str, Any] = {}
        for key, item in value.items():
            name = str(key)
            lowered = name.lower()
            if name.startswith("_") or lowered in _FRAME_PATH_KEYS:
                continue
            public[name] = _json_safe(item)
        return public
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return str(value)


def _read_contained_image(root: Path, view: Mapping[str, Any]) -> bytes | None:
    for key in _FRAME_PATH_KEYS:
        raw = view.get(key)
        if not raw:
            continue
        try:
            path = Path(raw)
            resolved = (
                path.resolve(strict=True)
                if path.is_absolute()
                else (root / path).resolve(strict=True)
            )
            resolved.relative_to(root.resolve(strict=False))
            if resolved.is_file():
                return resolved.read_bytes()
        except (OSError, TypeError, ValueError):
            continue
    return None


def _elapsed_s(result: Mapping[str, Any], started_at: Any) -> float | None:
    value = result.get("elapsed_s")
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return round(max(0.0, float(value)), 3)
    if isinstance(started_at, (int, float)):
        return round(max(0.0, time.monotonic() - float(started_at)), 3)
    return None


def _result_failed(result: Mapping[str, Any]) -> bool:
    return bool(
        result.get("primitive_success") is False
        or result.get("success") is False
        or result.get("error") not in (None, "", False)
        or result.get("truncated") is True
    )


__all__ = [
    "BEHAVIOR_DASHBOARD_SPEC",
    "BEHAVIOR_CAMERAS",
    "BehaviorControlBackend",
    "BehaviorControlController",
    "BehaviorDashboardServer",
    "BehaviorDashboardState",
    "ControlRequestError",
    "OfficialSuccessLatch",
    "create_server",
    "main",
]


if __name__ == "__main__":
    raise SystemExit(main())
