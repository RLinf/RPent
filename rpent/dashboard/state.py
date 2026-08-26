"""Thread-safe in-memory state for dashboard live runs."""

from __future__ import annotations

import re
import threading
import uuid
from dataclasses import dataclass, replace
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

import numpy as np

from rpent.dashboard.events import (
    DashboardEvent,
    RunStartedEvent,
    RuntimeStatusEvent,
    StepRecordEvent,
    ToolResultEvent,
    TranscriptEvent,
    UsageEvent,
)
from rpent.dashboard.interaction import (
    DashboardInteractionError,
    DashboardMessage,
    DashboardMessageConflictError,
    InteractionUnavailableError,
    PlannerActivity,
    UnknownDashboardMessageError,
)

if TYPE_CHECKING:
    from rpent.tools.state import EnvState, StepRecord

RUNTIME_STATUSES = {"pending", "starting", "ready", "failed"}
TERMINAL_RUN_STATES = {"succeeded", "failed", "cancelled"}
_PLANNER_ACTIVITIES = {"starting", "idle", "busy", "ended"}
InterruptRequestResult = Literal["accepted", "duplicate", "noop"]
InputMode = Literal["command_only", "conversation", "disabled"]
TaskRequest = dict[str, Any]
_INTEGER = re.compile(r"-?[0-9]+")
_UNSAFE_SLUG = re.compile(r"[^A-Za-z0-9_.-]+")


def _to_json_safe(value: Any) -> Any:
    """Normalize common non-JSON values (numpy / Path) used in Dashboard timeline data."""
    if isinstance(value, np.ndarray):
        return _to_json_safe(value.tolist())
    if isinstance(value, np.generic):
        return _to_json_safe(value.item())
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {key: _to_json_safe(val) for key, val in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_json_safe(item) for item in value]
    return value


def _parse_task(task_spec: dict[str, Any], text: str) -> TaskRequest | None:
    tokens = text.split()
    command = task_spec["command"]
    if tokens[0].startswith("/rpent-") and tokens[0] != command:
        raise ValueError(f"unknown Dashboard command: {tokens[0]}")
    if tokens[0] != command:
        return None

    fields = task_spec["fields"]
    if len(tokens) != len(fields) + 1:
        raise ValueError(f"expected {task_spec['usage']}")

    request: TaskRequest = {}
    for field, raw in zip(fields, tokens[1:], strict=True):
        name = field["name"]
        suggestions = field.get("suggestions", ())
        if suggestions and raw not in suggestions:
            raise ValueError(f"unsupported {name}: {raw}")
        if field.get("kind") != "integer":
            request[name] = raw
            continue
        if _INTEGER.fullmatch(raw) is None:
            raise ValueError(f"{name} must be an integer, got {raw!r}")
        value = int(raw)
        minimum = field.get("minimum")
        if minimum is not None and value < minimum:
            raise ValueError(f"{name} must be at least {minimum}, got {value}")
        request[name] = value
    return request


def _format_task(task_spec: dict[str, Any], key: str, request: TaskRequest) -> str:
    return task_spec[key].format(**request)


@dataclass(frozen=True, slots=True)
class ClaimedTask:
    """One last-write-wins task command claimed by the Session controller."""

    number: int
    request: TaskRequest
    output_dir: Path


class DashboardState:
    """Thread-safe projection for one sequential Dashboard Session."""

    @property
    def enabled(self) -> bool:
        return True

    def __init__(
        self,
        *,
        run_id: str,
        output_dir: str | Path,
        dashboard_spec: dict[str, Any],
    ) -> None:
        root = Path(output_dir)
        self.run_id = run_id
        self.output_dir = root
        self.video_path = root / "episode.mp4"
        self._session_root = root
        self._task_spec = dashboard_spec["task"]
        self._runtime_components = dashboard_spec["runtime_components"]
        self._frame_channels = dashboard_spec["frame_channels"]
        self._runtime_names = {
            component["name"] for component in self._runtime_components
        }
        self._frame_names = {channel["name"] for channel in self._frame_channels}

        self._lock = threading.Lock()
        self._condition = threading.Condition(self._lock)
        self._task_state: str | None = None
        self._terminated = False
        self._truncated = False
        self._error: str | None = None
        self._usage = {"in": 0, "out": 0, "tool_calls": 0}
        self._planner_usage_base = {"in": 0, "out": 0, "tool_calls": 0}
        self._runtime = {
            component["name"]: {"status": "pending", "error": None}
            for component in self._runtime_components
        }
        self._events: list[dict[str, Any]] = []
        self._timeline: list[dict[str, Any]] = []
        self._frames: dict[str, bytes] = {}
        self._frame_idx = -1
        self.env_state: EnvState | None = None
        self.frame_artifacts: dict[str, str] = {}
        self._state_step_offset = 0
        self._action_video_sources: dict[int, tuple[EnvState, int, str]] = {}
        self._accepting_input = False
        self._planner_activity: PlannerActivity = "starting"
        self._interrupt_requested = False
        self._interrupt_in_flight = False
        self._messages: list[DashboardMessage] = []
        self._messages_by_id: dict[str, DashboardMessage] = {}
        self._last_interaction_error: str | None = None
        self._interaction_version = 0
        self._session_state = "starting_shared_services"
        self._task_generation = 0
        self._pending_task: TaskRequest | None = None
        self._current_task: TaskRequest | None = None
        self._task_replacement_requested = False
        self._control_feedback: list[str] = []
        self._control_error: str | None = None
        self._shutdown_requested = False

    @property
    def session_state(self) -> str:
        with self._lock:
            return self._session_state

    @property
    def task_replacement_requested(self) -> bool:
        with self._lock:
            return self._task_replacement_requested

    def shared_services_ready(self) -> None:
        """Open the command channel after shared services have started."""
        with self._condition:
            if self._session_state == "fatal":
                return
            self._session_state = "ready"
            self._planner_activity = "ended"
            self._accepting_input = False
            self._control_error = None
            self._interaction_changed_locked()

    def fail_session(self, error: BaseException | str) -> None:
        """Make a shared-service failure fatal for this in-memory Session."""
        with self._condition:
            self._session_state = "fatal"
            self._error = str(error)
            self._control_error = str(error)
            self._pending_task = None
            self._task_replacement_requested = False
            self._seal_interaction_locked()
            self._interaction_changed_locked()

    def request_shutdown(self) -> None:
        """Wake the controller so process-level cleanup can proceed."""
        with self._condition:
            self._shutdown_requested = True
            self._interaction_changed_locked()

    def submit_input(self, text: str) -> DashboardMessage | TaskRequest:
        """Route a local task command or a normal conversation message."""
        if not isinstance(text, str) or not text.strip():
            return self._submit_message(text)
        try:
            request = _parse_task(self._task_spec, text)
        except ValueError as exc:
            with self._condition:
                self._control_error = str(exc)
                self._interaction_changed_locked()
            raise
        if request is not None:
            self.request_task(request)
            return request
        return self._submit_message(text)

    def request_task(self, request: TaskRequest) -> None:
        """Atomically record the latest desired TaskRun and close old input."""
        with self._condition:
            if self._session_state == "fatal":
                raise InteractionUnavailableError("Dashboard Session is fatal")
            if self._session_state == "starting_shared_services":
                raise InteractionUnavailableError(
                    "Dashboard Session is still starting shared services"
                )

            self._pending_task = dict(request)
            self._control_error = None
            self._control_feedback = [
                f"Task selected: {_format_task(self._task_spec, 'display', request)}"
            ]

            active = self._task_state in {"starting", "running"}
            if active:
                self._session_state = "switch_pending"
                self._task_replacement_requested = True
                self._accepting_input = False
                for message in self._messages:
                    if message.status == "pending":
                        message.status = "unsent"
                        message.error = None
            else:
                self._session_state = "task_starting"
            self._interaction_changed_locked()

    def wait_for_task(self, timeout: float | None = None) -> ClaimedTask | None:
        """Block until the controller can claim the latest pending task."""
        with self._condition:
            self._condition.wait_for(
                lambda: (
                    self._pending_task is not None
                    or self._shutdown_requested
                    or self._session_state == "fatal"
                ),
                timeout=timeout,
            )
            if self._shutdown_requested or self._session_state == "fatal":
                return None
            if self._pending_task is None:
                return None

            request = self._pending_task
            self._pending_task = None
            self._task_generation += 1
            number = self._task_generation
            output_dir = (
                self._session_root
                / "tasks"
                / f"{number:04d}_{self._task_output_slug(request)}"
            )
            self._begin_task_locked(request, number=number, output_dir=output_dir)
            return ClaimedTask(
                number=number,
                request=request,
                output_dir=output_dir,
            )

    def complete_task_replacement(self, error: str | None = None) -> None:
        """Seal the old planner at the current scheduling boundary."""
        with self._condition:
            if not self._task_replacement_requested:
                return
            if error is not None:
                self._last_interaction_error = str(error)
            self._seal_interaction_locked()
            self._interaction_changed_locked()

    def report_task_warning(self, warning: str) -> None:
        """Expose a non-fatal TaskRun warning without changing run success."""
        with self._condition:
            self._control_feedback.append(str(warning))
            self._interaction_changed_locked()

    def complete_task(
        self,
        *,
        state: Literal["succeeded", "failed", "cancelled"],
        error: BaseException | str | None = None,
    ) -> None:
        """Finish only the current TaskRun and reopen the command channel."""
        if state not in TERMINAL_RUN_STATES:
            raise ValueError(f"invalid terminal run state: {state!r}")
        with self._condition:
            self._task_state = state
            self._terminated = any(
                item.get("terminated") for item in self._timeline
            )
            self._truncated = any(
                item.get("truncated") for item in self._timeline
            )
            self._error = None if error is None else str(error)
            self._task_replacement_requested = False
            self._seal_interaction_locked()
            self._reset_unique_components_locked()
            self._session_state = (
                "task_starting" if self._pending_task is not None else "ready"
            )
            if error is not None:
                self._control_error = str(error)
            self._interaction_changed_locked()

    def _begin_task_locked(
        self,
        request: TaskRequest,
        *,
        number: int,
        output_dir: Path,
    ) -> None:
        self._current_task = request
        self.output_dir = output_dir
        self.video_path = output_dir / "episode.mp4"
        self._session_state = "task_starting"
        self._task_state = "starting"
        self._task_replacement_requested = False
        self._terminated = False
        self._truncated = False
        self._error = None
        self._usage = {"in": 0, "out": 0, "tool_calls": 0}
        self._planner_usage_base = {"in": 0, "out": 0, "tool_calls": 0}
        self._reset_unique_components_locked()
        self._events = []
        self._timeline = []
        self._frames = {}
        self._frame_idx = -1
        self.env_state = None
        self.frame_artifacts = {}
        self._state_step_offset = 0
        self._action_video_sources = {}
        self._accepting_input = False
        self._planner_activity = "starting"
        self._interrupt_requested = False
        self._interrupt_in_flight = False
        self._messages = []
        self._messages_by_id = {}
        self._last_interaction_error = None
        self._control_error = None
        self._control_feedback.append(f"TaskRun {number:04d} starting…")
        self._interaction_changed_locked()

    def _task_output_slug(self, request: TaskRequest) -> str:
        raw = _format_task(self._task_spec, "output_slug", request)
        slug = _UNSAFE_SLUG.sub("_", raw).strip("._")
        if not slug:
            raise ValueError("Dashboard task output slug must not be empty")
        return slug

    @property
    def planner_activity(self) -> PlannerActivity:
        """Return the current planner input activity."""
        with self._lock:
            return self._planner_activity

    @property
    def interaction_version(self) -> int:
        """Monotonic version for bridges waiting on interaction changes."""
        with self._lock:
            return self._interaction_version

    def set_planner_activity(
        self,
        activity: PlannerActivity,
        *,
        accepting_input: bool | None = None,
    ) -> None:
        """Update planner activity from the owning planner bridge.

        The bridge sets ``accepting_input=True`` only after the initial
        ``query()`` succeeds. Once ended, a planner session cannot be reopened.
        """
        if activity not in _PLANNER_ACTIVITIES:
            raise ValueError(f"unknown planner activity: {activity!r}")
        with self._condition:
            if self._planner_activity == "ended" and activity != "ended":
                raise InteractionUnavailableError("Dashboard Session has ended")
            if activity == "ended":
                self._seal_interaction_locked()
                self._interaction_changed_locked()
                return
            changed = self._planner_activity != activity
            self._planner_activity = activity
            if accepting_input is not None:
                requested_accepting = bool(accepting_input)
                if self._task_state in TERMINAL_RUN_STATES:
                    requested_accepting = False
                changed = changed or self._accepting_input != requested_accepting
                self._accepting_input = requested_accepting
            if changed:
                self._interaction_changed_locked()

    def begin_planner_session(self, *, video_path: Path | None = None) -> None:
        """Prepare the current TaskRun for a fresh planner and toolkit."""
        with self._condition:
            if self._task_state not in {"starting", "running"}:
                raise InteractionUnavailableError("Dashboard TaskRun is not active")
            if self._task_replacement_requested:
                raise InteractionUnavailableError("Dashboard TaskRun replacement is pending")
            self._planner_activity = "starting"
            self._accepting_input = False
            self._interrupt_requested = False
            self._interrupt_in_flight = False
            self._planner_usage_base = dict(self._usage)
            if video_path is not None:
                self.video_path = Path(video_path)
            self._interaction_changed_locked()

    def _submit_message(self, text: str) -> DashboardMessage:
        """Create one pending user message and notify the owning bridge."""
        if not isinstance(text, str) or not text.strip():
            raise ValueError("message text must not be blank")
        with self._condition:
            if (
                not self._accepting_input
                or self._planner_activity == "ended"
                or self._task_state in TERMINAL_RUN_STATES
            ):
                raise InteractionUnavailableError(
                    "Dashboard Session is not accepting input"
                )
            message = DashboardMessage(
                message_id=f"msg_{uuid.uuid4().hex}",
                text=text,
                status="pending",
            )
            self._messages.append(message)
            self._messages_by_id[message.message_id] = message
            self._interaction_changed_locked()
            return replace(message)

    def withdraw_message(self, message_id: str) -> DashboardMessage:
        """Atomically withdraw a message that is still pending."""
        with self._condition:
            message = self._message_locked(message_id)
            if message.status != "pending":
                raise DashboardMessageConflictError(
                    f"message is {message.status}, not pending"
                )
            message.status = "withdrawn"
            message.error = None
            self._interaction_changed_locked()
            return replace(message)

    def claim_next_pending_message(self) -> DashboardMessage | None:
        """Claim one message so task replacement can stop later sends."""
        with self._condition:
            if (
                self._planner_activity == "ended"
                or self._task_replacement_requested
                or self._interrupt_requested
            ):
                return None
            message = next(
                (item for item in self._messages if item.status == "pending"),
                None,
            )
            if message is None:
                return None
            message.status = "sending"
            message.error = None
            self._interaction_changed_locked()
            return replace(message)

    def mark_message_sent(self, message_id: str) -> DashboardMessage:
        """Commit one successfully queried message."""
        with self._condition:
            message = self._transition_sending_message_locked(
                message_id,
                status="sent",
                error=None,
            )
            self._interaction_changed_locked()
            return replace(message)

    def mark_message_failed(
        self,
        message_id: str,
        error: str,
    ) -> DashboardMessage:
        """Record one failed query without retrying it."""
        error_text = str(error)
        with self._condition:
            message = self._transition_sending_message_locked(
                message_id,
                status="failed",
                error=error_text,
            )
            self._interaction_changed_locked()
            return replace(message)

    def mark_message_unsent(self, message_id: str) -> DashboardMessage:
        """Restore one queued backend submission that never started."""
        with self._condition:
            message = self._transition_sending_message_locked(
                message_id,
                status="unsent",
                error=None,
            )
            self._interaction_changed_locked()
            return replace(message)

    def request_interrupt(self) -> InterruptRequestResult:
        """Record an Esc request without waiting for the planner backend."""
        with self._condition:
            if self._interrupt_requested:
                return "duplicate"
            if (
                self._planner_activity != "busy"
                or self._task_state in TERMINAL_RUN_STATES
            ):
                return "noop"
            self._interrupt_requested = True
            self._interrupt_in_flight = False
            self._interaction_changed_locked()
            return "accepted"

    def claim_interrupt_request(self) -> bool:
        """Claim a queued interrupt while keeping it visibly requested."""
        with self._condition:
            if not self._interrupt_requested or self._interrupt_in_flight:
                return False
            self._interrupt_in_flight = True
            self._interaction_changed_locked()
            return True

    def complete_interrupt(self, error: str | None = None) -> None:
        """Complete the claimed SDK interrupt, successfully or with an error."""
        with self._condition:
            if not self._interrupt_requested or not self._interrupt_in_flight:
                raise DashboardInteractionError("no interrupt request is in flight")
            self._interrupt_requested = False
            self._interrupt_in_flight = False
            if error is not None:
                self._last_interaction_error = str(error)
            else:
                self._last_interaction_error = None
            self._interaction_changed_locked()

    def seal_interaction(self) -> None:
        """End input and preserve every unfinished message as ``unsent``."""
        with self._condition:
            self._seal_interaction_locked()
            self._interaction_changed_locked()

    def wait_for_interaction_change(
        self,
        since: int,
        timeout: float | None = None,
    ) -> int:
        """Wait for any interaction state change and return its latest version."""
        with self._condition:
            self._condition.wait_for(
                lambda: self._interaction_version != since,
                timeout=timeout,
            )
            return self._interaction_version

    def emit(self, event: DashboardEvent) -> None:
        """Project one structured event into the existing frontend state."""
        if isinstance(event, TranscriptEvent):
            with self._lock:
                self._events.append(event.payload)
            return
        if isinstance(event, UsageEvent):
            with self._lock:
                self._usage = {
                    "in": self._planner_usage_base["in"] + int(event.inp),
                    "out": self._planner_usage_base["out"] + int(event.out),
                    "tool_calls": (
                        self._planner_usage_base["tool_calls"]
                        + int(event.tool_calls)
                    ),
                }
            return
        if isinstance(event, RuntimeStatusEvent):
            self._apply_runtime_status(event)
            return
        if isinstance(event, ToolResultEvent):
            self._apply_tool_result(event)
            return
        if isinstance(event, StepRecordEvent):
            with self._lock:
                if event.env_state is not self.env_state:
                    self._state_step_offset = max(0, self._frame_idx + 1)
                    self.env_state = event.env_state
                    self.frame_artifacts = dict(event.frame_artifacts)
                step_offset = self._state_step_offset
            self.on_step(event.record, step_offset=step_offset)
            return
        if isinstance(event, RunStartedEvent):
            self._start()
            return
        raise TypeError(f"unsupported dashboard event: {type(event).__name__}")

    def _apply_runtime_status(self, event: RuntimeStatusEvent) -> None:
        if event.component not in self._runtime_names:
            raise ValueError(f"unknown runtime component: {event.component!r}")
        if event.status not in RUNTIME_STATUSES:
            raise ValueError(f"unknown runtime status: {event.status!r}")
        with self._lock:
            self._runtime[event.component] = {
                "status": event.status,
                "error": None if event.error is None else str(event.error),
            }

    def _runtime_snapshot(self) -> dict[str, dict[str, str | None]]:
        """Return a detached copy of runtime status for a locked caller."""
        return {component: dict(status) for component, status in self._runtime.items()}

    def _reset_unique_components_locked(self) -> None:
        for component in self._runtime_components:
            if component["scope"] != "unique":
                continue
            self._runtime[component["name"]] = {
                "status": "pending",
                "error": None,
            }

    def _apply_tool_result(self, event: ToolResultEvent) -> None:
        name = event.name
        result = event.result
        if not isinstance(result, dict):
            return
        frames = {
            "camera": result.get("_image_cam_bytes") or result.get("_image_bytes"),
            "wrist": result.get("_image_wrist_bytes"),
        }
        self._update_frames(
            step=result.get("step"),
            frames={kind: data for kind, data in frames.items() if data},
        )
        log = result.get("log")
        if not isinstance(log, dict):
            return
        command = log.get("command")
        if not isinstance(command, dict) or command.get("action") != name:
            return
        try:
            step = int(result["step"])
        except Exception:
            return
        action = str(command.get("action", name))
        terminated = bool(result.get("terminated"))
        truncated = bool(result.get("truncated"))
        action_video_path = self._action_video_from_result(
            result,
            step=step,
            action=action,
        )
        action_video = str(action_video_path) if action_video_path is not None else None
        action_video_artifact = result.get("action_video_artifact")
        item = {
            "step": step,
            "action": action,
            "args": {k: v for k, v in command.items() if k != "action"},
            "result": _to_json_safe(log.get("result")),
            "elapsed_s": log.get("elapsed_s"),
            "terminated": terminated,
            "truncated": truncated,
            "action_video_path": action_video,
            "action_video_artifact": action_video_artifact,
            "has_action_video": bool(action_video_artifact or action_video_path),
        }
        with self._lock:
            self._timeline.append(item)
            self._terminated = self._terminated or terminated
            self._truncated = self._truncated or truncated

    def _action_video_from_result(
        self,
        result: dict[str, Any],
        *,
        step: int,
        action: str,
    ) -> Path | None:
        raw_path = result.get("action_video_path")
        if not raw_path:
            raw_path = f"action_videos/step_{step:02d}_{action}.mp4"
        try:
            path = self._resolve_output_path(raw_path)
        except TypeError:
            return None
        return path if path.exists() else None

    def on_step(self, record: StepRecord, *, step_offset: int = 0) -> None:
        """Project one recorded robot step into frames and timeline."""
        display_step = step_offset + record.step_idx
        self._update_step_frames(record, display_step=display_step)
        command = record.command
        if not isinstance(command, dict) or not command.get("action"):
            return
        action_video = next(
            (name for name in sorted(record.artifacts) if name.endswith(".mp4")),
            None,
        )
        item = {
            "step": display_step,
            "action": str(command.get("action")),
            "args": {key: value for key, value in command.items() if key != "action"},
            "result": _to_json_safe(record.result),
            "elapsed_s": record.elapsed_s,
            "terminated": record.terminated,
            "truncated": record.truncated,
            "action_video_artifact": action_video,
            "has_action_video": action_video is not None,
        }
        with self._lock:
            self._timeline.append(item)
            if action_video is not None and self.env_state is not None:
                self._action_video_sources[display_step] = (
                    self.env_state,
                    record.step_idx,
                    action_video,
                )
            self._terminated = self._terminated or record.terminated
            self._truncated = self._truncated or record.truncated

    def _update_step_frames(self, record: StepRecord, *, display_step: int) -> None:
        """Load dashboard frame bytes from the step's canonical artifacts."""
        env_state = self.env_state
        if env_state is None:
            return
        frames: dict[str, bytes] = {}
        for kind, artifact in self.frame_artifacts.items():
            if kind not in self._frame_names or artifact not in record.artifacts:
                continue
            try:
                frames[kind] = env_state.load_bytes(artifact, step=record.step_idx)
            except FileNotFoundError:
                continue
        self._update_frames(step=display_step, frames=frames)

    def _resolve_output_path(self, value: Any) -> Path:
        path = Path(value)
        if path.is_absolute() or path.is_relative_to(self.output_dir):
            return path
        return self.output_dir / path

    def _apply_frame_paths(self, result: dict[str, Any]) -> None:
        projected = result.get("frames")
        frame_paths = dict(projected) if isinstance(projected, dict) else {}
        for channel in self._frame_channels:
            name = channel["name"]
            path_key = channel.get("legacy_path_key")
            if name in frame_paths or path_key is None:
                continue
            if path_key in result:
                frame_paths[name] = result[path_key]
        if not frame_paths:
            return
        frames: dict[str, bytes] = {}
        for kind, path in frame_paths.items():
            if kind not in self._frame_names:
                continue
            if not path:
                continue
            try:
                frames[kind] = self._resolve_output_path(path).read_bytes()
            except (OSError, TypeError):
                continue
        self._update_frames(step=result.get("step"), frames=frames)

    def _update_frames(
        self,
        *,
        step: Any,
        frames: dict[str, bytes],
    ) -> None:
        try:
            frame_idx = int(step)
        except (TypeError, ValueError):
            frame_idx = None
        with self._lock:
            if frame_idx is not None and frame_idx < self._frame_idx:
                return
            self._frames = dict(frames)
            if frame_idx is not None:
                self._frame_idx = frame_idx

    def _start(self) -> None:
        with self._condition:
            self._task_state = "running"
            if not self._task_replacement_requested:
                self._session_state = "running"
            self._interaction_changed_locked()

    def _transition_sending_message_locked(
        self,
        message_id: str,
        *,
        status: Literal["sent", "failed", "unsent"],
        error: str | None,
    ) -> DashboardMessage:
        message = self._message_locked(message_id)
        if message.status != "sending":
            raise DashboardMessageConflictError(
                f"message is {message.status}, not sending"
            )
        message.status = status
        message.error = error
        return message

    def _message_locked(self, message_id: str) -> DashboardMessage:
        try:
            return self._messages_by_id[message_id]
        except KeyError as exc:
            raise UnknownDashboardMessageError(
                f"unknown Dashboard message: {message_id}"
            ) from exc

    def _seal_interaction_locked(self) -> None:
        self._planner_activity = "ended"
        self._accepting_input = False
        self._interrupt_requested = False
        self._interrupt_in_flight = False
        for message in self._messages:
            if message.status not in {"pending", "sending"}:
                continue
            message.status = "unsent"
            message.error = None

    def _interaction_changed_locked(self) -> None:
        self._interaction_version += 1
        self._condition.notify_all()

    def _interaction_snapshot_locked(self) -> dict[str, Any]:
        return {
            "session_id": self.run_id,
            "input_mode": self._input_mode_locked(),
            "planner_activity": self._planner_activity,
            "interrupt_requested": self._interrupt_requested,
            "messages": [message.as_dict() for message in self._messages],
            "last_error": self._last_interaction_error,
        }

    def _input_mode_locked(self) -> InputMode:
        if self._session_state in {"starting_shared_services", "fatal"}:
            return "disabled"
        if (
            self._session_state == "running"
            and self._accepting_input
            and not self._task_replacement_requested
        ):
            return "conversation"
        return "command_only"

    def _command_snapshot(self, request: TaskRequest | None) -> dict[str, Any] | None:
        if request is None:
            return None
        return {
            **request,
            "parameters": dict(request),
            "label": _format_task(self._task_spec, "display", request),
        }

    def _session_fields_locked(self) -> dict[str, Any]:
        return {
            "session_state": self._session_state,
            "task_generation": self._task_generation,
            "current_task": self._command_snapshot(self._current_task),
            "pending_task": self._command_snapshot(self._pending_task),
            "control_feedback": list(self._control_feedback),
            "control_error": self._control_error,
        }

    def _visible_state_locked(self) -> str:
        if self._session_state == "fatal":
            return "failed"
        if self._session_state in {"starting_shared_services", "task_starting"}:
            return "starting"
        return self._task_state or self._session_state

    def events_since(self, since: int) -> list[dict[str, Any]]:
        with self._lock:
            return list(self._events[since:])

    def frame(self, kind: str) -> bytes | None:
        if kind not in self._frame_names:
            raise ValueError(f"unknown frame kind: {kind!r}")
        with self._lock:
            return self._frames.get(kind)

    def action_video_path(self, step: int) -> Path | None:
        env_state = self.env_state
        with self._lock:
            source = self._action_video_sources.get(int(step))
            artifact = None
            raw_path = None
            for item in self._timeline:
                if int(item.get("step", -1)) != int(step):
                    continue
                artifact = item.get("action_video_artifact")
                raw_path = item.get("action_video_path")
                break
        if source is not None:
            source_state, source_step, source_artifact = source
            try:
                path = source_state.artifact_path(source_artifact, step=source_step)
            except (LookupError, ValueError):
                return None
            return path if path.exists() else None
        if artifact and env_state is not None:
            try:
                path = env_state.artifact_path(artifact, step=int(step))
            except (LookupError, ValueError):
                return None
            return path if path.exists() else None
        if raw_path:
            video_path = Path(raw_path)
            return video_path if video_path.exists() else None
        return None

    def has_video(self) -> bool:
        with self._lock:
            return self._task_state in TERMINAL_RUN_STATES and self.video_path.exists()

    def _frame_snapshot(self) -> tuple[int, dict[str, bool]]:
        available = {
            channel["name"]: channel["name"] in self._frames
            for channel in self._frame_channels
        }
        return self._frame_idx, available

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            frame_idx, frame_available = self._frame_snapshot()
            return {
                "state": self._visible_state_locked(),
                "terminated": self._terminated,
                "truncated": self._truncated,
                "error": self._error,
                "usage": dict(self._usage),
                "runtime": self._runtime_snapshot(),
                "has_video": (
                    self._task_state in TERMINAL_RUN_STATES and self.video_path.exists()
                ),
                "frame_idx": frame_idx,
                "frame_available": frame_available,
                "n_steps": len(self._timeline),
                "interaction": self._interaction_snapshot_locked(),
                **self._session_fields_locked(),
            }

    def run_info(self) -> dict[str, Any]:
        return {"id": self.run_id}

    def run_detail(self) -> dict[str, Any]:
        with self._lock:
            frame_idx, frame_available = self._frame_snapshot()
            return {
                "state": self._visible_state_locked(),
                "terminated": self._terminated,
                "truncated": self._truncated,
                "error": self._error,
                "usage": dict(self._usage),
                "runtime": self._runtime_snapshot(),
                "timeline": list(self._timeline),
                "has_video": (
                    self._task_state in TERMINAL_RUN_STATES and self.video_path.exists()
                ),
                "frame_idx": frame_idx,
                "frame_available": frame_available,
                "interaction": self._interaction_snapshot_locked(),
                **self._session_fields_locked(),
            }
