"""Thread-safe in-memory state for dashboard live runs."""
from __future__ import annotations

import threading
import uuid
from dataclasses import dataclass, replace
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

from rpent.dashboard.events import (
    DashboardEvent,
    RunStartedEvent,
    RuntimeStatusEvent,
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
    from rpent.dashboard.commands import TaskCommand

RUNTIME_COMPONENTS = ("env", "vla", "sam3")
RUNTIME_STATUSES = {"pending", "starting", "ready", "failed"}
FRAME_KINDS = ("camera", "wrist")
TERMINAL_RUN_STATES = {"succeeded", "failed", "cancelled"}
_PLANNER_ACTIVITIES = {"starting", "idle", "busy", "ended"}
InterruptRequestResult = Literal["accepted", "duplicate", "noop"]
InputMode = Literal["command_only", "conversation", "disabled"]


@dataclass(frozen=True, slots=True)
class ClaimedTask:
    """One last-write-wins task command claimed by the Session controller."""

    number: int
    command: TaskCommand
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
    ) -> None:
        root = Path(output_dir)
        self.run_id = run_id
        self.output_dir = root
        self.video_path = root / "episode.mp4"
        self._session_root = root

        self._lock = threading.Lock()
        self._condition = threading.Condition(self._lock)
        self._task_state: str | None = None
        self._terminated = False
        self._error: str | None = None
        self._usage = {"in": 0, "out": 0, "tool_calls": 0}
        self._runtime = {
            component: {"status": "pending", "error": None}
            for component in RUNTIME_COMPONENTS
        }
        self._events: list[dict[str, Any]] = []
        self._timeline: list[dict[str, Any]] = []
        self._frames: dict[str, bytes] = {}
        self._frame_idx = -1
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
        self._pending_task: TaskCommand | None = None
        self._current_task: TaskCommand | None = None
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

    def submit_input(self, text: str) -> DashboardMessage | TaskCommand:
        """Route a local task command or a normal conversation message."""
        from rpent.dashboard.commands import parse_dashboard_command

        if not isinstance(text, str) or not text.strip():
            return self._submit_message(text)
        try:
            command = parse_dashboard_command(text)
        except ValueError as exc:
            with self._condition:
                self._control_error = str(exc)
                self._interaction_changed_locked()
            raise
        if command is not None:
            self.request_task(command)
            return command
        return self._submit_message(text)

    def request_task(self, command: TaskCommand) -> None:
        """Atomically record the latest desired TaskRun and close old input."""
        with self._condition:
            if self._session_state == "fatal":
                raise InteractionUnavailableError("Dashboard Session is fatal")
            if self._session_state == "starting_shared_services":
                raise InteractionUnavailableError(
                    "Dashboard Session is still starting shared services"
                )

            self._pending_task = command
            self._control_error = None
            self._control_feedback = [
                f"Task selected: {command.suite} / task {command.task} / seed {command.seed}"
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

            command = self._pending_task
            self._pending_task = None
            self._task_generation += 1
            number = self._task_generation
            output_dir = (
                self._session_root
                / "tasks"
                / f"{number:04d}_{command.suite}_t{command.task}_s{command.seed}"
            )
            self._begin_task_locked(command, number=number, output_dir=output_dir)
            return ClaimedTask(
                number=number,
                command=command,
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
            self._error = None if error is None else str(error)
            self._task_replacement_requested = False
            self._seal_interaction_locked()
            self._runtime["env"] = {"status": "pending", "error": None}
            self._session_state = (
                "task_starting" if self._pending_task is not None else "ready"
            )
            if error is not None:
                self._control_error = str(error)
            self._interaction_changed_locked()

    def _begin_task_locked(
        self,
        command: TaskCommand,
        *,
        number: int,
        output_dir: Path,
    ) -> None:
        self._current_task = command
        self.output_dir = output_dir
        self.video_path = output_dir / "episode.mp4"
        self._session_state = "task_starting"
        self._task_state = "starting"
        self._task_replacement_requested = False
        self._terminated = False
        self._error = None
        self._usage = {"in": 0, "out": 0, "tool_calls": 0}
        self._runtime["env"] = {"status": "pending", "error": None}
        self._events = []
        self._timeline = []
        self._frames = {}
        self._frame_idx = -1
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
        ``query()`` succeeds. Once ended, a Session cannot be reopened.
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
            if self._planner_activity == "ended" or self._task_replacement_requested:
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
                    "in": int(event.inp),
                    "out": int(event.out),
                    "tool_calls": int(event.tool_calls),
                }
            return
        if isinstance(event, RuntimeStatusEvent):
            self._apply_runtime_status(event)
            return
        if isinstance(event, ToolResultEvent):
            self._apply_tool_result(event)
            return
        if isinstance(event, RunStartedEvent):
            self._start()
            return
        raise TypeError(f"unsupported dashboard event: {type(event).__name__}")

    def _apply_runtime_status(self, event: RuntimeStatusEvent) -> None:
        if event.component not in RUNTIME_COMPONENTS:
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

    def _apply_tool_result(self, event: ToolResultEvent) -> None:
        name = event.name
        result = event.result
        if not isinstance(result, dict):
            return
        self._apply_frame_paths(result)
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
        terminated = bool(result.get("libero_terminated"))
        item = {
            "step": step,
            "action": str(command.get("action", name)),
            "args": {k: v for k, v in command.items() if k != "action"},
            "result": log.get("result"),
            "elapsed_s": log.get("elapsed_s"),
            "terminated": terminated,
            "has_action_video": (
                self.output_dir
                / "action_videos"
                / f"step_{step:02d}_{command.get('action', name)}.mp4"
            ).exists(),
        }
        with self._lock:
            self._timeline.append(item)
            self._terminated = self._terminated or terminated

    def _apply_frame_paths(self, result: dict[str, Any]) -> None:
        path_keys = {
            "camera": "image_cam_path",
            "wrist": "image_wrist_path",
        }
        if not any(key in result for key in path_keys.values()):
            return

        frames: dict[str, bytes] = {}
        for kind, key in path_keys.items():
            path = result.get(key)
            if not path:
                continue
            try:
                frames[kind] = Path(path).read_bytes()
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
            self._frames = {
                kind: bytes(data)
                for kind, data in frames.items()
                if kind in FRAME_KINDS
            }
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
        status: Literal["sent", "failed"],
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

    @staticmethod
    def _command_snapshot(command: TaskCommand | None) -> dict[str, Any] | None:
        if command is None:
            return None
        return {
            "suite": command.suite,
            "task": command.task,
            "seed": command.seed,
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
        if kind not in FRAME_KINDS:
            raise ValueError(f"unknown frame kind: {kind!r}")
        with self._lock:
            return self._frames.get(kind)

    def action_video_path(self, step: int) -> Path | None:
        with self._lock:
            for item in self._timeline:
                if int(item.get("step", -1)) != int(step):
                    continue
                video_path = (
                    self.output_dir
                    / "action_videos"
                    / f"step_{int(step):02d}_{item.get('action', '')}.mp4"
                )
                return video_path if video_path.exists() else None
        return None

    def has_video(self) -> bool:
        with self._lock:
            return (
                self._task_state in TERMINAL_RUN_STATES
                and self.video_path.exists()
            )

    def _frame_snapshot(self) -> tuple[int, dict[str, bool]]:
        available = {kind: kind in self._frames for kind in FRAME_KINDS}
        return self._frame_idx, available

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            frame_idx, frame_available = self._frame_snapshot()
            return {
                "state": self._visible_state_locked(),
                "terminated": self._terminated,
                "error": self._error,
                "usage": dict(self._usage),
                "runtime": self._runtime_snapshot(),
                "has_video": (
                    self._task_state in TERMINAL_RUN_STATES
                    and self.video_path.exists()
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
                "error": self._error,
                "usage": dict(self._usage),
                "runtime": self._runtime_snapshot(),
                "timeline": list(self._timeline),
                "has_video": (
                    self._task_state in TERMINAL_RUN_STATES
                    and self.video_path.exists()
                ),
                "frame_idx": frame_idx,
                "frame_available": frame_available,
                "interaction": self._interaction_snapshot_locked(),
                **self._session_fields_locked(),
            }
