"""Thread-safe in-memory state for dashboard live runs."""
from __future__ import annotations

import threading
import time
import uuid
from dataclasses import replace
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

from rpent.dashboard.events import (
    DashboardEvent,
    RunFinishedEvent,
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
    from rpent.envs.env_spec import RunConfig

RUNTIME_COMPONENTS = ("env", "vla", "sam3")
RUNTIME_STATUSES = {"pending", "starting", "ready", "failed"}
FRAME_KINDS = ("camera", "wrist")
TERMINAL_RUN_STATES = {"succeeded", "failed", "cancelled"}
_PLANNER_ACTIVITIES = {"starting", "idle", "busy", "ended"}
InterruptRequestResult = Literal["accepted", "duplicate", "noop"]


class DashboardState:
    """Thread-safe dashboard state for one run."""

    def __init__(
        self,
        *,
        run_id: str,
        name: str,
        suite: str,
        task: int,
        seed: int,
        output_dir: str,
        video_path: str,
    ) -> None:
        self.run_id = run_id
        self.name = name
        self.suite = suite
        self.task = task
        self.seed = seed
        self.output_dir = Path(output_dir)
        self.video_path = Path(video_path)

        self._lock = threading.Lock()
        self._condition = threading.Condition(self._lock)
        self._state = "starting"
        self._terminated = False
        self._error: str | None = None
        self._finish_reason: str | None = None
        self._usage = {"in": 0, "out": 0, "tool_calls": 0}
        self._runtime = {
            component: {"status": "pending", "error": None}
            for component in RUNTIME_COMPONENTS
        }
        self._events: list[dict[str, Any]] = []
        self._timeline: list[dict[str, Any]] = []
        self._frames: dict[str, bytes] = {}
        self._frame_idx = -1
        self._session_id: str | None = None
        self._accepting_input = False
        self._planner_activity: PlannerActivity = "starting"
        self._interrupt_requested = False
        self._interrupt_in_flight = False
        self._messages: list[DashboardMessage] = []
        self._messages_by_id: dict[str, DashboardMessage] = {}
        self._last_interaction_error: str | None = None
        self._interaction_version = 0

    @classmethod
    def from_run_config(
        cls,
        run_config: RunConfig,
    ) -> DashboardState:
        """Build the Dashboard projection for one parsed environment run."""
        task_desc = run_config.task_desc
        suite = str(task_desc["suite"])
        task = int(task_desc["task"])
        seed = int(task_desc["seed"])
        output_dir = run_config.output_dir
        return cls(
            run_id=f"{suite}/{output_dir.name}",
            name=run_config.recipe_tag,
            suite=suite,
            task=task,
            seed=seed,
            output_dir=str(output_dir),
            video_path=str(output_dir / "episode.mp4"),
        )

    @property
    def session_id(self) -> str | None:
        """Return the Session id used by interaction HTTP routes."""
        with self._lock:
            return self._session_id

    @property
    def planner_activity(self) -> PlannerActivity:
        """Return the current planner input activity."""
        with self._lock:
            return self._planner_activity

    @property
    def accepting_input(self) -> bool:
        """Whether the Session currently accepts new user messages."""
        with self._lock:
            return self._accepting_input

    @property
    def interrupt_requested(self) -> bool:
        """Whether an Esc request is queued or being handled."""
        with self._lock:
            return self._interrupt_requested

    @property
    def interaction_version(self) -> int:
        """Monotonic version for bridges waiting on interaction changes."""
        with self._lock:
            return self._interaction_version

    def enable_interaction(self, session_id: str | None = None) -> None:
        """Attach an initially-starting planner interaction to this run.

        Calling this method more than once with the same Session id is
        idempotent. A run that has already finished cannot start a Session.
        """
        resolved_session_id = self.run_id if session_id is None else str(session_id)
        if not resolved_session_id.strip():
            raise ValueError("session_id must not be blank")
        with self._condition:
            if self._state in TERMINAL_RUN_STATES:
                raise InteractionUnavailableError("run has already ended")
            if self._session_id is not None:
                if self._session_id != resolved_session_id:
                    raise DashboardInteractionError(
                        "run already has a different Dashboard Session"
                    )
                return
            self._session_id = resolved_session_id
            self._accepting_input = False
            self._planner_activity = "starting"
            self._interaction_changed_locked()

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
            self._require_interaction_locked()
            if self._planner_activity == "ended" and activity != "ended":
                raise InteractionUnavailableError("Dashboard Session has ended")
            if activity == "ended":
                self._seal_interaction_locked()
                return
            changed = self._planner_activity != activity
            self._planner_activity = activity
            if accepting_input is not None:
                requested_accepting = bool(accepting_input)
                if self._state in TERMINAL_RUN_STATES:
                    requested_accepting = False
                changed = changed or self._accepting_input != requested_accepting
                self._accepting_input = requested_accepting
            if changed:
                self._interaction_changed_locked()

    def submit_message(self, text: str) -> DashboardMessage:
        """Create one pending user message and notify the owning bridge."""
        if not isinstance(text, str) or not text.strip():
            raise ValueError("message text must not be blank")
        with self._condition:
            self._require_interaction_locked()
            if (
                not self._accepting_input
                or self._planner_activity == "ended"
                or self._state in TERMINAL_RUN_STATES
            ):
                raise InteractionUnavailableError(
                    "Dashboard Session is not accepting input"
                )
            message = DashboardMessage(
                message_id=f"msg_{uuid.uuid4().hex}",
                text=text,
                created_at=time.time(),
                status="pending",
            )
            self._messages.append(message)
            self._messages_by_id[message.message_id] = message
            self._interaction_changed_locked()
            return replace(message)

    def withdraw_message(self, message_id: str) -> DashboardMessage:
        """Atomically withdraw a message that is still pending."""
        with self._condition:
            self._require_interaction_locked()
            message = self._message_locked(message_id)
            if message.status != "pending":
                raise DashboardMessageConflictError(
                    f"message is {message.status}, not pending"
                )
            message.status = "withdrawn"
            message.error = None
            self._interaction_changed_locked()
            return replace(message)

    def begin_pending_batch(self) -> list[DashboardMessage]:
        """Atomically claim all currently pending messages in creation order."""
        with self._condition:
            self._require_interaction_locked()
            if self._planner_activity == "ended":
                return []
            batch = [
                message for message in self._messages if message.status == "pending"
            ]
            for message in batch:
                message.status = "sending"
                message.error = None
            if batch:
                self._interaction_changed_locked()
            return [replace(message) for message in batch]

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
            self._require_interaction_locked()
            if self._interrupt_requested:
                return "duplicate"
            if self._planner_activity != "busy" or self._state in TERMINAL_RUN_STATES:
                return "noop"
            self._interrupt_requested = True
            self._interrupt_in_flight = False
            self._interaction_changed_locked()
            return "accepted"

    def claim_interrupt_request(self) -> bool:
        """Claim a queued interrupt while keeping it visibly requested."""
        with self._condition:
            self._require_interaction_locked()
            if not self._interrupt_requested or self._interrupt_in_flight:
                return False
            self._interrupt_in_flight = True
            self._interaction_changed_locked()
            return True

    def complete_interrupt(self, error: str | None = None) -> None:
        """Complete the claimed SDK interrupt, successfully or with an error."""
        with self._condition:
            self._require_interaction_locked()
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

    def interaction_snapshot(self) -> dict[str, Any]:
        """Return a detached, refresh-safe view of Session interaction state."""
        with self._lock:
            return self._interaction_snapshot_locked()

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
        if isinstance(event, RunFinishedEvent):
            self._finish(event)
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
        with self._lock:
            if self._state == "starting":
                self._state = "running"

    def _finish(self, event: RunFinishedEvent) -> None:
        if event.state not in TERMINAL_RUN_STATES:
            raise ValueError(f"invalid terminal run state: {event.state!r}")
        terminated = event.terminated
        with self._condition:
            if self._state in TERMINAL_RUN_STATES:
                return
            self._state = event.state
            if terminated is None:
                terminated = any(item.get("terminated") for item in self._timeline)
            self._terminated = bool(terminated)
            self._finish_reason = event.reason
            self._error = None if event.error is None else str(event.error)
            self._seal_interaction_locked()

    def _transition_sending_message_locked(
        self,
        message_id: str,
        *,
        status: Literal["sent", "failed"],
        error: str | None,
    ) -> DashboardMessage:
        self._require_interaction_locked()
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

    def _require_interaction_locked(self) -> None:
        if self._session_id is None:
            raise InteractionUnavailableError(
                "Dashboard interaction is not enabled for this run"
            )

    def _seal_interaction_locked(self) -> None:
        changed = self._planner_activity != "ended" or self._accepting_input
        self._planner_activity = "ended"
        self._accepting_input = False
        self._interrupt_requested = False
        self._interrupt_in_flight = False
        for message in self._messages:
            if message.status not in {"pending", "sending"}:
                continue
            message.status = "unsent"
            message.error = None
            changed = True
        if changed:
            self._interaction_changed_locked()

    def _interaction_changed_locked(self) -> None:
        self._interaction_version += 1
        self._condition.notify_all()

    def _interaction_snapshot_locked(self) -> dict[str, Any]:
        return {
            "session_id": self._session_id,
            "accepting_input": self._accepting_input,
            "planner_activity": self._planner_activity,
            "interrupt_requested": self._interrupt_requested,
            "messages": [message.as_dict() for message in self._messages],
            "last_error": self._last_interaction_error,
        }

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
            return self._state in TERMINAL_RUN_STATES and self.video_path.exists()

    def _frame_snapshot(self) -> tuple[int, dict[str, bool]]:
        available = {kind: kind in self._frames for kind in FRAME_KINDS}
        return self._frame_idx, available

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            frame_idx, frame_available = self._frame_snapshot()
            return {
                "state": self._state,
                "terminated": self._terminated,
                "error": self._error,
                "finish_reason": self._finish_reason,
                "usage": dict(self._usage),
                "runtime": self._runtime_snapshot(),
                "has_video": (
                    self._state in TERMINAL_RUN_STATES and self.video_path.exists()
                ),
                "frame_idx": frame_idx,
                "frame_available": frame_available,
                "n_steps": len(self._timeline),
                "interaction": self._interaction_snapshot_locked(),
            }

    def run_info(self) -> dict[str, Any]:
        with self._lock:
            return {
                "id": self.run_id,
                "name": self.name,
                "suite": self.suite,
                "task": self.task,
                "seed": self.seed,
                "state": self._state,
                "error": self._error,
                "finish_reason": self._finish_reason,
                "runtime": self._runtime_snapshot(),
                "n_steps": len(self._timeline),
            }

    def run_detail(self) -> dict[str, Any]:
        with self._lock:
            frame_idx, frame_available = self._frame_snapshot()
            return {
                "state": self._state,
                "terminated": self._terminated,
                "error": self._error,
                "finish_reason": self._finish_reason,
                "suite": self.suite,
                "name": self.name,
                "task": self.task,
                "seed": self.seed,
                "usage": dict(self._usage),
                "runtime": self._runtime_snapshot(),
                "timeline": list(self._timeline),
                "has_video": (
                    self._state in TERMINAL_RUN_STATES and self.video_path.exists()
                ),
                "frame_idx": frame_idx,
                "frame_available": frame_available,
                "interaction": self._interaction_snapshot_locked(),
            }
