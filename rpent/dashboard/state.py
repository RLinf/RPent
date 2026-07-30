"""Thread-safe in-memory state for dashboard live runs."""
from __future__ import annotations

import threading
from pathlib import Path
from typing import TYPE_CHECKING, Any

from rpent.dashboard.events import (
    DashboardEvent,
    RunFinishedEvent,
    RuntimeStatusEvent,
    ToolResultEvent,
    TranscriptEvent,
    UsageEvent,
)

if TYPE_CHECKING:
    from rpent.envs.env_spec import RunConfig

RUNTIME_COMPONENTS = ("env", "vla", "sam3")
RUNTIME_STATUSES = {"pending", "starting", "ready", "failed"}


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
        self._state = "running"
        self._terminated = False
        self._usage = {"in": 0, "out": 0, "tool_calls": 0}
        self._runtime = {
            component: {"status": "pending", "error": None}
            for component in RUNTIME_COMPONENTS
        }
        self._events: list[dict[str, Any]] = []
        self._timeline: list[dict[str, Any]] = []
        self._frame_png: bytes | None = None
        self._frame_cam_png: bytes | None = None
        self._frame_idx = -1

    @classmethod
    def from_run_config(cls, run_config: RunConfig) -> DashboardState:
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
    def enabled(self) -> bool:
        return True

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
        return {
            component: dict(status)
            for component, status in self._runtime.items()
        }

    def _apply_tool_result(self, event: ToolResultEvent) -> None:
        name = event.name
        result = event.result
        if not isinstance(result, dict):
            return
        image_path = result.get("overlay_path") or result.get("image_path")
        image_cam_path = result.get("image_cam_path")
        self._update_frame(
            step=result.get("step"),
            image=Path(image_path).read_bytes() if image_path else None,
            image_cam=Path(image_cam_path).read_bytes() if image_cam_path else None,
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

    def _update_frame(
        self,
        *,
        step: Any,
        image: bytes | None = None,
        image_cam: bytes | None = None,
    ) -> None:
        if image is None and image_cam is None:
            return
        try:
            frame_idx = int(step)
        except Exception:
            frame_idx = None
        with self._lock:
            if frame_idx is not None:
                self._frame_idx = frame_idx
            if image is not None:
                self._frame_png = bytes(image)
            if image_cam is not None:
                self._frame_cam_png = bytes(image_cam)

    def _finish(self, event: RunFinishedEvent) -> None:
        terminated = event.terminated
        with self._lock:
            self._state = "done"
            if terminated is None:
                terminated = any(item.get("terminated") for item in self._timeline)
            self._terminated = bool(terminated)

    def events_since(self, since: int) -> list[dict[str, Any]]:
        with self._lock:
            return list(self._events[since:])

    def frame(self, kind: str) -> bytes | None:
        with self._lock:
            if kind == "camera":
                return self._frame_cam_png
            return self._frame_png

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
            return self._state == "done" and self.video_path.exists()

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            return {
                "state": self._state,
                "terminated": self._terminated,
                "usage": dict(self._usage),
                "runtime": self._runtime_snapshot(),
                "has_video": self._state == "done" and self.video_path.exists(),
                "frame_idx": self._frame_idx,
                "n_steps": len(self._timeline),
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
                "runtime": self._runtime_snapshot(),
                "n_steps": len(self._timeline),
            }

    def run_detail(self) -> dict[str, Any]:
        with self._lock:
            return {
                "state": self._state,
                "terminated": self._terminated,
                "suite": self.suite,
                "name": self.name,
                "task": self.task,
                "seed": self.seed,
                "usage": dict(self._usage),
                "runtime": self._runtime_snapshot(),
                "timeline": list(self._timeline),
                "has_video": self._state == "done" and self.video_path.exists(),
                "frame_idx": self._frame_idx,
            }
