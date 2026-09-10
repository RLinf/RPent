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

"""Native tool execution with one active invocation per toolkit."""

from __future__ import annotations

import json
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Generic

import numpy as np
from pydantic import ValidationError

from rpent.dashboard.events import (
    DashboardEventSink,
    NullDashboardEventSink,
    StepRecordEvent,
)
from rpent.memory import MemoryManager
from rpent.session import EnvState
from rpent.tools.base import (
    RobotT,
    Tool,
    ToolContext,
    ToolResult,
)
from rpent.tools.common_tools import COMMON_TOOLS
from rpent.utils.logging import get_logger

logger = get_logger("tools")


@dataclass(slots=True)
class _ToolOperation:
    cancel_event: threading.Event = field(default_factory=threading.Event)
    done_event: threading.Event = field(default_factory=threading.Event)


class Toolkit(Generic[RobotT]):
    """A fixed tool collection and its execution resources for one planner session.

    Robot tools submit RGB frames through ctx.record_frame(). Robot toolkits
    save their action clips, episode video, and replay recipes.
    """

    def __init__(
        self,
        *,
        state: EnvState,
        memory: MemoryManager,
        robot: RobotT,
        output_dir: str | Path,
        tools: tuple[Tool, ...],
        dashboard_events: DashboardEventSink | None = None,
    ) -> None:
        self._state = state
        self._memory = memory
        self._robot = robot
        self._task_output_dir = Path(output_dir).resolve()
        self._dashboard_events = dashboard_events or NullDashboardEventSink()
        self._tools: dict[str, Tool] = {
            item.name: item for item in (*COMMON_TOOLS, *tools)
        }
        self._operation_lock = threading.Lock()
        self._active_operation: _ToolOperation | None = None
        self._finish_result: dict[str, str] | None = None
        self._frames: list[np.ndarray] = []

    @property
    def state(self) -> EnvState:
        return self._state

    @property
    def memory(self) -> MemoryManager:
        return self._memory

    @property
    def finish_result(self) -> dict[str, str] | None:
        """Return the accepted finish result for the planner; admission is unchanged."""
        result = self._finish_result
        return dict(result) if result is not None else None

    def list_tools(self) -> tuple[Tool, ...]:
        return tuple(self._tools.values())

    def record_frame(self, rgb: np.ndarray) -> None:
        """Collect one environment-step image for action and episode videos."""
        self._frames.append(np.ascontiguousarray(np.asarray(rgb)))

    def execute_tool(self, name: str, arguments: dict) -> ToolResult:
        """Validate and execute one call, then capture its observation."""
        tool = self._tools.get(name)
        if tool is None:
            return ToolResult(error=f"Unknown tool: {name}")
        try:
            args = tool.args_schema.model_validate(arguments)
        except ValidationError as exc:
            errors = exc.errors(
                include_url=False, include_context=False, include_input=False
            )
            details = json.dumps({"errors": errors})
            return ToolResult(error=f"Invalid arguments for {name}.\n{details}")
        with self._operation_lock:
            if self._active_operation is not None:
                return ToolResult(error="another tool operation is still active")
            operation = _ToolOperation()
            self._active_operation = operation

        try:
            if operation.cancel_event.is_set():
                return ToolResult(error="Tool call cancelled.")
            ctx = ToolContext(
                state=self._state,
                memory=self._memory,
                robot=self._robot,
                output_dir=self._task_output_dir,
                record_frame=self.record_frame,
                _cancel_event=operation.cancel_event,
            )
            capture = (
                not tool.readonly and tool not in COMMON_TOOLS and name != "finish"
            )
            started = time.perf_counter()
            # Read fields directly so nested models reach the handler intact.
            kwargs = {name: getattr(args, name) for name in type(args).model_fields}
            try:
                result = tool.handler(**kwargs, ctx=ctx)
            except Exception as exc:
                logger.exception("Tool %s failed", name)
                result = ToolResult(error=str(exc)[:500])
            if capture:
                elapsed_s = time.perf_counter() - started
                previous = self._state.latest_record()
                try:
                    observation_data, observation_images = self._capture_observation(
                        command={"action": tool.name, **args.model_dump()},
                        result=result,
                        elapsed_s=elapsed_s,
                    )
                    # The observation replaces action data; retain the action's error.
                    result.data = observation_data
                    result.images = result.images + observation_images
                except Exception as exc:
                    logger.exception("State capture failed after %s", tool.name)
                    error = f"State capture failed: {str(exc)[:500]}"
                    if result.is_error:
                        error = f"{result.error}\n{error}"
                    result.error = error
                record = self._state.latest_record()
                if record is not None and record is not previous:
                    try:
                        self._dashboard_events.emit(
                            StepRecordEvent(record=record, env_state=self._state)
                        )
                    except Exception:
                        logger.exception(
                            "Dashboard failed to publish step %s", record.step_idx
                        )
            if name == "finish" and not result.is_error:
                self._finish_result = {
                    key: result.data[key] for key in ("status", "summary")
                }
            # Images are logged by their owning artifact paths, not their bytes.
            logger.info(
                "Tool %s result: %s",
                name,
                json.dumps(
                    result.to_dict(),
                    ensure_ascii=False,
                    allow_nan=False,
                ),
            )
            return result
        finally:
            with self._operation_lock:
                self._active_operation = None
                operation.done_event.set()

    def _capture_observation(
        self, *, command: dict[str, Any], result: ToolResult, elapsed_s: float
    ) -> tuple[dict[str, Any], list[bytes]]:
        """Save a step with the action log and return its observation and images.

        Returned data replaces the action's data and must include the recorded
        step and its artifact names. Include action details in that data where
        needed (e.g. log.result). Robot toolkits save action video artifacts.
        The executor appends the images and retains the action's error. Raise if
        capture fails; an already saved step is still published to the Dashboard.
        """
        raise NotImplementedError("This toolkit does not capture robot observations.")

    def cancel_active_and_wait(self) -> None:
        """Request cancellation and wait for the active tool to return."""
        with self._operation_lock:
            operation = self._active_operation
            if operation is None:
                return
            operation.cancel_event.set()
        operation.done_event.wait()

    def close(self) -> None:
        """Release robot resources at the end of a run. Default: no-op."""

    def solved(self) -> bool:
        raise NotImplementedError

    def write_recipe(self, recipe_tag: str) -> str | None:
        """Write a replay recipe for this robot, if supported."""
        return None
