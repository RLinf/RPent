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

"""Planner tools for a Python session borrowing an existing robot toolkit."""

from __future__ import annotations

import json
import math
import time
from dataclasses import replace
from functools import partial
from pathlib import Path
from typing import Any, Literal

from rpent.code_policy.runtime import CodeSession
from rpent.dashboard.events import NullDashboardEventSink
from rpent.tools.base import Tool, ToolResult, tool
from rpent.tools.toolkit import Toolkit

MEMORY_TOOLS = frozenset({"read_text_file", "write_text_file", "list_dir"})
# Operator requests wait for a person, which can outlast a cell's time limit.
OPERATOR_TOOLS = frozenset({"request_scene_reset", "request_operator_verdict"})
MAX_IMAGES = 8
MAX_IMAGE_BYTES = 16 * 1024 * 1024


class CodePolicyTools(Toolkit):
    """Expose code execution without taking ownership of the robot toolkit.

    Python cells and operator requests are serialized at this boundary. Robot
    calls acquire only the underlying toolkit's scheduler, so each primitive
    retains its own validation, observation capture and cancellation semantics.
    A worker timeout or crash resumes automatically once the robot drains; an
    external stop waits for ``resume_calls``.
    """

    def __init__(
        self,
        toolkit: Toolkit,
        *,
        output_dir: Path,
        timeout_s: float = 120.0,
        max_calls: int = 100,
        include_images: bool = True,
    ) -> None:
        if not math.isfinite(timeout_s) or timeout_s <= 0:
            raise ValueError("code timeout must be finite and positive")
        if type(max_calls) is not int or max_calls <= 0:
            raise ValueError("code call budget must be a positive integer")
        self._toolkit = toolkit
        self._declarations = {
            item.name: item
            for item in toolkit.list_tools()
            if item.name not in MEMORY_TOOLS
        }
        if "python" in self._declarations:
            raise ValueError("Robot tools conflict with the code-policy API")
        self._include_images = include_images
        self._timeout_s = timeout_s
        self._max_calls = max_calls
        self._output_dir = Path(output_dir) / "code_policy"
        self._output_dir.mkdir(parents=True, exist_ok=True)
        self._stop: Literal["internal", "external"] | None = None
        self._cancel_error: Exception | None = None
        self._calls = 0
        self._images: list[bytes] = []
        self._last_result: dict[str, Any] | None = None
        self._finish_result: dict[str, Any] | None = None
        super().__init__(
            dashboard_events=NullDashboardEventSink(),
            state=toolkit.state,
            memory=None,
        )
        self._session = CodeSession(
            tools=[item.name for item in self.robot_tools()],
            dispatch=self._dispatch,
            workspace=self._output_dir,
            on_interrupt=self._stop_robot,
        )

    def _register_common_tools(self) -> None:
        self.add_tool(self.python)
        for declaration in self._declarations.values():
            if declaration.name in OPERATOR_TOOLS:
                self.add_tool(
                    replace(
                        declaration.with_handler(
                            partial(self._request_operator, declaration.name)
                        ),
                        readonly=False,
                    )
                )

    def robot_tools(self) -> tuple[Tool, ...]:
        """Return the original declarations available through the Python proxy."""
        return tuple(
            item
            for item in self._declarations.values()
            if item.name not in OPERATOR_TOOLS
        )

    @tool
    def python(self, code: str) -> ToolResult:
        """Execute Python with the robot API `robo` in a persistent session.

        Call registered tools as robo.<name>(**keyword_arguments). Results retain
        their original dictionaries and error fields. Variables persist between
        calls; use print(...) to return values in stdout alongside the latest
        tool images. End the cell for required visual checks. A successful
        robo.finish(...) ends the cell and the task. Timeout/cancellation
        discards variables, but does not undo executed actions.

        Args:
            code: Python source. Prefer bounded loops with explicit feedback checks.
        """
        self._calls = 0
        self._images = []
        self._last_result = None
        result = self._session.execute(code, timeout_s=self._timeout_s)
        result.update(
            primitive_call_count=self._calls,
            last_tool_result=self._last_result,
            finished=self._finish_result is not None,
        )
        result_path = self._output_dir / f"cell_{result['cell']:06d}.json"
        result_path.write_text(
            json.dumps(result, indent=2, ensure_ascii=False, default=str) + "\n",
            encoding="utf-8",
        )
        if self._finish_result is not None:
            # Planners end on the finish payload; later cells are rejected.
            self._scheduler.cancel(close=True)
            return ToolResult(data=self._finish_result, images=self._images)
        return ToolResult(data=result, images=self._images)

    def execute_tool(self, name: str, input_dict: dict[str, Any]) -> ToolResult:
        """Use Toolkit dispatch, then recover an interrupted Python session."""
        result = super().execute_tool(name, input_dict)
        if name == "python" and result.data.get("namespace_reset"):
            # Resume only after Toolkit has released the active cell.
            self._resume(external=False)
        return result

    def _request_operator(self, name: str, **arguments: Any) -> ToolResult:
        result = self._toolkit.execute_tool(name, arguments)
        return result if self._include_images else ToolResult(data=result.data)

    def _capture_state(
        self,
        name: str,
        input_dict: dict[str, Any],
        native: ToolResult,
        elapsed_s: float,
    ) -> ToolResult:
        # Each primitive already captures and publishes its own observation.
        return native

    def _dispatch(self, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        with self._scheduler.condition:
            if self._scheduler.paused or self._scheduler.closed:
                raise RuntimeError("Code-policy execution was cancelled")
        if self._finish_result is not None:
            raise RuntimeError("The task has finished; no further robot calls")
        if not self._declarations[name].readonly:
            self._images = []
        if self._calls >= self._max_calls:
            raise RuntimeError("Primitive call budget exhausted for this cell")
        self._calls += 1
        started = time.perf_counter()
        entry: dict[str, Any] = {
            "cell": self._session.cell,
            "call": self._calls,
            "tool": name,
            "arguments": arguments,
        }
        try:
            result = self._toolkit.execute_tool(name, arguments)
            self._last_result = result.to_dict()
            if not result.is_error and result.data.get("_finish") is True:
                self._finish_result = self._last_result
            entry["result"] = self._last_result
            if self._include_images and result.images:
                images = []
                size = 0
                for image in result.images[:MAX_IMAGES]:
                    if size + len(image) > MAX_IMAGE_BYTES:
                        break
                    images.append(image)
                    size += len(image)
                self._images = images
            if result.data.get("interrupted"):
                with self._scheduler.condition:
                    # A robot-side stop is external unless this toolkit paused.
                    if not self._scheduler.paused:
                        self._stop = "external"
                self._session.cancel()
            # Match ToolResult's public JSON representation without its text limit.
            return json.loads(
                json.dumps(
                    self._last_result,
                    allow_nan=False,
                    default=str,
                )
            )
        except Exception as exc:
            entry["error"] = f"{type(exc).__name__}: {exc}"
            raise
        finally:
            entry["elapsed_s"] = time.perf_counter() - started
            with (self._output_dir / "calls.jsonl").open("a", encoding="utf-8") as file:
                file.write(json.dumps(entry, ensure_ascii=False, default=str) + "\n")

    def _stop_robot(self) -> None:
        """Pause admission and drain the robot once per worker interruption."""
        with self._scheduler.condition:
            self._stop = self._stop or "internal"
            self._scheduler.cancel()
        try:
            self._toolkit.cancel_active_and_wait()
        except Exception as exc:
            self._cancel_error = exc

    def cancel_active_and_wait(self) -> None:
        """Stop Python and drain its callbacks and underlying physical operations."""
        with self._scheduler.condition:
            self._stop = "external"
        self._session.cancel()
        super().cancel_active_and_wait()
        if self._cancel_error is not None:
            raise RuntimeError("Code-policy robot cancellation failed") from (
                self._cancel_error
            )

    def resume_calls(self) -> None:
        """Resume a drained session; an interrupted worker gets a fresh namespace."""
        self._resume(external=True)

    def _resume(self, *, external: bool) -> None:
        with self._scheduler.condition:
            calls = [*self._scheduler.pending, *self._scheduler.active]
        self._scheduler.wait(calls)
        with self._scheduler.condition:
            if (
                self._stop is None
                or self._scheduler.closed
                or (self._stop == "external" and not external)
            ):
                return
            if self._cancel_error is not None:
                raise RuntimeError("Robot cancellation failed") from self._cancel_error
            self._toolkit.resume_calls()
            self._session.resume()
            super().resume_calls()
            self._stop = None

    def close(self) -> None:
        """Release code resources, leaving robot finalization to the run owner."""
        self._scheduler.cancel(close=True)
        try:
            self.cancel_active_and_wait()
        finally:
            self._session.close()
        # The original toolkit remains available to operator verdict/recipe hooks.
        self._toolkit.resume_calls()
