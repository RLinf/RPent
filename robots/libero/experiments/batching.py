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

"""Bounded planner tool batches, preserving each primitive's execution path."""

from __future__ import annotations

import json
import threading
import time
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError, create_model

from robots.libero.toolkit import LiberoToolkit
from rpent.tools import Tool, ToolResult


class Action(BaseModel):
    """One existing primitive and its already-known arguments."""

    model_config = ConfigDict(extra="forbid")
    action: Literal[
        "move_to",
        "move_pose",
        "rotate_wrist",
        "set_gripper",
        "pi0_pick",
        "pi0_doubled",
        "release",
    ]
    arguments: dict[str, Any]


def execution_policy(batch_size: int) -> str:
    """Return the common experiment policy and the selected treatment."""
    common = (
        "Complete the entire task, including transport and placement, and verify "
        "the environment result. Do not reset the episode. Use only the RPent "
        "tools for robot interaction. Do not modify code or experiment files. "
    )
    if batch_size == 0:
        return common + (
            "Choose robot actions using the available ordinary tools and inspect "
            "their feedback before choosing dependent actions."
        )
    return common + (
        f"Use execute_sequence to group up to {batch_size} already-planned robot "
        "actions whenever their arguments are known. Do perception first when "
        "coordinates are unknown. After a sequence returns, inspect its final "
        "observation and stop reason before planning further. Do not continue "
        "carrying after a failed grasp."
    )


def stop_reason(action: Action, record: Any, result: ToolResult) -> str | None:
    """Interpret LIBERO primitive results at a tool boundary."""
    if result.is_error:
        return "tool_error"
    if record.terminated:
        return "native_success"
    if record.truncated:
        return "environment_truncated"
    data = record.result or {}
    if action.action == "pi0_doubled":
        # Its success field means task termination, not contact-skill success.
        return "contact_requires_observation"
    if action.action == "pi0_pick" and data.get("success") is not True:
        return "pick_unconfirmed"
    if data.get("converged") is False or data.get("position_hold_ok") is False:
        return "action_failed"
    if action.action in {"move_to", "move_pose"}:
        distance = data.get("final_dist_m")
        if distance is None or distance > action.arguments.get("tol", 0.012) + 0.0001:
            return "position_not_reached"
    return None


class BatchToolkit(LiberoToolkit):
    """Experiment-only dispatch; each child uses the existing toolkit scheduler."""

    def __init__(self, *, batch_size: int, event_path: Path, **kwargs: Any) -> None:
        if batch_size < 0:
            raise ValueError("batch_size must be nonnegative; zero is ordinary mode")
        self._dispatch_lock = threading.RLock()
        self._batch_cancelled = threading.Event()
        self._event_path = event_path
        self._sequence_schema = create_model(
            "SequenceArgs",
            __config__=ConfigDict(extra="forbid"),
            steps=(list[Action], Field(min_length=1, max_length=max(1, batch_size))),
        )
        self._batch_size = batch_size
        super().__init__(**kwargs)
        if batch_size:
            self.add_tool(
                Tool(
                    name="execute_sequence",
                    description=(
                        f"Execute 1–{batch_size} existing robot tools in order without "
                        "another planner response. All arguments must be known before "
                        "execution; no references to future results. Preserve each "
                        "tool's control loop and state recording. Return early on "
                        "error, unconfirmed pick, failed motion, cancellation or "
                        "environment termination. Always return after pi0_doubled "
                        "so the planner can inspect contact results. Return child "
                        "results and the final observation."
                    ),
                    args_schema=self._sequence_schema,
                    _input_schema=self._sequence_schema.model_json_schema(),
                    handler=self._sequence,
                )
            )

    def _sequence(self, **payload: Any) -> ToolResult:
        try:
            steps = self._sequence_schema.model_validate(payload, strict=True).steps
            for step in steps:
                self._tools[step.action].args_schema.model_validate(
                    step.arguments, strict=True
                )
        except (ValidationError, KeyError) as exc:
            return ToolResult(data={"error": str(exc), "actions_executed": 0})
        summaries = []
        last = ToolResult()
        reason = "sequence_complete"
        for step in steps:
            if self._batch_cancelled.is_set():
                reason = "cancelled"
                break
            if self.solved():
                reason = "native_success"
                break
            # Do not acquire an outer scheduler operation: child calls own it.
            last = super().execute_tool(step.action, step.arguments)
            record = self.state.latest_record()
            summaries.append(
                {
                    "action": step.action,
                    "result": record.result,
                    "step_idx": record.step_idx,
                }
            )
            reason = stop_reason(step, record, last) or "sequence_complete"
            if reason != "sequence_complete":
                break
        return ToolResult(
            data={
                **last.data,
                "sequence": summaries,
                "stop_reason": reason,
                "actions_executed": len(summaries),
                "actions_planned": len(steps),
            },
            images=last.images,
        )

    def execute_tool(self, name: str, input_dict: dict[str, Any]) -> ToolResult:
        """Serialize batches and ordinary calls, logging only outer dispatches."""
        with self._dispatch_lock:
            started = time.perf_counter()
            if self._batch_cancelled.is_set():
                result = ToolResult(data={"error": "tool operation interrupted"})
            elif name == "execute_sequence" and self._batch_size:
                result = self._sequence(**input_dict)
            else:
                result = super().execute_tool(name, input_dict)
            with self._event_path.open("a") as stream:
                stream.write(
                    json.dumps(
                        {
                            "name": name,
                            "elapsed_s": time.perf_counter() - started,
                            "error": result.data.get("error"),
                            "actions_executed": result.data.get("actions_executed"),
                            "stop_reason": result.data.get("stop_reason"),
                        }
                    )
                    + "\n"
                )
            return result

    def cancel_active_and_wait(self) -> None:
        """Stop remaining batch children as well as active primitive execution."""
        self._batch_cancelled.set()
        super().cancel_active_and_wait()
        with self._dispatch_lock:
            pass

    def resume_calls(self) -> None:
        """Reopen admission after the cancelled batch has drained."""
        with self._dispatch_lock:
            super().resume_calls()
            self._batch_cancelled.clear()
