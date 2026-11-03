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

"""R2 toolkit integrated with RPent's centralized environment state."""

from __future__ import annotations

import sys
import time
from collections.abc import Callable
from functools import partial
from pathlib import Path
from typing import TYPE_CHECKING, Annotated, Any

from pydantic import Field

from robots.r2 import tools as r2_tools
from rpent.dashboard.events import DashboardEventSink
from rpent.session import EnvState
from rpent.tools import ToolResult, iter_tools, tool
from rpent.tools.toolkit import Toolkit
from rpent.utils.logging import get_output_dir

if TYPE_CHECKING:
    from rpent.memory.manager import MemoryManager

#: Asks the onsite operator a question; ``None`` means nobody can answer.
OperatorReader = Callable[[str], "str | None"]

_MOTIONS = {
    "move_to",
    "move_delta",
    "rotate_delta",
    "set_gripper",
    "look",
    "go_home",
    "move_base",
}


def read_terminal_line(prompt: str) -> str | None:
    """Ask on RPent's terminal, or return ``None`` when it has none."""
    if sys.stdin is None or not sys.stdin.isatty():
        return None
    try:
        return input(prompt).strip()
    except EOFError:
        return None


class R2Toolkit(Toolkit):
    """Common RPent tools plus bounded R2 motion, perception, and state readers.

    R2 has no automatic success check, so the onsite operator judges the
    outcome: ``finish`` takes its status from the operator's verdict, a motion
    after a verdict discards it, and an operator abort refuses further motion.
    """

    def __init__(
        self,
        *,
        runtime_kwargs: dict[str, Any],
        dashboard_events: DashboardEventSink,
        memory: MemoryManager,
        state_output_dir: Path | str | None = None,
        operator_input: OperatorReader = read_terminal_line,
    ) -> None:
        runtime_kwargs = dict(runtime_kwargs)
        sam3_client = runtime_kwargs.pop("sam3_client", None)
        self._operator_input = operator_input
        self._verdict: str | None = None
        self._verdict_notes = ""
        self._operator_aborted = False
        self._operator_unavailable = False
        self._events: list[dict[str, Any]] = []
        state = EnvState(Path(state_output_dir or get_output_dir()))
        super().__init__(dashboard_events=dashboard_events, state=state, memory=memory)
        self._primitives = r2_tools.R2Primitives(
            check_cancelled=self.raise_if_cancelled, **runtime_kwargs
        )
        meta = self._primitives.env.meta
        for definition in iter_tools(self._primitives):
            if definition.name == "move_base" and not meta["base_motion"]:
                continue
            self.add_tool(
                definition.with_handler(partial(self._guard_motion, definition))
            )
        for definition in iter_tools(r2_tools):
            resources: dict[str, Any] = {"state": self._state}
            if definition.name == "segment":
                if sam3_client is None:
                    continue
                resources["sam3_client"] = sam3_client
            self.add_tool(definition.with_handler(partial(definition, **resources)))
        self.add_tool(self.request_operator_verdict)
        finish = self._tools["finish"]
        self.add_tool(
            finish.with_handler(partial(self._finish_with_verdict, finish)),
            replace=True,
        )
        self._state.reset()
        record = r2_tools.dump_state(
            self._primitives,
            self._state,
            command=None,
            result=None,
            elapsed_s=None,
            extra={"robot_info": meta},
        )
        self._publish_step(record)

    def get_env_state(
        self,
        *,
        command: dict[str, Any],
        result: dict[str, Any],
        elapsed_s: float,
    ) -> ToolResult:
        record = r2_tools.dump_state(
            self._primitives,
            self._state,
            command=command,
            result=result,
            elapsed_s=elapsed_s,
        )
        output = r2_tools.view_env_state(record.step_idx, state=self._state)
        output.data["agent_elapsed_s"] = elapsed_s
        return output

    def cancel_active_and_wait(self) -> None:
        """Cancel the running calls, stopping the robot if one of them is moving it."""
        with self._scheduler.condition:
            calls = self._scheduler.cancel()
            active = bool(self._scheduler.active)
        try:
            if active:
                self._primitives.env.request_stop()
        finally:
            self._scheduler.wait(calls)

    def solved(self) -> bool:
        """Return whether the operator judged the task a success."""
        return self._verdict == "success"

    def _event(self, kind: str, **fields: Any) -> None:
        self._events.append(
            {
                "kind": kind,
                "step": self._state.latest_step,
                "timestamp": time.time(),
                **fields,
            }
        )
        self._state.save("operator_events.json", self._events, step=None)

    def _guard_motion(self, motion: Any, **kwargs: Any) -> ToolResult:
        if self._operator_aborted:
            return ToolResult(
                data={
                    "error": "motion refused: the operator aborted this run",
                    "motion_refused": True,
                }
            )
        if self._verdict is not None:
            # The verdict judged a scene this motion is about to change.
            self._event("verdict_discarded", verdict=self._verdict, by=motion.name)
            self._verdict, self._verdict_notes = None, ""
        return motion(**kwargs)

    @tool
    def request_operator_verdict(
        self,
        question: Annotated[
            str,
            Field(json_schema_extra={"default": "Is the task done as instructed?"}),
        ] = "Is the task done as instructed?",
    ) -> ToolResult:
        """Ask the onsite operator to judge the robot's physical result; finish follows their verdict.

        The operator answers success, failure, continue (keep working), or abort
        (stop moving), optionally with notes. Moving the robot afterwards
        discards a success or failure verdict.

        Args:
            question: What the operator should judge, phrased for a person looking at the robot.
        """
        self.raise_if_cancelled()
        response = self._operator_input(
            f"\n[R2] {question}\n"
            "Reply success, failure, continue, or abort, optionally followed by notes: "
        )
        self.raise_if_cancelled()
        if response is None:
            self._operator_unavailable = True
            self._event("verdict_unavailable", question=question)
            return ToolResult(
                data={
                    "error": "no operator can answer from this terminal",
                    "notice": "finish now records the run as unverified failure",
                }
            )
        parts = response.split(maxsplit=1)
        verdict = parts[0].lower() if parts else ""
        notes = parts[1] if len(parts) > 1 else ""
        if verdict not in {"success", "failure", "continue", "abort"}:
            return ToolResult(
                data={
                    "error": f"invalid operator reply {response!r}",
                    "expected": ["success", "failure", "continue", "abort"],
                }
            )
        self._event("verdict", verdict=verdict, notes=notes, question=question)
        if verdict == "abort":
            self._operator_aborted = True
        elif verdict in {"success", "failure"}:
            self._verdict, self._verdict_notes = verdict, notes
        return ToolResult(
            data={"ok": True, "operator_verdict": verdict, "operator_notes": notes}
        )

    def _finish_with_verdict(self, finish: Any, **kwargs: Any) -> ToolResult:
        if (
            self._verdict is None
            and not self._operator_aborted
            and not self._operator_unavailable
        ):
            return ToolResult(
                data={
                    "error": "finish refused",
                    "reason": "Call request_operator_verdict first; only the operator can judge the physical result.",
                }
            )
        # The planner's own status never overrides the operator's verdict.
        kwargs["status"] = "success" if self.solved() else "failure"
        result = finish(**kwargs)
        result.data.update(
            operator_verdict=self._verdict,
            operator_notes=self._verdict_notes,
            operator_aborted=self._operator_aborted,
        )
        self._event("finish", status=kwargs["status"])
        return result
