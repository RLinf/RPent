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

"""Dual-Franka toolkit integrated with RPent's centralized environment state."""

from __future__ import annotations

import sys
from functools import partial
from pathlib import Path
from typing import TYPE_CHECKING, Any

from robots.dual_franka import perception as dual_franka_perception
from robots.dual_franka import tools as dual_franka_tools
from robots.franka import tools as franka_tools
from robots.franka.toolkit import FrankaToolkit
from rpent.dashboard.events import DashboardEventSink
from rpent.tools.toolkit import readonly

if TYPE_CHECKING:
    from rpent.memory.manager import MemoryManager


_EXPLORATION_ONLY_TOOLS = {"request_scene_reset"}


class DualFrankaToolkit(FrankaToolkit):
    """Common RPent tools plus safe dual-Franka planner primitives."""

    _tools_module = dual_franka_tools
    _primitives_cls = dual_franka_tools.DualFrankaPrimitives

    def __init__(
        self,
        *,
        primitives_kwargs: dict[str, Any],
        dashboard_events: DashboardEventSink,
        memory: MemoryManager,
        mode: str = "evaluation",
        attempts_per_session: int = 0,
        state_output_dir: Path | str | None = None,
    ) -> None:
        if mode not in {"evaluation", "exploration"}:
            raise ValueError(f"unsupported dual-Franka toolkit mode: {mode!r}")
        self._mode = mode
        self._attempt: int = 1
        self._attempts_per_session: int = max(0, int(attempts_per_session))
        self._session_attempt: int = 1
        self._operator_verdict: str | None = None
        self._operator_notes: str = ""
        super().__init__(
            primitives_kwargs=primitives_kwargs,
            dashboard_events=dashboard_events,
            memory=memory,
            state_output_dir=state_output_dir,
        )

    def _register_tools(self) -> None:
        state_handlers = {
            "view_env_state": partial(
                dual_franka_tools.view_env_state, state=self._state
            ),
            "view_camera_meta": partial(
                franka_tools.view_camera_meta,
                state=self._state,
            ),
            "back_project": partial(
                dual_franka_perception.back_project,
                state=self._state,
            ),
            "segment": partial(
                dual_franka_perception.segment,
                state=self._state,
                sam3_client=self._primitives._sam3_client,
            ),
            "request_scene_reset": self._request_scene_reset,
            "request_operator_verdict": self._request_operator_verdict,
        }
        for spec in self._tools_module.TOOLS_SPEC:
            name = spec["name"]
            if name in _EXPLORATION_ONLY_TOOLS and self._mode != "exploration":
                continue
            handler = state_handlers.get(name) or getattr(self._primitives, name)
            self.add_tool(name, spec, handler)
        finish_spec, finish_handler = self._tools["finish"]
        self.add_tool(
            "finish",
            finish_spec,
            partial(self._guarded_finish, finish_handler),
        )

    def _read_operator_line(self, prompt: str) -> str | None:
        if sys.stdin is None or not sys.stdin.isatty():
            return None
        try:
            return input(prompt).strip()
        except EOFError:
            return None

    def _request_scene_reset(
        self,
        reason: str,
        expected_scene_state: str = "",
    ) -> dict[str, Any]:
        """Pause until the operator makes reset safe, then reset robot posture."""
        budget = self._attempts_per_session
        if budget and self._session_attempt >= budget:
            return {
                "error": "scene reset refused",
                "reason": (
                    f"This session's attempt budget is spent ({budget} attempts). "
                    "Archive this attempt, update wip notes, and finish so the "
                    "next session can continue."
                ),
                "attempt": self._attempt,
                "attempts_per_session": budget,
            }

        print("\n[dual_franka exploration] scene reset requested")
        print(f"reason: {reason}")
        if expected_scene_state:
            print(f"expected scene: {expected_scene_state}")
        print(
            "Remove/secure any held objects and restore the real tabletop scene, "
            "then type 'done' to let the robot reset its own posture."
        )
        response = self._read_operator_line(
            "operator scene-safe confirmation [done/abort]: "
        )
        if response is None:
            return {
                "error": "operator confirmation unavailable",
                "reason": (
                    "request_scene_reset requires a runner terminal so the "
                    "human operator can confirm the physical scene reset."
                ),
                "attempt": self._attempt,
            }
        normalized = response.lower()
        if normalized not in {"done", "d", "yes", "y"}:
            return {
                "error": "scene reset aborted",
                "operator_response": response,
                "attempt": self._attempt,
            }

        self.raise_if_cancelled()
        robot_reset = self._primitives.reset()
        self._attempt += 1
        self._session_attempt += 1
        self._operator_verdict = None
        self._operator_notes = ""
        return {
            "ok": True,
            "action": "request_scene_reset",
            "attempt": self._attempt,
            "session_attempt": self._session_attempt,
            "attempts_per_session": budget,
            "robot_reset": robot_reset,
            "notice": (
                "Operator confirmed the physical scene was safe/restored, then "
                "the robot reset its own posture. Re-run perception before any "
                "motion; the tabletop scene was restored by the operator, not "
                "by a simulator-style environment reset."
            ),
        }

    @readonly
    def _request_operator_verdict(
        self,
        question: str = "Does the current real-robot scene satisfy the task?",
    ) -> dict[str, Any]:
        """Collect terminal feedback from the real-robot operator."""
        print("\n[dual_franka exploration] operator verdict requested")
        print(question)
        print("Type one of: success, failure, continue. Optional notes may follow.")
        response = self._read_operator_line("operator verdict: ")
        if response is None:
            return {
                "error": "operator verdict unavailable",
                "reason": (
                    "request_operator_verdict requires a runner terminal so "
                    "the human operator can judge the physical task state."
                ),
                "attempt": self._attempt,
            }
        parts = response.strip().split(maxsplit=1)
        verdict = parts[0].lower() if parts else ""
        notes = parts[1] if len(parts) > 1 else ""
        if verdict not in {"success", "failure", "continue"}:
            return {
                "error": "invalid operator verdict",
                "operator_response": response,
                "expected": ["success", "failure", "continue"],
                "attempt": self._attempt,
            }
        if verdict == "continue":
            return {
                "ok": True,
                "status": "continue",
                "operator_notes": notes,
                "attempt": self._attempt,
                "notice": "Operator requested more action; do not finish yet.",
            }
        self._operator_verdict = verdict
        self._operator_notes = notes
        return {
            "ok": True,
            "status": verdict,
            "operator_notes": notes,
            "attempt": self._attempt,
        }

    @readonly
    def _guarded_finish(self, inner: Any, **kwargs: Any) -> dict[str, Any]:
        """Require real-robot operator feedback before finishing a task."""
        if self._operator_verdict is None:
            return {
                "error": "finish refused",
                "reason": (
                    "Real-robot tasks require request_operator_verdict "
                    "before finish so the operator can judge the physical state."
                ),
            }
        budget = self._attempts_per_session
        if (
            self._mode == "exploration"
            and budget
            and self._operator_verdict != "success"
            and self._session_attempt < budget
        ):
            remaining = budget - self._session_attempt
            return {
                "error": "finish refused",
                "reason": (
                    f"The operator marked this attempt as {self._operator_verdict!r}, "
                    f"and this session still has {remaining} of {budget} attempts "
                    "left. Archive the attempt, call request_scene_reset, and try "
                    "a different approach."
                ),
            }
        result = inner(**kwargs)
        if isinstance(result, dict):
            result.setdefault("operator_verdict", self._operator_verdict)
            if self._operator_notes:
                result.setdefault("operator_notes", self._operator_notes)
        return result

    def solved(self) -> bool:
        """Return whether the real-robot operator accepted task success."""
        return self._operator_verdict == "success"
