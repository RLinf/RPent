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

import json
import sys
import threading
import time
from collections.abc import Callable
from functools import partial
from pathlib import Path
from typing import TYPE_CHECKING, Any

from robots.dual_franka import perception as dual_franka_perception
from robots.dual_franka import tools as dual_franka_tools
from robots.franka import tools as franka_tools
from robots.franka.toolkit import FrankaToolkit
from rpent.dashboard.events import DashboardEventSink
from rpent.tools.toolkit import ToolCancelled, ToolResult, readonly
from rpent.utils.logging import get_output_dir

if TYPE_CHECKING:
    from rpent.memory import MemoryManager

OperatorReader = Callable[[str, Callable[[], None]], str | None]

_EXPLORATION_ONLY_TOOLS = {"request_scene_reset"}

_MOTION_TOOLS = {
    "move_delta",
    "rotate_delta",
    "open_gripper",
    "close_gripper",
    "recover_joint_posture",
    "vla_right_grasp",
    "vla_handoff",
    "vla_left_place",
}


class DualFrankaToolkit(FrankaToolkit):
    """Dual-arm tools and the attended exploration lifecycle from PR #176.

    Hardware observations keep their original format. Operator decisions and
    attempt boundaries are additional artifacts, not simulator termination flags.
    """

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
        operator_input: OperatorReader | None = None,
    ) -> None:
        if mode not in {"evaluation", "exploration"}:
            raise ValueError(f"unsupported dual-Franka mode: {mode!r}")
        if attempts_per_session < 0:
            raise ValueError("attempts_per_session must be nonnegative")
        self._mode = mode
        self._attempt = 0  # first attempt begins only after scene confirmation
        self._budget = attempts_per_session
        self._attempts_per_session = attempts_per_session
        self._session_attempt = 1
        self._scene_ready = mode != "exploration"
        self._operator_input = operator_input
        self._operator_verdict = None
        self._operator_notes = ""
        self._operator_aborted = False
        self._verdict_step = None
        self._attempt_start_step = -1
        self._events = []
        self._direct_verdict_event = threading.Event()
        self._direct_verdict: str | None = None
        super().__init__(
            primitives_kwargs=primitives_kwargs,
            dashboard_events=dashboard_events,
            memory=memory,
            state_output_dir=state_output_dir,
        )

    @property
    def direct_verdict_requested(self) -> bool:
        """Whether operator control has sealed the active session."""
        return self._direct_verdict_event.is_set()

    def request_direct_verdict(self, verdict: str) -> bool:
        """Seal this attempt immediately; cancel in-flight work at its next boundary."""
        if verdict not in {"success", "failure", "abort"}:
            raise ValueError("verdict must be success, failure or abort")
        with self._operation_lock:
            if self._direct_verdict_event.is_set():
                return self._direct_verdict == verdict
            if self._mode != "exploration" or (
                verdict == "success"
                and (not self._scene_ready or self._operator_aborted)
            ):
                return False
            self._direct_verdict = verdict
            self._direct_verdict_event.set()
            if self._active_operation is not None:
                self._active_operation.cancel_event.set()
        return True

    def raise_if_cancelled(self) -> None:
        if self._direct_verdict_event.is_set():
            raise ToolCancelled(
                "operator submitted a terminal verdict; stopping exploration"
            )
        super().raise_if_cancelled()

    def execute_tool(self, name: str, input_dict: dict[str, Any]) -> ToolResult:
        if self._direct_verdict_event.is_set():
            return ToolResult(
                name=name,
                result={
                    "error": "operator submitted a terminal verdict; exploration is closing",
                    "motion_refused": True,
                },
            )
        return super().execute_tool(name, input_dict)

    def finalize_direct_verdict(self) -> dict[str, Any]:
        """Record fresh operator evidence after all active work has stopped."""
        self.cancel_active_and_wait()
        if not self._direct_verdict_event.is_set():
            raise RuntimeError("no direct verdict request")
        if self._direct_verdict == "abort":
            self._operator_aborted = True
            self._scene_ready = False
            self._clear_verdict()
            result = {
                "status": "failure",
                "operator_verdict": "abort",
                "operator_finished": True,
                "operator_aborted": True,
                "verdict_source": "interactive_command",
            }
            self._event("verdict", verdict="abort", source="interactive_command")
            self._event("finish", **result)
            return result
        self.get_env_state(
            command={"action": "observe_for_verdict"}, result={}, elapsed_s=0.0
        )
        self._validate_observation()
        record = self.state.latest_record()
        self._publish_step(record)
        self._scene_ready = True
        self._operator_verdict = self._direct_verdict
        self._operator_notes = (
            f"Operator entered /{self._direct_verdict} in the interactive terminal."
        )
        self._verdict_step = record.step_idx
        self._event(
            "verdict",
            verdict=self._direct_verdict,
            source="interactive_command",
            notes=self._operator_notes,
        )
        result = {
            "status": self._direct_verdict,
            "operator_verdict": self._direct_verdict,
            "operator_finished": True,
            "verdict_source": "interactive_command",
            "operator_aborted": False,
        }
        self._event("finish", **result)
        return result

    @readonly
    def _describe_exploration_setup(self, inner):
        result = inner()
        result["phase"] = "exploration"
        result["reset_policy"] = (
            "No automatic reset was performed by the client. Before motion in "
            "each session call request_scene_reset and wait for operator confirmation."
        )
        result["scene_ready"] = self._scene_ready
        return result

    def _clear_verdict(self) -> None:
        self._operator_verdict = None
        self._operator_notes = ""
        self._verdict_step = None

    def _guard_motion(self, inner, **kwargs):
        if (
            self._direct_verdict_event.is_set()
            or not self._scene_ready
            or self._operator_aborted
        ):
            return {
                "error": "motion refused; request_scene_reset and obtain operator confirmation first",
                "motion_refused": True,
            }
        self._clear_verdict()
        return inner(**kwargs)

    @readonly
    def _current_perception(self, inner, **kwargs):
        step = kwargs.get("step")
        if not self._scene_ready or (
            step is not None and step != -1 and step < self._attempt_start_step
        ):
            return {
                "error": "localization refused; use fresh observations after confirmed scene reset"
            }
        return inner(**kwargs)

    def _ask_operator(self, prompt: str, *, kind: str | None = None) -> str | None:
        self.raise_if_cancelled()
        if self._operator_input is None:
            response = self._read_operator_line(prompt)
            self.raise_if_cancelled()
            return response
        request = getattr(self._operator_input, "request", None)
        response = (
            request(prompt, self.raise_if_cancelled, kind=kind)
            if callable(request)
            else self._operator_input(prompt, self.raise_if_cancelled)
        )
        self.raise_if_cancelled()
        return response

    def _event(self, kind, **fields):
        event = {
            "kind": kind,
            "attempt": self._attempt,
            "step": self.state.latest_step,
            "timestamp": time.time(),
            **fields,
        }
        self._events.append(event)
        self.state.save("operator_events.json", self._events, step=None)
        return event

    def _request_scene_reset(
        self, reason: str, expected_scene_state: str = ""
    ) -> dict[str, Any]:
        if self._operator_aborted:
            return {"error": "operator aborted this run; finish without further motion"}
        if self._budget and self._attempt >= self._budget:
            return {
                "error": "scene reset refused; attempt budget spent",
                "attempt": self._attempt,
            }
        self._clear_verdict()
        self._scene_ready = False
        self._event(
            "reset_requested", reason=reason, expected_scene_state=expected_scene_state
        )
        response = self._ask_operator(
            f"Scene reset requested: {reason}\nExpected scene: {expected_scene_state}\n"
            "Remove/secure held objects and restore the tabletop. The robot will then reset "
            "its posture. Reply done to confirm, or abort to stop this run.",
            kind="reset",
        )
        self._event("reset_response", response=response)
        if response is None or response.strip().lower() != "done":
            self._operator_aborted = (
                response is None or response.strip().lower() == "abort"
            )
            return {
                "error": "scene reset not confirmed",
                "operator_aborted": self._operator_aborted,
            }
        # Never count a failed reset or a failed post-reset observation as a new attempt.
        result = self._primitives.reset()
        if (
            not isinstance(result, dict)
            or result.get("ok") is not True
            or result.get("error")
        ):
            self._event("reset_failed", result=result)
            return {"error": "robot reset failed", "robot_reset": result}
        return {
            "ok": True,
            "robot_reset": result,
            "scene_reset_confirmed": True,
            "notice": "Scene restored by operator; robot posture reset. Re-localize from the new images.",
        }

    @readonly
    def _request_operator_verdict(
        self, question="Does the current scene satisfy the task success criteria?"
    ):
        self._clear_verdict()
        if (
            self._direct_verdict_event.is_set()
            or not self._scene_ready
            or self._operator_aborted
        ):
            return {"error": "verdict refused; no active confirmed attempt"}
        # Save the evidence being judged using the existing camera/state logger.
        self.get_env_state(
            command={"action": "observe_for_verdict"}, result={}, elapsed_s=0.0
        )
        self._validate_observation()
        record = self.state.latest_record()
        self._publish_step(record)
        response = self._ask_operator(
            f"{question}\nAttempt {self._attempt}, observation step {record.step_idx}. "
            "Reply success, failure, continue, or abort; optional notes may follow.",
            kind="verdict",
        )
        parts = (response or "").strip().split(maxsplit=1)
        verdict = parts[0].lower() if parts else "unavailable"
        notes = parts[1] if len(parts) > 1 else ""
        event = self._event("verdict", verdict=verdict, notes=notes, question=question)
        if verdict == "abort" or response is None:
            self._operator_aborted = True
            self._scene_ready = False
        elif verdict in {"success", "failure"}:
            self._operator_verdict = verdict
            self._operator_notes = notes
            self._verdict_step = record.step_idx
        elif verdict != "continue":
            return {"error": "invalid operator verdict", "evidence": event}
        return {
            "ok": True,
            "status": verdict,
            "operator_notes": notes,
            "attempt": self._attempt,
            "evidence_step": record.step_idx,
            "operator_aborted": self._operator_aborted,
        }

    @readonly
    def _guarded_finish(self, inner, **kwargs):
        if not self._operator_aborted:
            if self._operator_verdict is None:
                return {"error": "finish refused; request_operator_verdict first"}
            if not self.solved() and self._budget and self._attempt < self._budget:
                return {
                    "error": "finish refused; archive this attempt and request_scene_reset",
                    "attempt": self._attempt,
                }
        # Agent-authored finish status must never override the operator verdict.
        kwargs["status"] = "success" if self.solved() else "failure"
        result = inner(**kwargs)
        result.update(
            operator_verdict=self._operator_verdict,
            operator_notes=self._operator_notes,
            operator_aborted=self._operator_aborted,
        )
        self._event("finish", **result)
        return result

    def get_env_state(self, *, command, result, elapsed_s):
        if self._mode != "exploration":
            return super().get_env_state(
                command=command, result=result, elapsed_s=elapsed_s
            )
        try:
            output = super().get_env_state(
                command=command, result=result, elapsed_s=elapsed_s
            )
            if command["action"] == "request_scene_reset" and result.get(
                "scene_reset_confirmed"
            ):
                self._validate_observation()
                self._attempt += 1
                self._scene_ready = True
                self._attempt_start_step = self.state.latest_step
                self._event("reset_completed")
            status = {
                "attempt": self._attempt,
                "attempts_per_session": self._budget,
                "scene_ready": self._scene_ready,
                "operator_verdict": self._operator_verdict,
                "attempt_start_step": self._attempt_start_step,
            }
            self.state.save("exploration.json", status)
            output["exploration"] = status
            # Keep the original record layout, and expose lifecycle errors to the planner.
            if result.get("error"):
                output["error"] = result["error"]
            return output
        except Exception:
            self._scene_ready = False
            self._clear_verdict()
            raise

    def _validate_observation(self) -> None:
        record = self.state.latest_record()
        meta = (
            self.state.load("camera_meta.json")
            if self.state.exists("camera_meta.json")
            else {}
        )
        camera_map = meta.get("observation_camera_map", {})
        names = {
            dual_franka_tools._camera_alias_from_key(key) for key in camera_map.values()
        }
        names.discard(None)
        names.update(
            dual_franka_tools._agent_observation_policy(meta)["inline_cameras"]
        )
        required = {f"{name}.png" for name in names}
        if not required.issubset(record.artifacts) or not all(
            record.state.get(arm) for arm in ("left_arm", "right_arm")
        ):
            self._scene_ready = False
            self._clear_verdict()
            raise RuntimeError("incomplete post-action robot/camera observation")

    def solved(self) -> bool:
        if self._mode != "exploration":
            return self._operator_verdict == "success"
        return (
            self._scene_ready
            and not self._operator_aborted
            and self._operator_verdict == "success"
        )

    def write_recipe(self, recipe_tag: str) -> str | None:
        if self._mode != "exploration" or not self.solved():
            return None
        records = [
            r
            for r in self.state.records()
            if self._attempt_start_step < r.step_idx <= self._verdict_step
            and (r.command or {}).get("action") in _MOTION_TOOLS
        ]
        # Keep all issued motion commands from the winning attempt, including
        # unsuccessful corrections: omitting them would misrepresent the trace.
        commands = [
            r.command for r in records if not (r.result or {}).get("motion_refused")
        ]
        root = Path(get_output_dir())
        recipe = root / f"{recipe_tag}_recipe.jsonl"
        recipe.write_text("".join(json.dumps(c) + "\n" for c in commands))
        audit_path = root / f"{recipe_tag}.json"
        agent_audit = None
        if audit_path.exists():
            try:
                agent_audit = json.loads(audit_path.read_text())
            except (ValueError, OSError):
                agent_audit = {"unparsed_text": audit_path.read_text()}
        audit = {
            "robot": "dual_franka",
            "cell": recipe_tag,
            "success": True,
            "success_source": "operator",
            "operator_notes": self._operator_notes,
            "attempt": self._attempt,
            "evidence_step": self._verdict_step,
            "attempt_start_step": self._attempt_start_step,
            "state_trace": str(
                self.state.artifact_path("operator_events.json", step=None).parent
                / "states.json"
            ),
            "command_sequence": commands,
            "agent_audit": agent_audit,
        }
        audit_path.write_text(json.dumps(audit, indent=2) + "\n")
        return str(recipe)

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
                sam3_client=getattr(self._primitives, "_sam3_client", None),
            ),
            "request_scene_reset": self._request_scene_reset,
            "request_operator_verdict": self._request_operator_verdict
            if self._mode == "exploration"
            else self._evaluation_verdict,
        }
        for spec in self._tools_module.TOOLS_SPEC:
            name = spec["name"]
            if name in _EXPLORATION_ONLY_TOOLS and self._mode != "exploration":
                continue
            handler = state_handlers.get(name) or getattr(self._primitives, name, None)
            if handler is None:
                continue
            if self._mode == "exploration":
                if name in _MOTION_TOOLS:
                    handler = partial(self._guard_motion, handler)
                elif name in {"back_project", "segment"}:
                    handler = partial(self._current_perception, handler)
                elif name == "describe_dual_franka_setup":
                    handler = partial(self._describe_exploration_setup, handler)
            self.add_tool(name, spec, handler)
        if self._mode != "exploration":
            finish_spec, finish_handler = self._tools["finish"]
            self.add_tool(
                "finish", finish_spec, partial(self._evaluation_finish, finish_handler)
            )
        if self._mode == "exploration":
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

    @readonly
    def _evaluation_verdict(
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
    def _evaluation_finish(self, inner: Any, **kwargs: Any) -> dict[str, Any]:
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
