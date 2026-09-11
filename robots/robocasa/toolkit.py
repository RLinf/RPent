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

"""RoboCasa toolkit: common tools + RoboCasa primitives.

Inherits the common file/IO tools from :class:`Toolkit` and registers the
RoboCasa primitives (``move_to``, ``rldx_skill``, ``release``, ...) on top.
"""

from __future__ import annotations

from functools import partial
from pathlib import Path
from typing import TYPE_CHECKING, Any

from robots.robocasa import tools as robocasa_tools
from rpent.dashboard.events import DashboardEventSink
from rpent.session import EnvState
from rpent.tools.exploration import (
    ExplorationLifecycle,
    ResetAfterSuccess,
    ResetRefusal,
    refusal_result,
)
from rpent.tools.toolkit import Toolkit, readonly
from rpent.utils.logging import get_logger, get_output_dir

if TYPE_CHECKING:
    from rpent.memory.manager import MemoryManager

logger = get_logger("robocasa_toolkit")


class RoboCasaToolkit(Toolkit):
    """Toolkit for the RoboCasa robot."""

    def __init__(
        self,
        *,
        primitives_kwargs: dict[str, Any],
        dashboard_events: DashboardEventSink,
        memory: MemoryManager,
        mode: str = "evaluation",
        attempts_per_session: int = 0,
        state_output_dir: Path | str | None = None,
        recipe_output_dir: Path | str | None = None,
    ) -> None:
        """Create a RoboCasa toolkit, wiring the primitives and tools."""
        self._exploration = ExplorationLifecycle(
            mode=mode,
            attempts_per_session=attempts_per_session,
            adapter_name="RoboCasa",
            reset_after_success=ResetAfterSuccess.REFUSE,
        )
        self._recipe_output_dir = Path(recipe_output_dir or get_output_dir())
        self._state_output_dir = Path(state_output_dir or get_output_dir())
        state = EnvState(self._state_output_dir)
        super().__init__(
            dashboard_events=dashboard_events,
            state=state,
            memory=memory,
        )
        self.init_primitives(primitives_kwargs=primitives_kwargs)
        self._register_robocasa_tools()

    # ---- registration: one explicit add_tool per RoboCasa tool ----
    def _register_robocasa_tools(self) -> None:
        # Stateless perception tools: bind a state= kwarg via partial.
        state_handlers = {
            "view_env_state": partial(robocasa_tools.view_env_state, state=self._state),
            "view_camera_meta": partial(
                robocasa_tools.view_camera_meta, state=self._state
            ),
            "back_project": partial(robocasa_tools.back_project, state=self._state),
            "back_project_batch": partial(
                robocasa_tools.back_project_batch, state=self._state
            ),
            "query_world_map": partial(
                robocasa_tools.query_world_map, state=self._state
            ),
        }
        for spec in robocasa_tools.TOOLS_SPEC:
            name = spec["name"]
            if name in state_handlers:
                handler = state_handlers[name]
            elif name == "finish":
                handler = robocasa_tools.finish
            else:
                handler = getattr(self._primitives, name, None)
                if handler is None:
                    continue  # spec without a backing primitive method
            self.add_tool(name, spec, handler)
        if self._exploration.is_exploration:
            for name in robocasa_tools._PRIMITIVE_ACTIONS - {"reset"}:
                spec, handler = self._tools[name]
                self.add_tool(name, spec, partial(self._exploration_action, handler))
            self.add_tool(
                "reset",
                {
                    "name": "reset",
                    "description": (
                        "Archive failure, then reinitialize with configured seed. "
                        "Full physical layout determinism requires simulator verification."
                    ),
                    "input_schema": {
                        "type": "object",
                        "properties": {"reason": {"type": "string"}},
                        "required": ["reason"],
                    },
                },
                self._reset_episode,
            )
            spec, _ = self._tools["finish"]
            self.add_tool("finish", spec, self._guarded_finish)

    def _exploration_action(self, handler, **kwargs):
        if self._exploration.reset_failed:
            return {"error": "Reset failed; reset successfully before further actions."}
        if self.solved():
            return {"error": "Task already solved; write audit and finish."}
        return handler(**kwargs)

    def _reset_episode(self, reason: str) -> dict[str, Any]:
        refusal = self._exploration.request_reset(solved=self.solved())
        if refusal is ResetRefusal.TASK_SOLVED:
            return refusal_result("reset", "Task already solved; finish.")
        if refusal is ResetRefusal.ATTEMPT_BUDGET_SPENT:
            return refusal_result(
                "reset", "Attempt budget spent; archive and finish for handoff."
            )
        latest = self._state.latest_record()
        result = self._exploration.perform_reset(
            self._primitives.reset_exploration,
            successful_boundary_after_step=(
                latest.step_idx if latest is not None else -1
            ),
        )
        return self._exploration.build_reset_result(result, reason=reason)

    @readonly
    def _guarded_finish(self, *, status: str, summary: str) -> dict[str, Any]:
        refusal = self._exploration.finish_refusal(solved=self.solved())
        if refusal is not None:
            return refusal_result(
                "finish", "Attempts remain; archive, reset and try another approach."
            )
        return robocasa_tools.finish(status=status, summary=summary)

    def get_env_state(
        self,
        *,
        command: dict[str, Any],
        result: dict[str, Any],
        elapsed_s: float,
    ) -> dict[str, Any]:
        frame_start = self._action_frame_cursor
        self._action_frame_cursor = self._primitives.recorded_frame_count()
        record = robocasa_tools.dump_state(
            self._primitives,
            self._state,
            log={"command": command, "result": result, "elapsed_s": elapsed_s},
        )
        self._exploration.observe_record(record)
        if self._dashboard_events.enabled:
            try:
                frames = self._primitives.frame_slice(frame_start)
                if frames:
                    candidate = f"action_{command['action']}.mp4"
                    self._state.save(
                        candidate,
                        frames,
                        step=record.step_idx,
                        fps=20,
                    )
            except Exception as e:
                logger.warning(
                    "failed to save action clip for step %s: %s",
                    record.step_idx,
                    e,
                )
        out = robocasa_tools.view_env_state(record.step_idx, state=self._state)
        out["agent_elapsed_s"] = elapsed_s
        if result.get("interrupted") or (
            self._exploration.is_exploration and result.get("error")
        ):
            out.update(result)
        return out

    def init_primitives(
        self,
        *,
        primitives_kwargs: dict[str, Any],
    ) -> None:
        """Wipe stale run artifacts, build the primitives, dump step 0."""
        self._state.reset()

        from robots.robocasa.primitives import RoboCasaPrimitives

        if self._exploration.is_exploration:
            primitives_kwargs = {
                **primitives_kwargs,
                "workdir": str(self._state_output_dir),
            }
        primitives = RoboCasaPrimitives(
            check_cancelled=self.raise_if_cancelled,
            **({"exploration": True} if self._exploration.is_exploration else {}),
            **primitives_kwargs,
        )
        if self._exploration.is_exploration:
            reset_result = primitives.reset_exploration()
            logger.warning(reset_result["notice"])
        else:
            primitives.reset()
        primitives.start_recording()
        self._action_frame_cursor = primitives.recorded_frame_count()
        record = robocasa_tools.dump_state(
            primitives,
            self._state,
            log=(
                {
                    "command": {"action": "reset"},
                    "result": reset_result,
                    "elapsed_s": 0.0,
                }
                if self._exploration.is_exploration
                else None
            ),
        )
        self._exploration.observe_record(record)
        try:
            self._state.save(
                "success_criteria.md",
                primitives.dump_success_criteria(),
                step=None,
            )
        except Exception as e:
            logger.warning("failed to save success_criteria.md: %s", e)
        self._primitives = primitives
        self._publish_step(record)

    def close(self) -> None:
        """Flush the agent-side video buffer through ``EnvState``."""
        try:
            frames = self._primitives.stop_recording()
            if frames:
                self._state.save("episode.mp4", frames, step=None, fps=20)
        except Exception as e:
            logger.warning("failed to save episode video: %s", e)

    def solved(self) -> bool:
        """Return the success value from the final recorded environment state."""
        lifecycle = getattr(self, "_exploration", None)
        if lifecycle is not None and lifecycle.reset_failed:
            return False
        record = self._state.latest_record()
        return bool(record is not None and record.extras.get("success", False))

    def write_recipe(self, recipe_tag: str) -> str:
        """Write the RoboCasa recipe JSONL from the dumped state trace."""
        if self._exploration.is_exploration:
            if not self.solved():
                return ""
            return robocasa_tools.write_recipe_from_states(
                self._state,
                recipe_tag,
                output_dir=self._recipe_output_dir,
                after_step=self._exploration.latest_successful_reset_step,
            )
        return robocasa_tools.write_recipe_from_states(self._state, recipe_tag)
