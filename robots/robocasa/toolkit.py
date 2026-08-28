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
from typing import Any

from robots.robocasa import tools as robocasa_tools
from rpent.dashboard.events import DashboardEventSink
from rpent.tools.state import EnvState
from rpent.tools.toolkit import Toolkit
from rpent.utils.logging import get_logger, get_output_dir

logger = get_logger("robocasa_toolkit")


class RoboCasaToolkit(Toolkit):
    """Toolkit for the RoboCasa robot."""

    _FRAME_ARTIFACTS = {
        "camera": "agentview.png",
        "wrist": "wrist.png",
    }

    def __init__(
        self,
        *,
        primitives_kwargs: dict[str, Any],
        dashboard_events: DashboardEventSink,
    ) -> None:
        """Create a RoboCasa toolkit, wiring the primitives and tools."""
        state = EnvState(get_output_dir())
        super().__init__(dashboard_events=dashboard_events, state=state)
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
        if result.get("interrupted"):
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

        primitives = RoboCasaPrimitives(
            check_cancelled=self.raise_if_cancelled,
            **primitives_kwargs,
        )
        primitives.reset()
        primitives.start_recording()
        self._action_frame_cursor = primitives.recorded_frame_count()
        record = robocasa_tools.dump_state(primitives, self._state, log=None)
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

    def write_recipe(self, recipe_tag: str) -> str:
        """Write the RoboCasa recipe JSONL from the dumped state trace."""
        return robocasa_tools.write_recipe_from_states(self._state, recipe_tag)
