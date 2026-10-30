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

"""RoboDojo tool registration and observation capture."""

from __future__ import annotations

from functools import partial
from pathlib import Path
from typing import TYPE_CHECKING, Any

from robots.robodojo.env_client import ExecutionSyncError
from robots.robodojo.primitives import RoboDojoPrimitives
from rpent.dashboard.events import DashboardEventSink
from rpent.session import EnvState
from rpent.tools import Toolkit, ToolResult, iter_tools, tool
from rpent.utils.logging import get_logger, get_output_dir

logger = get_logger(__name__)

OBSERVATION_TOOLS = frozenset(
    {"view_env_state", "segment", "back_project", "sample_world_xyz", "query_world_map"}
)
MOTION_TOOLS = frozenset(
    {
        "move_to",
        "rotate_wrist",
        "set_gripper",
        "release",
    }
)
POLICY_TOOLS = {"vpp2": "vpp2_act", "awomo05": "awomo05_act"}

if TYPE_CHECKING:
    from rpent.memory.manager import MemoryManager


class RoboDojoToolkit(Toolkit):
    """Toolkit for the RoboDojo (Isaac Sim) environment.

    Legacy policy code and composites are not exposed. ``eval_fair`` requires a mode-validated
    environment, removes file tools, and disables development perception
    recording. Scoring and safety diagnostics are never planner tools.
    """

    def __init__(
        self,
        *,
        primitives_kwargs: dict[str, Any],
        dashboard_events: DashboardEventSink,
        memory: MemoryManager,
        eval_fair: bool = False,
        state_output_dir: Path | str | None = None,
    ) -> None:
        from robots.robodojo import tools as robodojo_tools

        self.eval_fair = eval_fair
        if eval_fair and primitives_kwargs.get("vla_backend") == "awomo05":
            raise ValueError("Stateful policy execution is not supported in eval-fair")
        if eval_fair and not getattr(primitives_kwargs.get("env"), "eval_fair", False):
            raise ValueError("eval-fair requires a mode-validated environment client")
        state = EnvState(Path(state_output_dir or get_output_dir()))
        super().__init__(
            dashboard_events=dashboard_events,
            state=state,
            memory=memory,
        )
        self.init_primitives(primitives_kwargs=primitives_kwargs)
        try:
            self._register_robodojo_tools(primitives_kwargs.get("vla_backend", "vpp2"))
            self._publish_step(
                robodojo_tools.dump_state(self._primitives, self._state, log=None)
            )
        except Exception:
            try:
                self._primitives.close()
            except Exception:
                logger.warning(
                    "Failed to release primitives after startup error", exc_info=True
                )
            raise

    def flash_observation(self) -> dict:
        """Refresh the public observation used by live Flash grounding."""
        from robots.robodojo.access import public_observation

        if not self.eval_fair:
            raise RuntimeError("Flash requires eval-fair toolkit")
        obs = public_observation(self._primitives.env.get_obs())
        self._primitives._last_obs = obs
        return obs

    def init_primitives(self, *, primitives_kwargs: dict[str, Any]) -> None:
        """Start a new trace from the runtime-owned environment."""
        self._pending_perception = []
        self._state.reset()
        self._primitives = RoboDojoPrimitives(
            env=primitives_kwargs["env"],
            sam3_client=primitives_kwargs.get("sam3_client"),
            vla_client=primitives_kwargs.get("vla_client"),
            vla_backend=primitives_kwargs.get("vla_backend", "vpp2"),
            action_type=primitives_kwargs.get("action_type", "joint"),
            check_cancelled=self.raise_if_cancelled,
        )

    def solved(self) -> bool:
        """Read official success for the runner after planner execution."""
        if self.eval_fair:
            return False  # Plan completion is not an official task verdict.
        return self._primitives.env.is_success()

    def _register_robodojo_tools(self, vla_backend: str) -> None:
        from robots.robodojo import tools as robodojo_tools

        policy_tool = (
            POLICY_TOOLS[vla_backend]
            if self._primitives.vla_client is not None
            else None
        )
        for definition in iter_tools(robodojo_tools):
            if definition.name not in OBSERVATION_TOOLS:
                continue
            resources = {"state": self._state}
            if definition.name in {"sample_world_xyz", "query_world_map"}:
                resources = {"env_state": self._state}
            elif definition.name == "segment":
                resources["primitives"] = self._primitives
            bound = definition.with_handler(partial(definition, **resources))
            if not self.eval_fair and definition.name in {"segment", "back_project"}:
                bound = bound.with_handler(partial(self._call_robodojo_tool, bound))
            self.add_tool(bound)
        for definition in iter_tools(self._primitives):
            if (
                definition.name in POLICY_TOOLS.values()
                and definition.name != policy_tool
            ):
                continue
            if definition.name == "finish":
                definition = definition.with_handler(
                    partial(self._call_robodojo_tool, definition)
                )
            self.add_tool(definition, replace=definition.name == "finish")
        self.add_tool(self.render)
        if self.eval_fair:
            allowed = (
                OBSERVATION_TOOLS
                | MOTION_TOOLS
                | {"finish", "render"}
                | set(POLICY_TOOLS.values())
            )
            self._tools = {
                name: entry for name, entry in self._tools.items() if name in allowed
            }

    @tool
    def render(self) -> ToolResult:
        """Capture fresh RGB and robot state, plus available geometry for enabled tools."""
        self.raise_if_cancelled()
        return ToolResult(data={"success": True})

    def get_env_state(
        self,
        *,
        command: dict[str, Any],
        result: dict[str, Any],
        elapsed_s: float,
    ) -> ToolResult:
        from robots.robodojo import tools as robodojo_tools

        record = robodojo_tools.dump_state(
            self._primitives,
            self._state,
            log={"command": command, "result": result, "elapsed_s": elapsed_s},
            perception=self._pending_perception,
        )
        self._pending_perception = []
        return robodojo_tools.view_env_state(record.step_idx, state=self._state)

    def close(self) -> None:
        """Drain tools; the runtime owns the environment across planner sessions."""
        drained = False
        try:
            super().close()
            drained = True
        finally:
            try:
                self._primitives.close()
            except ExecutionSyncError:
                if drained:
                    raise
                logger.exception("Failed to close execution after tool cleanup failed")

    def cancel_active_and_wait(self) -> None:
        super().cancel_active_and_wait()
        self._primitives.env.pause_execution()

    def resume_calls(self) -> None:
        self._primitives.env.resume_execution()
        super().resume_calls()

    def _call_robodojo_tool(self, definition, **kwargs):
        name = definition.name
        result = definition(**kwargs)
        if name == "finish" and result.data.get("_finish"):
            try:
                self._primitives.env.end_execution()
            except ExecutionSyncError as exc:
                logger.exception("Failed to close execution after finish")
                result.data.setdefault("error", str(exc))
        if not self.eval_fair and name in {"segment", "back_project"}:
            import copy

            self._pending_perception.append(
                copy.deepcopy(
                    {
                        "action": name,
                        "arguments": kwargs,
                        "result": result.to_dict(),
                    }
                )
            )
        return result
