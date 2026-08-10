"""Franka toolkit integrated with RPent's centralized environment state."""

from __future__ import annotations

from functools import partial
from typing import Any

from robots.franka import tools as franka_tools
from rpent.dashboard.events import DashboardEventSink
from rpent.tools.state import EnvState
from rpent.tools.toolkit import Toolkit
from rpent.utils.logging import get_output_dir


class FrankaToolkit(Toolkit):
    """Common RPent tools plus safe single-Franka planner primitives."""

    def __init__(
        self,
        *,
        primitives_kwargs: dict[str, Any],
        dashboard_events: DashboardEventSink,
    ) -> None:
        state = EnvState(get_output_dir())
        super().__init__(dashboard_events=dashboard_events, state=state)
        self._primitives = franka_tools.FrankaPrimitives(
            check_cancelled=self.raise_if_cancelled,
            **primitives_kwargs,
        )
        self._register_franka_tools()
        self._state.reset()
        self._primitives.reset()
        record = franka_tools.dump_state(
            self._primitives,
            self._state,
            command=None,
            result=None,
            elapsed_s=None,
        )
        self._publish_step(record)

    def _register_franka_tools(self) -> None:
        state_handlers = {
            "view_env_state": partial(franka_tools.view_env_state, state=self._state),
            "view_camera_meta": partial(
                franka_tools.view_camera_meta,
                state=self._state,
            ),
        }
        for spec in franka_tools.TOOLS_SPEC:
            name = spec["name"]
            handler = state_handlers.get(name) or getattr(self._primitives, name)
            self.add_tool(name, spec, handler)

    def get_env_state(
        self,
        *,
        command: dict[str, Any],
        result: dict[str, Any],
        elapsed_s: float,
    ) -> dict[str, Any]:
        record = franka_tools.dump_state(
            self._primitives,
            self._state,
            command=command,
            result=result,
            elapsed_s=elapsed_s,
        )
        output = franka_tools.view_env_state(record.step_idx, state=self._state)
        output["agent_elapsed_s"] = elapsed_s
        return output
