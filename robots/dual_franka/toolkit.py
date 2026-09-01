"""Dual-Franka toolkit integrated with RPent's centralized environment state."""

from __future__ import annotations

from functools import partial
from typing import Any

from robots.dual_franka import perception as dual_franka_perception
from robots.dual_franka import tools as dual_franka_tools
from robots.franka import tools as franka_tools
from robots.franka.runtime_config import set_calibration_path
from rpent.dashboard.events import DashboardEventSink
from rpent.tools.state import EnvState
from rpent.tools.toolkit import Toolkit
from rpent.utils.logging import get_output_dir


class DualFrankaToolkit(Toolkit):
    """Common RPent tools plus safe dual-Franka planner primitives."""

    def __init__(
        self,
        *,
        primitives_kwargs: dict[str, Any],
        dashboard_events: DashboardEventSink,
    ) -> None:
        state = EnvState(get_output_dir())
        super().__init__(dashboard_events=dashboard_events, state=state)
        calibration_path = primitives_kwargs.pop("calibration_path", None)
        if calibration_path is not None:
            set_calibration_path(calibration_path)
        self._primitives = dual_franka_tools.DualFrankaPrimitives(
            check_cancelled=self.raise_if_cancelled,
            **primitives_kwargs,
        )
        self._register_dual_franka_tools()
        self._state.reset()
        self._primitives.reset()
        record = dual_franka_tools.dump_state(
            self._primitives,
            self._state,
            command=None,
            result=None,
            elapsed_s=None,
        )
        self._publish_step(record)

    def _register_dual_franka_tools(self) -> None:
        state_handlers = {
            "view_env_state": partial(
                dual_franka_tools.view_env_state, state=self._state
            ),
            "view_camera_meta": partial(
                franka_tools.view_camera_meta,
                state=self._state,
            ),
            "back_project_base_pixel": partial(
                dual_franka_perception.back_project_base_pixel,
                state=self._state,
            ),
            "back_project_d455_pixel": partial(
                dual_franka_perception.back_project_d455_pixel,
                state=self._state,
            ),
        }
        for spec in dual_franka_tools.TOOLS_SPEC:
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
        record = dual_franka_tools.dump_state(
            self._primitives,
            self._state,
            command=command,
            result=result,
            elapsed_s=elapsed_s,
        )
        output = dual_franka_tools.view_env_state(record.step_idx, state=self._state)
        output["agent_elapsed_s"] = elapsed_s
        return output
