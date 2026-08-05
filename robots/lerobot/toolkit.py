"""LeRobot SO101 toolkit: common tools + SO101 primitives.

Inherits the common file/IO tools (including ``finish``) from :class:`Toolkit`
and registers the SO101-specific tools (``view_driver_state``, ``back_project``,
driver readers, and the move primitives) on top.
"""
from __future__ import annotations

from functools import partial
from typing import Any

from robots.lerobot import tools as lerobot_tools
from rpent.tools.state import EnvState
from rpent.tools.toolkit import Toolkit
from rpent.utils.logging import get_output_dir


class LerobotToolkit(Toolkit):
    """Toolkit for the LeRobot SO101 environment."""

    _VIEW_IMAGE_SLOTS = {
        "_image_bytes": "scene.png",
        "_image_cam_bytes": "arm.png",
    }

    def __init__(
        self,
        *,
        env: Any,
        model: Any | None = None,
        dashboard: Any = None,
    ) -> None:
        state = EnvState(get_output_dir())
        super().__init__(dashboard=dashboard, state=state)
        self.init_driver_clean(env=env, model=model)
        self._register_tools()

    # ------------------------------------------------------------------
    # Registration
    # ------------------------------------------------------------------
    def _register_tools(self) -> None:
        # Read-only tools whose handlers aren't driver methods (they need the
        # run's EnvState bound in). Every other spec binds to its primitive-
        # driver method; @updatestate on the method decides state capture.
        state_handlers = {
            "view_driver_state": partial(
                self._state.view, image_slots=self._VIEW_IMAGE_SLOTS
            ),
            "back_project": partial(lerobot_tools.back_project, state=self._state),
        }
        for spec in lerobot_tools.TOOLS_SPEC:
            name = spec["name"]
            if name in state_handlers:
                handler = state_handlers[name]
            else:
                handler = getattr(self._driver, name, None)
                if handler is None:
                    continue  # spec without a backing driver method
            self.add_tool(name, spec, handler)

    def get_state(
        self,
        *,
        command: dict[str, Any],
        result: dict[str, Any],
        elapsed_s: float,
    ) -> dict[str, Any]:
        self._driver._refresh()
        record = lerobot_tools.dump_state(
            self._driver,
            self._state,
            log={"command": command, "result": result, "elapsed_s": elapsed_s},
        )
        out = self._state.view(record.step_idx, image_slots=self._VIEW_IMAGE_SLOTS)
        out["agent_elapsed_s"] = elapsed_s
        return out

    def init_driver_clean(self, *, env: Any, model: Any | None = None) -> None:
        """Wipe stale run artifacts, build the primitive driver, dump step 0."""
        self._state.reset()
        driver = lerobot_tools.LerobotPrimitives(env=env, model=model)
        driver.reset()
        lerobot_tools.dump_state(driver, self._state, log=None)

        self._driver = driver

    def close(self) -> None:
        """End-of-run cleanup hook. TODO: flush an episode video if desired."""
        return None
