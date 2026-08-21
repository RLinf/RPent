"""LeRobot SO101 toolkit: common tools + SO101 primitives.

Inherits the common file/IO tools (including ``finish``) from :class:`Toolkit`
and registers the SO101-specific tools (``view_env_state``, ``back_project``,
driver readers, and the move primitives) on top.
"""
from __future__ import annotations

from functools import partial
from typing import Any

from robots.lerobot import tools as lerobot_tools
from rpent.dashboard.events import DashboardEventSink
from rpent.tools.state import EnvState
from rpent.tools.toolkit import Toolkit
from rpent.utils.logging import get_output_dir


class LerobotToolkit(Toolkit):
    """Toolkit for the LeRobot SO101 environment."""

    _VIEW_IMAGE_SLOTS = {
        "_image_bytes": "scene.png",
        "_image_cam_bytes": "arm.png",
    }
    # Per-env artifact names for the dashboard's live frame images.
    _FRAME_ARTIFACTS = {
        "camera": "scene.png",
        "wrist": "arm.png",
    }

    def __init__(
        self,
        *,
        env: Any,
        dashboard_events: DashboardEventSink,
    ) -> None:
        state = EnvState(get_output_dir())
        super().__init__(dashboard_events=dashboard_events, state=state)
        self.init_driver_clean(env=env)
        self._register_tools()

    # ------------------------------------------------------------------
    # Registration
    # ------------------------------------------------------------------
    def _register_tools(self) -> None:
        # State-backed readers need the run's EnvState bound in. Every other
        # spec binds to its primitive-driver method; @readonly decides capture.
        state_handlers = {
            "view_env_state": partial(
                lerobot_tools.view_env_state,
                state=self._state,
                image_slots=self._VIEW_IMAGE_SLOTS,
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

    def get_env_state(
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
        out = lerobot_tools.view_env_state(
            record.step_idx,
            state=self._state,
            image_slots=self._VIEW_IMAGE_SLOTS,
        )
        out["agent_elapsed_s"] = elapsed_s
        return out

    def init_driver_clean(self, *, env: Any) -> None:
        """Wipe stale run artifacts, build the primitive driver, dump step 0."""
        self._state.reset()
        driver = lerobot_tools.LerobotPrimitives(env=env)
        driver.reset()
        record = lerobot_tools.dump_state(driver, self._state, log=None)
        self._driver = driver
        self._publish_step(record)

    def close(self) -> None:
        """End-of-run cleanup hook. TODO: flush an episode video if desired."""
        return None
