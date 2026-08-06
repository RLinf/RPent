"""Franka toolkit: common tools plus conservative Cartesian primitives."""
from __future__ import annotations

from functools import partial
from typing import Any

from robots.franka import tools as franka_tools
from rpent.dashboard.events import DashboardEventSink
from rpent.tools.state import EnvState
from rpent.tools.toolkit import Toolkit
from rpent.utils.logging import get_output_dir


class FrankaToolkit(Toolkit):
    """Toolkit for the standalone Franka environment."""

    # view_env_state image slots: primary scene -> _image_bytes, wrist -> _image_cam_bytes.
    _VIEW_IMAGE_SLOTS = {
        "_image_bytes": "scene.png",
        "_image_cam_bytes": "wrist.png",
    }
    # Per-env artifact names for the dashboard's live frame images.
    _FRAME_ARTIFACTS = {
        "camera": "scene.png",
        "wrist": "wrist.png",
    }

    def __init__(
        self,
        *,
        env: Any,
        dashboard_events: DashboardEventSink,
    ) -> None:
        # EnvState owns the trace + counter for this run (explicit output_dir,
        # no process-global). The runner will own its lifecycle in a later cut;
        # for now the toolkit constructs it from get_output_dir().
        state = EnvState(get_output_dir())
        super().__init__(dashboard_events=dashboard_events, state=state)
        self.init_driver_clean(env=env)
        self._register_tools()

    def _register_tools(self) -> None:
        # Read-only tools whose handlers aren't driver methods (they need the
        # run's EnvState bound in). Every other spec binds to its primitive-
        # driver method; @updatestate on the method decides state capture.
        state_handlers = {
            "view_env_state": partial(
                self._state.view, image_slots=self._VIEW_IMAGE_SLOTS
            ),
            "back_project": partial(franka_tools.back_project, state=self._state),
        }
        for spec in franka_tools.TOOLS_SPEC:
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
        self._driver._refresh(increment_step=command["action"] != "observe")
        record = franka_tools.dump_state(
            self._driver,
            self._state,
            log={"command": command, "result": result, "elapsed_s": elapsed_s},
        )
        out = self._state.view(record.step_idx, image_slots=self._VIEW_IMAGE_SLOTS)
        out["agent_elapsed_s"] = elapsed_s
        return out

    def init_driver_clean(self, *, env: Any) -> None:
        self._state.reset()
        driver = franka_tools.FrankaPrimitives(env=env)
        driver.reset()
        record = franka_tools.dump_state(driver, self._state, log=None)
        self._driver = driver
        self._publish_step(record)

    def close(self) -> None:
        return None
