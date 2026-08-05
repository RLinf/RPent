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
    # Read-only tools backed by a live driver call (no state dump). These query
    # the robot/scene directly: forward kinematics + scene camera calibration.
    _DRIVER_READERS = (
        "get_ee_pose",
        "get_scene_camera_meta",
    )
    # Primitive tools move the robot; get_state refreshes and records state.
    _PRIMITIVE_TOOLS: tuple[str, ...] = (
        "move_to",
        "move_joints_delta",
    )

    # Tool schemas keyed by name, built once from the canonical ordered list.
    _SPECS = {spec["name"]: spec for spec in lerobot_tools.TOOLS_SPEC}

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
        spec = self._SPECS
        self.add_tool(
            "view_driver_state",
            spec["view_driver_state"],
            partial(self._state.view, image_slots=self._VIEW_IMAGE_SLOTS),
            captures_state=False,
        )
        self.add_tool(
            "back_project",
            spec["back_project"],
            partial(lerobot_tools.back_project, state=self._state),
            captures_state=False,
        )
        for name in self._DRIVER_READERS:
            self.add_tool(
                name,
                spec[name],
                self._make_driver_reader(name),
                captures_state=False,
            )
        for name in self._PRIMITIVE_TOOLS:
            self.add_tool(
                name,
                spec[name],
                getattr(self._driver, name),
                captures_state=True,
            )

    def _make_driver_reader(self, name: str):
        """Bind a read-only tool to ``self._driver.<name>`` (no state dump)."""
        def _reader(**kwargs) -> dict:
            result = getattr(self._driver, name)(**kwargs)
            return result if isinstance(result, dict) else {"value": result}
        return _reader

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
