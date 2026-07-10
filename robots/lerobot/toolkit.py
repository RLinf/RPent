"""LeRobot SO101 toolkit: common tools + SO101 primitives.

Inherits the common file/IO tools (including ``finish``) from :class:`Toolkit`
and registers the SO101-specific tools (``view_driver_state``, ``back_project``,
driver readers, and the move primitives) on top.
"""
from __future__ import annotations

import shutil
import time
from functools import partial
from typing import Any

from robots.lerobot import tools as lerobot_tools
from rpent.tools.toolkit import Toolkit
from rpent.utils.logging import get_output_dir


class LerobotToolkit(Toolkit):
    """Toolkit for the LeRobot SO101 environment."""

    # Stateless reader tools bound directly to module-level functions.
    _STATELESS_TOOLS = (
        "view_driver_state",
        "back_project",
    )
    # Read-only tools backed by a live driver call (no state dump). These query
    # the robot/scene directly: forward kinematics + scene camera calibration.
    _DRIVER_READERS = (
        "get_ee_pose",
        "get_scene_camera_meta",
    )
    # Primitive tools routed through ``_step`` (look up driver method by name).
    # Each moves the robot and re-renders state after running.
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
        video_path: str | None = None,
        dashboard: Any = None,
    ) -> None:
        super().__init__(dashboard=dashboard)
        self._next_step: int = 0
        self._video_path: str | None = video_path
        self.init_driver_clean(env=env, model=model)
        self._register_tools()

    # ------------------------------------------------------------------
    # Registration
    # ------------------------------------------------------------------
    def _register_tools(self) -> None:
        spec = self._SPECS
        for name in self._STATELESS_TOOLS:
            self.add_tool(name, spec[name], getattr(lerobot_tools, name))
        for name in self._DRIVER_READERS:
            self.add_tool(name, spec[name], self._make_driver_reader(name))
        for name in self._PRIMITIVE_TOOLS:
            self.add_tool(name, spec[name], partial(self._step, name))

    def _make_driver_reader(self, name: str):
        """Bind a read-only tool to ``self._driver.<name>`` (no state dump)."""
        def _reader(**kwargs) -> dict:
            result = getattr(self._driver, name)(**kwargs)
            return result if isinstance(result, dict) else {"value": result}
        return _reader

    def _step(self, name: str, **kwargs) -> dict:
        """Run ``self._driver.<name>(**kwargs)``, dump the new step, and
        return the rendered state view + log.
        """
        command = {"action": name, **kwargs}
        t0 = time.time()
        result = getattr(self._driver, name)(**kwargs)
        elapsed = round(time.time() - t0, 2)

        result_dict = result if isinstance(result, dict) else {"value": result}

        self._next_step += 1
        step_idx = self._next_step
        lerobot_tools.dump_state(
            self._driver,
            str(get_output_dir()),
            step_idx=step_idx,
            log={"command": command, "result": result_dict, "elapsed_s": elapsed},
        )
        out = lerobot_tools.view_driver_state(step_idx)
        out["agent_elapsed_s"] = elapsed
        return out

    def init_driver_clean(self, *, env: Any, model: Any | None = None) -> None:
        """Wipe stale run artifacts, build the primitive driver, dump step 0."""
        out_dir = get_output_dir()
        out_dir.mkdir(parents=True, exist_ok=True)
        images_dir = out_dir / "images"
        if images_dir.exists():
            shutil.rmtree(images_dir)
        states_file = out_dir / "states.json"
        if states_file.exists():
            states_file.unlink()

        driver = lerobot_tools.LerobotPrimitives(env=env, model=model)
        driver.reset()
        lerobot_tools.dump_state(driver, str(out_dir), step_idx=0, log=None)

        self._driver = driver

    def close(self) -> None:
        """End-of-run cleanup hook. TODO: flush an episode video if desired."""
        return None
