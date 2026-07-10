"""Franka toolkit: common tools plus conservative Cartesian primitives."""
from __future__ import annotations

import shutil
import time
from functools import partial
from typing import Any

from robots.franka import tools as franka_tools
from rpent.tools.toolkit import Toolkit
from rpent.utils.logging import get_output_dir


class FrankaToolkit(Toolkit):
    """Toolkit for the standalone Franka environment."""

    _STATELESS_TOOLS = (
        "view_driver_state",
        "back_project",
    )
    _DRIVER_READERS = (
        "get_ee_pose",
        "get_robot_spec",
        "get_camera_meta",
    )
    _PRIMITIVE_TOOLS = (
        "observe",
        "move_to",
        "move_delta",
        "rotate_wrist_yaw",
        "rotate_gripper",
        "open_gripper",
        "close_gripper",
    )

    _SPECS = {spec["name"]: spec for spec in franka_tools.TOOLS_SPEC}

    def __init__(
        self,
        *,
        env: Any,
        video_path: str | None = None,
        dashboard: Any = None,
    ) -> None:
        super().__init__(dashboard=dashboard)
        self._next_step = 0
        self._video_path = video_path
        self.init_driver_clean(env=env)
        self._register_tools()

    def _register_tools(self) -> None:
        spec = self._SPECS
        for name in self._STATELESS_TOOLS:
            self.add_tool(name, spec[name], getattr(franka_tools, name))
        for name in self._DRIVER_READERS:
            self.add_tool(name, spec[name], self._make_driver_reader(name))
        for name in self._PRIMITIVE_TOOLS:
            self.add_tool(name, spec[name], partial(self._step, name))

    def _make_driver_reader(self, name: str):
        def _reader(**kwargs) -> dict:
            result = getattr(self._driver, name)(**kwargs)
            return result if isinstance(result, dict) else {"value": result}

        return _reader

    def _step(self, name: str, **kwargs) -> dict:
        command = {"action": name, **kwargs}
        t0 = time.time()
        result = getattr(self._driver, name)(**kwargs)
        elapsed = round(time.time() - t0, 2)
        result_dict = result if isinstance(result, dict) else {"value": result}

        self._next_step += 1
        step_idx = self._next_step
        franka_tools.dump_state(
            self._driver,
            str(get_output_dir()),
            step_idx=step_idx,
            log={"command": command, "result": result_dict, "elapsed_s": elapsed},
        )
        out = franka_tools.view_driver_state(step_idx)
        out["agent_elapsed_s"] = elapsed
        return out

    def init_driver_clean(self, *, env: Any) -> None:
        out_dir = get_output_dir()
        out_dir.mkdir(parents=True, exist_ok=True)
        images_dir = out_dir / "images"
        if images_dir.exists():
            shutil.rmtree(images_dir)
        states_file = out_dir / "states.json"
        if states_file.exists():
            states_file.unlink()

        driver = franka_tools.FrankaPrimitives(env=env)
        driver.reset()
        franka_tools.dump_state(driver, str(out_dir), step_idx=0, log=None)
        self._driver = driver

    def close(self) -> None:
        return None
