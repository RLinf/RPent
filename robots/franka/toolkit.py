"""Franka toolkit: common tools plus conservative Cartesian primitives."""
from __future__ import annotations

import time
from functools import partial
from typing import Any

from robots.franka import tools as franka_tools
from rpent.tools.state import EnvState
from rpent.tools.toolkit import Toolkit
from rpent.utils.logging import get_output_dir


class FrankaToolkit(Toolkit):
    """Toolkit for the standalone Franka environment."""

    # Streams wiped on init (LIBERO layout: scene=image/depth, wrist=image_wrist/depth_wrist).
    _WIPE_STREAMS = ("image", "image_wrist", "depth", "depth_wrist")
    # view_driver_state image slots: primary scene -> _image_bytes, wrist -> _image_cam_bytes.
    _VIEW_IMAGE_SLOTS = {"_image_bytes": "image", "_image_cam_bytes": "image_wrist"}

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
        # EnvState owns the trace + counter for this run (explicit output_dir,
        # no process-global). The runner will own its lifecycle in a later cut;
        # for now the toolkit constructs it from get_output_dir().
        state = EnvState(get_output_dir())
        super().__init__(dashboard=dashboard, state=state)
        self._video_path = video_path
        self.init_driver_clean(env=env)
        self._register_tools()

    def _register_tools(self) -> None:
        spec = self._SPECS
        # view_driver_state / back_project read trace state -> bind to EnvState.
        self.add_tool(
            "view_driver_state",
            spec["view_driver_state"],
            partial(self._state.view, image_slots=self._VIEW_IMAGE_SLOTS),
        )
        self.add_tool(
            "back_project",
            spec["back_project"],
            partial(franka_tools.back_project, state=self._state),
        )
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

        step_idx = self._state.next_step_idx
        franka_tools.dump_state(
            self._driver,
            self._state,
            step_idx=step_idx,
            log={"command": command, "result": result_dict, "elapsed_s": elapsed},
        )
        out = self._state.view(step_idx, image_slots=self._VIEW_IMAGE_SLOTS)
        out["agent_elapsed_s"] = elapsed
        return out

    def init_driver_clean(self, *, env: Any) -> None:
        self._state.reset(wipe_streams=self._WIPE_STREAMS)
        driver = franka_tools.FrankaPrimitives(env=env)
        driver.reset()
        franka_tools.dump_state(driver, self._state, step_idx=0, log=None)
        self._driver = driver

    def close(self) -> None:
        return None
