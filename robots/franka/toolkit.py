"""Franka toolkit: common tools plus conservative Cartesian primitives."""
from __future__ import annotations

from functools import partial
from typing import Any

from robots.franka import tools as franka_tools
from rpent.tools.state import EnvState
from rpent.tools.toolkit import Toolkit
from rpent.utils.logging import get_output_dir


class FrankaToolkit(Toolkit):
    """Toolkit for the standalone Franka environment."""

    # view_driver_state image slots: primary scene -> _image_bytes, wrist -> _image_cam_bytes.
    _VIEW_IMAGE_SLOTS = {
        "_image_bytes": "scene.png",
        "_image_cam_bytes": "wrist.png",
    }

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
        dashboard: Any = None,
    ) -> None:
        # EnvState owns the trace + counter for this run (explicit output_dir,
        # no process-global). The runner will own its lifecycle in a later cut;
        # for now the toolkit constructs it from get_output_dir().
        state = EnvState(get_output_dir())
        super().__init__(dashboard=dashboard, state=state)
        self.init_driver_clean(env=env)
        self._register_tools()

    def _register_tools(self) -> None:
        spec = self._SPECS
        # view_driver_state / back_project read trace state -> bind to EnvState.
        self.add_tool(
            "view_driver_state",
            spec["view_driver_state"],
            partial(self._state.view, image_slots=self._VIEW_IMAGE_SLOTS),
            captures_state=False,
        )
        self.add_tool(
            "back_project",
            spec["back_project"],
            partial(franka_tools.back_project, state=self._state),
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
        franka_tools.dump_state(driver, self._state, log=None)
        self._driver = driver

    def close(self) -> None:
        return None
