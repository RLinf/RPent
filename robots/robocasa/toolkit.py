"""RoboCasa toolkit: common tools + RoboCasa primitives.

Inherits the common file/IO tools from :class:`Toolkit` and registers the
RoboCasa primitives (``move_to``, ``rldx_skill``, ``release``, ...) on top.
"""
from __future__ import annotations

import shutil
import time
from functools import partial
from typing import Any

from robots.robocasa import tools as robocasa_tools
from rpent.dashboard.events import DashboardEventSink, ToolResultEvent
from rpent.tools.toolkit import ToolCancelled, Toolkit
from rpent.utils.logging import get_logger, get_output_dir

logger = get_logger("robocasa_toolkit")


class RoboCasaToolkit(Toolkit):
    """Toolkit for the RoboCasa environment."""

    _SPECS = {spec["name"]: spec for spec in robocasa_tools.TOOLS_SPEC}

    def __init__(
        self,
        *,
        primitives_kwargs: dict[str, Any],
        dashboard_events: DashboardEventSink,
        video_path: str | None = None,
    ) -> None:
        """Create a RoboCasa toolkit, wiring the primitives and tools."""
        super().__init__(dashboard_events=dashboard_events)
        self._next_step: int = 0
        self._video_path: str | None = video_path
        self.init_primitives(primitives_kwargs=primitives_kwargs)
        self._register_robocasa_tools()

    # ------------------------------------------------------------------
    # Registration — one explicit add_tool per RoboCasa tool.
    # ------------------------------------------------------------------
    def _register_robocasa_tools(self) -> None:
        spec = self._SPECS
        # Stateless perception tools: directly point at the robocasa_tools module functions.
        for name in (
            "view_driver_state",
            "view_camera_meta",
            "back_project",
            "back_project_batch",
            "query_world_map",
            "finish",
        ):
            self.add_tool(name, spec[name], getattr(robocasa_tools, name))
        # Primitive tools: each goes through _step, which dispatches to the
        # matching primitives method via getattr at call time.
        for name in (
            "move_to",
            "move_delta",
            "rotate_pitch",
            "set_gripper",
            "release",
            "scripted_grasp",
            "rldx_skill",
            "rldx_arm",
            "navigate_to",
            "move_base",
            "reset",
        ):
            self.add_tool(name, spec[name], partial(self._step, name))

    def _step(self, name: str, **kwargs) -> dict:
        """Run ``self._primitives.<name>(**kwargs)`` and return the rendered state view.

        Dispatches the primitive action, records elapsed time, dumps the
        new step, and returns the rendered state view.
        """
        command = {"action": name, **kwargs}
        t0 = time.time()
        start_frame = self._primitives.recorded_frame_count()
        try:
            result = getattr(self._primitives, name)(**kwargs)
            self.raise_if_cancelled()
        except ToolCancelled as exc:
            result = {
                "error": str(exc),
                "code": "tool_cancelled",
                "interrupted": True,
            }
        elapsed = round(time.time() - t0, 2)

        if isinstance(result, dict):
            result_dict = result
        else:
            result_dict = {"value": result}

        self._next_step += 1
        step_idx = self._next_step
        if self._dashboard_events.enabled:
            video_dir = get_output_dir() / "action_videos"
            video_path = video_dir / f"step_{step_idx:02d}_{name}.mp4"
            try:
                self._primitives.save_frame_slice(start_frame, str(video_path), fps=20)
            except Exception as e:
                logger.warning(
                    f"failed to save action clip to {video_path}: {e}"
                )
        robocasa_tools.dump_state(
            self._primitives,
            str(get_output_dir()),
            step_idx=step_idx,
            log={"command": command, "result": result_dict, "elapsed_s": elapsed},
        )
        out = robocasa_tools.view_driver_state(step_idx)
        out["agent_elapsed_s"] = elapsed
        if result_dict.get("interrupted"):
            out.update(result_dict)
        return out

    def init_primitives(
        self,
        *,
        primitives_kwargs: dict[str, Any],
    ) -> None:
        """Wipe stale run artifacts, build the primitives instance, dump step 0."""
        from robots.robocasa.primitives import (
            RoboCasaPrimitives,
        )

        out_dir = get_output_dir()
        if out_dir:
            out_dir.mkdir(parents=True, exist_ok=True)
            for sub in (
                "images",
                "images_cam",
                "depths",
                "world",
                "images_wrist",
                "depths_wrist",
                "world_wrist",
                "images_cam_hi",
                "world_hi",
                "world_nav",
            ):
                target = out_dir / sub
                if target.exists():
                    shutil.rmtree(target)
            for fname in ("states.json", "camera_meta.json"):
                target = out_dir / fname
                if target.exists():
                    target.unlink()

        primitives = RoboCasaPrimitives(
            check_cancelled=self.raise_if_cancelled,
            **primitives_kwargs,
        )
        primitives.reset()
        primitives.start_recording()
        robocasa_tools.dump_state(
            primitives,
            str(out_dir) if out_dir else "/tmp",
            step_idx=0,
            log=None,
        )
        self._dashboard_events.emit(
            ToolResultEvent(
                name="view_driver_state",
                result=robocasa_tools.view_driver_state(0),
            )
        )
        self._primitives = primitives

    def close(self) -> None:
        """Flush the agent-side video buffer to disk (end-of-run)."""
        if self._video_path is None:
            return
        try:
            self._primitives.stop_recording_and_save(self._video_path)
        except Exception as e:
            logger.warning(
                f"failed to save video to {self._video_path}: {e}"
            )

    def write_recipe(self, recipe_tag: str) -> str:
        """Write the RoboCasa recipe JSONL from the dumped state trace."""
        return robocasa_tools.write_recipe_from_states(
            str(get_output_dir()), recipe_tag
        )
