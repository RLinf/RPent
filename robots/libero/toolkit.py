"""LIBERO toolkit: common tools + LIBERO primitives.

Inherits the common file/IO tools from :class:`Toolkit` and registers the
LIBERO primitives (``move_to``, ``pi0_pick``, ``release``, ...) on top.
"""
from __future__ import annotations

import shutil
import time
from functools import partial
from typing import Any

from robots.libero import tools as libero_tools
from rpent.tools.toolkit import Toolkit
from rpent.utils.logging import get_logger, get_output_dir


class LiberoToolkit(Toolkit):
    """Toolkit for the LIBERO environment."""

    # Tool schemas keyed by name (built once from the canonical ordered list
    # in libero_tools.TOOLS_SPEC) so each tool registers with its own spec.
    _SPECS = {spec["name"]: spec for spec in libero_tools.TOOLS_SPEC}

    def __init__(
        self,
        *,
        primitives_kwargs: dict[str, Any],
        video_path: str | None = None,
        dashboard: Any = None,
    ) -> None:
        super().__init__(dashboard=dashboard)
        self._next_step: int = 0
        self._video_path: str | None = video_path
        self.init_primitives_clean(primitives_kwargs=primitives_kwargs)
        self._register_libero_tools()

    # ------------------------------------------------------------------
    # Registration
    # ------------------------------------------------------------------
    def _register_libero_tools(self) -> None:
        specs = self._SPECS
        # Inspection tools do not advance environment state. Most are stateless
        # module functions; segment is bound to the primitives-owned SAM3 client.
        inspection_handlers = {
            "view_driver_state": libero_tools.view_driver_state,
            "view_camera_meta": libero_tools.view_camera_meta,
            "back_project": libero_tools.back_project,
            "segment": self._primitives.segment,
        }
        for name, handler in inspection_handlers.items():
            self.add_tool(name, specs[name], handler)
        # Primitive tools: each goes through _step, which looks up the
        # matching primitive method via getattr at call time.
        for name in libero_tools.PRIMITIVE_TOOL_NAMES:
            self.add_tool(name, specs[name], partial(self._step, name))

    def _step(self, name: str, **kwargs) -> dict:
        """Run ``self._primitives.<name>(**kwargs)``, dump the new step, and
        return the rendered state view + log.
        """
        command = {"action": name, **kwargs}
        t0 = time.time()
        start_frame = self._primitives.recorded_frame_count()
        result = getattr(self._primitives, name)(**kwargs)
        elapsed = round(time.time() - t0, 2)

        if isinstance(result, dict):
            result_dict = result
        else:
            result_dict = {"value": result}

        self._next_step += 1
        step_idx = self._next_step
        if self._dashboard is not None:
            video_dir = libero_tools.artifact_path(get_output_dir(), "action_videos")
            video_path = video_dir / f"step_{step_idx:02d}_{name}.mp4"
            try:
                self._primitives.save_frame_slice(start_frame, str(video_path), fps=20)
            except Exception as e:
                get_logger("libero_toolkit").warning(
                    f"failed to save action clip to {video_path}: {e}"
                )
        libero_tools.dump_state(
            self._primitives,
            str(get_output_dir()),
            step_idx=step_idx,
            log={"command": command, "result": result_dict, "elapsed_s": elapsed},
        )
        out = libero_tools.view_driver_state(step_idx)
        out["agent_elapsed_s"] = elapsed
        return out

    def init_primitives_clean(
        self,
        *,
        primitives_kwargs: dict[str, Any],
    ) -> None:
        """Wipe stale run artifacts, build the LiberoPrimitives, dump step 0."""
        out_dir = get_output_dir()
        out_dir.mkdir(parents=True, exist_ok=True)
        for sub in libero_tools.ARTIFACT_DIRECTORIES:
            target = out_dir / sub
            if target.exists():
                shutil.rmtree(target)
        for target in (
            libero_tools.artifact_path(out_dir, "states"),
            libero_tools.artifact_path(out_dir, "metadata", camera="agentview", resolution="low"),
            libero_tools.artifact_path(out_dir, "episode_video"),
        ):
            if target.exists():
                target.unlink()

        primitives = libero_tools.LiberoPrimitives(**primitives_kwargs)
        primitives.reset()
        primitives.start_recording()
        libero_tools.dump_state(primitives, str(out_dir), step_idx=0, log=None)
        if self._dashboard is not None:
            self._dashboard.on_tool_result("view_driver_state", libero_tools.view_driver_state(0))

        self._primitives = primitives

    def close(self) -> None:
        """Flush the agent-side video buffer to disk (end-of-run).
        """
        if self._video_path is None:
            return
        try:
            self._primitives.stop_recording_and_save(self._video_path)
        except Exception as e:
            # The runner is in the cleanup path; never let a video save
            # abort it.
            get_logger("libero_toolkit").warning(
                f"failed to save video to {self._video_path}: {e}"
            )

    def write_recipe(self, recipe_tag: str) -> str:
        """Write the LIBERO recipe JSONL from the dumped state trace."""
        return libero_tools.write_recipe_from_states(str(get_output_dir()), recipe_tag)
