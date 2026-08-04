"""LIBERO toolkit: common tools + LIBERO primitives.

Inherits the common file/IO tools from :class:`Toolkit` and registers the
LIBERO primitives (``move_to``, ``pi0_pick``, ``release``, ...) on top.
"""
from __future__ import annotations

import time
from functools import partial
from typing import Any

from robots.libero import tools as libero_tools
from rpent.dashboard.events import DashboardEventSink, ToolResultEvent
from rpent.tools.state import EnvState
from rpent.tools.toolkit import ToolCancelled, Toolkit
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
        dashboard_events: DashboardEventSink,
    ) -> None:
        state = EnvState(get_output_dir())
        super().__init__(dashboard_events=dashboard_events, state=state)
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
            "view_driver_state": partial(
                libero_tools.view_driver_state, state=self._state
            ),
            "view_camera_meta": partial(
                libero_tools.view_camera_meta, state=self._state
            ),
            "back_project": partial(libero_tools.back_project, state=self._state),
            "segment": partial(self._primitives.segment, state=self._state),
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

        record = libero_tools.dump_state(
            self._primitives,
            self._state,
            log={"command": command, "result": result_dict, "elapsed_s": elapsed},
        )
        action_video_name = None
        if self._dashboard_events.enabled:
            try:
                frames = self._primitives.frame_slice(start_frame)
                if frames:
                    candidate = f"action_{name}.mp4"
                    if self._state.save(
                        candidate,
                        frames,
                        step=record.step_idx,
                        fps=20,
                    ):
                        action_video_name = candidate
            except Exception as e:
                get_logger("libero_toolkit").warning(
                    "failed to save action clip for step %s: %s",
                    record.step_idx,
                    e,
                )
        out = libero_tools.view_driver_state(record.step_idx, state=self._state)
        out["agent_elapsed_s"] = elapsed
        if result_dict.get("interrupted"):
            out.update(result_dict)
        if action_video_name is not None:
            out["action_video_artifact"] = action_video_name
        return out

    def init_primitives_clean(
        self,
        *,
        primitives_kwargs: dict[str, Any],
    ) -> None:
        """Wipe stale run artifacts, build the LiberoPrimitives, dump step 0."""
        self._state.reset()

        primitives = libero_tools.LiberoPrimitives(
            check_cancelled=self.raise_if_cancelled,
            **primitives_kwargs,
        )
        primitives.reset()
        primitives.start_recording()
        libero_tools.dump_state(primitives, self._state, log=None)
        self._dashboard_events.emit(
            ToolResultEvent(
                name="view_driver_state",
                result=libero_tools.view_driver_state(0, state=self._state),
            )
        )

        self._primitives = primitives

    def close(self) -> None:
        """Flush the agent-side video buffer through ``EnvState``."""
        try:
            frames = self._primitives.stop_recording()
            if frames:
                self._state.save("episode.mp4", frames, step=None, fps=20)
        except Exception as e:
            get_logger("libero_toolkit").warning(
                f"failed to save episode video: {e}"
            )

    def write_recipe(self, recipe_tag: str) -> str:
        """Write the LIBERO recipe JSONL from the dumped state trace."""
        return libero_tools.write_recipe_from_states(self._state, recipe_tag)
