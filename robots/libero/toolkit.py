"""LIBERO toolkit: common tools + LIBERO primitives.

Inherits the common file/IO tools from :class:`Toolkit` and registers the
LIBERO primitives (``move_to``, ``pi0_pick``, ``release``, ...) on top.
"""
from __future__ import annotations

from functools import partial
from typing import Any

from robots.libero import tools as libero_tools
from rpent.dashboard.events import DashboardEventSink
from rpent.tools.state import EnvState
from rpent.tools.toolkit import Toolkit
from rpent.utils.logging import get_logger, get_output_dir


class LiberoToolkit(Toolkit):
    """Toolkit for the LIBERO environment."""

    _FRAME_ARTIFACTS = {
        "camera": "agentview.png",
        "wrist": "wrist.png",
    }

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
        # These read-only handlers need the run's EnvState bound in. Every
        # other spec binds to a primitive-driver method and captures state by
        # default unless that method is explicitly marked @readonly.
        state_handlers = {
            "view_env_state": partial(
                libero_tools.view_env_state, state=self._state
            ),
            "view_camera_meta": partial(
                libero_tools.view_camera_meta, state=self._state
            ),
            "back_project": partial(libero_tools.back_project, state=self._state),
            "segment": partial(self._primitives.segment, state=self._state),
        }
        for spec in libero_tools.TOOLS_SPEC:
            name = spec["name"]
            if name in state_handlers:
                handler = state_handlers[name]
            else:
                handler = getattr(self._primitives, name, None)
                if handler is None:
                    continue  # spec without a backing primitive method
            self.add_tool(name, spec, handler)

    def get_env_state(
        self,
        *,
        command: dict[str, Any],
        result: dict[str, Any],
        elapsed_s: float,
    ) -> dict[str, Any]:
        frame_start = self._action_frame_cursor
        self._action_frame_cursor = self._primitives.recorded_frame_count()
        record = libero_tools.dump_state(
            self._primitives,
            self._state,
            log={"command": command, "result": result, "elapsed_s": elapsed_s},
        )
        if self._dashboard_events.enabled:
            try:
                frames = self._primitives.frame_slice(frame_start)
                if frames:
                    candidate = f"action_{command['action']}.mp4"
                    self._state.save(
                        candidate,
                        frames,
                        step=record.step_idx,
                        fps=20,
                    )
            except Exception as e:
                get_logger("libero_toolkit").warning(
                    "failed to save action clip for step %s: %s",
                    record.step_idx,
                    e,
                )
        out = libero_tools.view_env_state(record.step_idx, state=self._state)
        out["agent_elapsed_s"] = elapsed_s
        if result.get("interrupted"):
            out.update(result)
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
        self._action_frame_cursor = primitives.recorded_frame_count()
        record = libero_tools.dump_state(primitives, self._state, log=None)
        self._primitives = primitives
        self._publish_step(record)

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
