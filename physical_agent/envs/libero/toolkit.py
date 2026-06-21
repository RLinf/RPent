"""LIBERO toolkit: common tools + LIBERO primitives.

Inherits the common file/IO tools from :class:`Toolkit` and registers the
LIBERO primitives (``move_to``, ``pi0_pick``, ``release``, ...) on top.
"""
from __future__ import annotations

import shutil
import time
from typing import Any

from physical_agent.driver_client.proxies import RemoteEnvProxy
from physical_agent.envs.libero import tools as libero_tools
from physical_agent.tools.toolkit import Toolkit
from physical_agent.utils.logging import get_logger, get_output_dir


class LiberoToolkit(Toolkit):
    """Toolkit for the LIBERO environment."""

    # MCP tool names this toolkit exposes (namespaced for the SDK allowlist).
    _ALLOWED_MCP_TOOL_NAMES = tuple(
        f"mcp__physical_agent__{name}"
        for name in [
            "move_to",
            "pi0_pick",
            "release",
            "set_gripper",
            "rotate_wrist",
            "rotate_pitch",
            "move_pose",
            "view_driver_state",
            "view_camera_meta",
            "back_project",
            "read_text_file",
            "write_text_file",
            "mcp_list_dir",
            "finish",
        ]
    )
    allowed_mcp_tool_names = _ALLOWED_MCP_TOOL_NAMES

    # Tool schemas keyed by name (built once from the canonical ordered list
    # in libero_tools.TOOLS_SPEC) so each tool registers with its own spec.
    _SPECS = {spec["name"]: spec for spec in libero_tools.TOOLS_SPEC}

    # Bound in set_driver_client() and non-null for the rest of the run —
    # toolkit contract: primitive tools are only invoked afterwards, so call
    # sites use self._driver directly with no None-check.
    _driver: libero_tools.LiberoPrimitives

    def __init__(self) -> None:
        super().__init__()
        self._next_step: int = 0
        self._video_path: str | None = None
        self._register_libero_tools()

    # ------------------------------------------------------------------
    # Registration — one explicit add_tool per LIBERO tool.
    # ------------------------------------------------------------------
    def _register_libero_tools(self) -> None:
        spec = self._SPECS  # name -> schema, built once from libero_tools.TOOLS_SPEC
        # Read tools: stateless readers over the output dir (no env step).
        self.add_tool("view_driver_state", spec["view_driver_state"],
                      self.view_driver_state)
        self.add_tool("view_camera_meta", spec["view_camera_meta"],
                      self.view_camera_meta)
        self.add_tool("back_project", spec["back_project"], self.back_project)
        self.add_tool("finish", spec["finish"], self.finish)
        # Primitive tools: step the env, dump state, return the rendered view.
        self.add_tool("move_to", spec["move_to"], self.move_to)
        self.add_tool("pi0_pick", spec["pi0_pick"], self.pi0_pick)
        self.add_tool("release", spec["release"], self.release)
        self.add_tool("set_gripper", spec["set_gripper"], self.set_gripper)
        self.add_tool("rotate_wrist", spec["rotate_wrist"], self.rotate_wrist)
        self.add_tool("rotate_pitch", spec["rotate_pitch"], self.rotate_pitch)
        self.add_tool("move_pose", spec["move_pose"], self.move_pose)

    # ------------------------------------------------------------------
    # Read tools — straight delegation to the stateless readers.
    # ------------------------------------------------------------------
    def view_driver_state(self, step: int | None = None) -> dict:
        return libero_tools.view_driver_state(step)

    def view_camera_meta(self) -> dict:
        return libero_tools.view_camera_meta()

    def back_project(self, row: int, col: int, step: int | None = None) -> dict:
        return libero_tools.back_project(row, col, step)

    def finish(self, status: str, summary: str) -> dict:
        return libero_tools.finish(status, summary)

    # ------------------------------------------------------------------
    # Primitive tools — each calls its named driver method, then records
    # the step via _step(). The driver is bound in set_driver_client(),
    # which always runs before any primitive is invoked.
    # ------------------------------------------------------------------
    def move_to(self, **kwargs) -> dict:
        return self._step("move_to", self._driver.move_to, kwargs)

    def pi0_pick(self, **kwargs) -> dict:
        return self._step("pi0_pick", self._driver.pi0_pick, kwargs)

    def release(self, **kwargs) -> dict:
        return self._step("release", self._driver.release, kwargs)

    def set_gripper(self, **kwargs) -> dict:
        return self._step("set_gripper", self._driver.set_gripper, kwargs)

    def rotate_wrist(self, **kwargs) -> dict:
        return self._step("rotate_wrist", self._driver.rotate_wrist, kwargs)

    def rotate_pitch(self, **kwargs) -> dict:
        return self._step("rotate_pitch", self._driver.rotate_pitch, kwargs)

    def move_pose(self, **kwargs) -> dict:
        return self._step("move_pose", self._driver.move_pose, kwargs)

    def _step(self, name: str, fn: Any, kwargs: dict) -> dict:
        """Run a bound driver primitive ``fn``, dump the new step, and return
        the rendered state view + log.

        ``name`` is the tool name (also the driver method name) used to label
        the command log; ``fn`` is the already-bound driver method, e.g.
        ``self._driver.move_to``.
        """
        command = {"action": name, **kwargs}
        t0 = time.time()
        result = fn(**kwargs)
        elapsed = round(time.time() - t0, 2)

        if isinstance(result, dict):
            result_dict = result
        elif hasattr(result, "__dataclass_fields__"):
            result_dict = result.__dict__
        else:
            result_dict = {"value": result}

        self._next_step += 1
        step_idx = self._next_step
        libero_tools.dump_state(
            self._driver,
            str(get_output_dir()),
            step_idx=step_idx,
            log={"command": command, "result": result_dict, "elapsed_s": elapsed},
        )
        out = libero_tools.view_driver_state(step_idx)
        out["agent_elapsed_s"] = elapsed
        return out

    def set_driver_client(
        self,
        client: Any,
        *,
        model: Any,
        hide_object_coords: bool = False,
        video_path: str | None = None,
    ) -> None:
        """Bind the wire, build the LIBERO primitive driver, and dump step 0."""
        out_dir = get_output_dir()
        out_dir.mkdir(parents=True, exist_ok=True)
        for sub in ("images", "images_cam", "depths"):
            target = out_dir / sub
            if target.exists():
                shutil.rmtree(target)
        for fname in ("states.json", "camera_meta.json", "episode.mp4"):
            target = out_dir / fname
            if target.exists():
                target.unlink()

        driver = libero_tools.LiberoPrimitives(
            env=RemoteEnvProxy(client),
            model=model,
            action_chunk=5,
        )
        driver._hide_object_coords = hide_object_coords
        driver.reset()
        driver.start_recording()
        libero_tools.dump_state(driver, str(out_dir), step_idx=0, log=None)

        self._driver = driver
        self._next_step = 0
        self._video_path = video_path

    def release_driver_client(self) -> None:
        """Flush the agent-side video buffer to disk (end-of-run).
        """
        if self._video_path is None:
            return
        try:
            self._driver.stop_recording_and_save(self._video_path)
        except Exception as e:
            # The runner is in the cleanup path; never let a video save
            # abort it.
            get_logger("libero_toolkit").warning(
                f"failed to save video to {self._video_path}: {e}"
            )
