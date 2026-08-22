"""RPent tools for the RLinf RoboTwin environment."""

from __future__ import annotations

from functools import partial
from pathlib import Path
from typing import Any

import numpy as np

from robots.robotwin import tools
from robots.robotwin.env_client import _CAMERA_NAMES
from robots.robotwin.primitives import RoboTwinPrimitives
from rpent.dashboard.events import DashboardEventSink
from rpent.tools.state import EnvState
from rpent.tools.toolkit import Toolkit, readonly
from rpent.utils.config import get_resources_dir
from rpent.utils.logging import get_output_dir


def _world_from_depth(
    depth_metric: np.ndarray, camera_meta: dict[str, Any]
) -> np.ndarray:
    """Back-project metric depth into the RoboTwin world frame."""
    depth = np.asarray(depth_metric, dtype=np.float64)
    if depth.ndim != 2:
        raise ValueError(f"RoboTwin depth must have shape [H,W], got {depth.shape}")

    intrinsic = np.asarray(camera_meta.get("intrinsic_K"), dtype=np.float64)
    cam2world = np.asarray(camera_meta.get("cam2world_gl"), dtype=np.float64)
    if intrinsic.shape != (3, 3):
        raise ValueError("RoboTwin camera intrinsic_K must have shape (3,3)")
    if cam2world.shape != (4, 4):
        raise ValueError("RoboTwin camera cam2world_gl must have shape (4,4)")
    if not np.isfinite(intrinsic).all() or not np.isfinite(cam2world).all():
        raise ValueError("RoboTwin camera calibration must contain only finite values")

    height, width = depth.shape
    if camera_meta.get("height") != height or camera_meta.get("width") != width:
        raise ValueError(
            "RoboTwin depth shape does not match camera metadata: "
            f"depth={depth.shape}, metadata="
            f"({camera_meta.get('height')}, {camera_meta.get('width')})"
        )
    fx, fy = intrinsic[0, 0], intrinsic[1, 1]
    cx, cy = intrinsic[0, 2], intrinsic[1, 2]
    rows, cols = np.mgrid[0:height, 0:width]
    camera_points = np.stack(
        [
            (cols - cx) * depth / fx,
            -(rows - cy) * depth / fy,
            -depth,
        ],
        axis=-1,
    )
    world = camera_points @ cam2world[:3, :3].T + cam2world[:3, 3]
    return world.astype(np.float32)


MAX_RESOURCE_CHARS = 40_000
_REPO_PREFIX = ("resources", "robotwin")


def _truncate(text: str, max_chars: int) -> str:
    if len(text) <= max_chars:
        return text
    return (
        text[:max_chars]
        + f"\n\n[TRUNCATED — file is {len(text)} chars, showed first {max_chars}]"
    )


class _RoboTwinResourceReader:
    """Expose only files below ``resources/robotwin`` to the Planner."""

    def __init__(self, root: Path) -> None:
        self._root = root.resolve()

    def _resolve(
        self,
        path: str,
        *,
        allow_root: bool = False,
    ) -> tuple[Path | None, str | None]:
        relative = Path(path)
        if relative.is_absolute():
            return None, "absolute paths are not allowed"
        parts = relative.parts
        if parts[:2] == _REPO_PREFIX:
            relative = Path(*parts[2:])
        if not relative.parts:
            if allow_root:
                return self._root, None
            return None, "path must name a file under resources/robotwin"
        if ".." in relative.parts:
            return None, "path must stay under resources/robotwin"
        resolved = (self._root / relative).resolve()
        try:
            resolved.relative_to(self._root)
        except ValueError:
            return None, "path must stay under resources/robotwin"
        return resolved, None

    @readonly
    def read_text_file(
        self,
        path: str,
        max_chars: int = MAX_RESOURCE_CHARS,
    ) -> dict[str, Any]:
        resolved, error = self._resolve(path)
        if error is not None:
            return {"error": error, "path": path}
        assert resolved is not None
        if not resolved.exists():
            return {"error": "file not found", "path": path}
        if not resolved.is_file():
            return {"error": "is not a file", "path": path}
        try:
            text = resolved.read_text(encoding="utf-8", errors="replace")
        except OSError as exc:
            return {"error": str(exc), "path": path}
        limit = min(max(1, int(max_chars)), MAX_RESOURCE_CHARS)
        return {
            "path": str(resolved.relative_to(self._root)),
            "size": len(text),
            "content": _truncate(text, limit),
        }

    @readonly
    def list_dir(self, path: str = ".") -> dict[str, Any]:
        resolved, error = self._resolve(path, allow_root=True)
        if error is not None:
            return {"error": error, "path": path}
        assert resolved is not None
        if not resolved.exists():
            return {"error": "directory not found", "path": path}
        if not resolved.is_dir():
            return {"error": "is not a directory", "path": path}
        entries = sorted(child.name for child in resolved.iterdir())
        relative = resolved.relative_to(self._root)
        return {
            "path": "." if str(relative) == "." else str(relative),
            "count": len(entries),
            "files": entries,
        }


class RoboTwinToolkit(Toolkit):
    """Common RPent tools plus RoboTwin primitives."""

    _SPECS = {spec["name"]: spec for spec in tools.TOOLS_SPEC}
    _FRAME_ARTIFACTS = {
        "camera": "head_rgb.png",
        "left_wrist": "left_wrist_rgb.png",
        "right_wrist": "right_wrist_rgb.png",
    }

    def __init__(
        self,
        *,
        primitives_kwargs: dict[str, Any],
        dashboard_events: DashboardEventSink,
    ):
        state = EnvState(get_output_dir())
        super().__init__(dashboard_events=dashboard_events, state=state)
        self._recipe: list[dict[str, Any]] = []
        self._latest_status: dict[str, Any] = {}
        self._resource_reader = _RoboTwinResourceReader(get_resources_dir("robotwin"))
        self._primitives = RoboTwinPrimitives(
            check_cancelled=self.raise_if_cancelled,
            **primitives_kwargs,
        )
        reset_result = {
            **self._primitives.env.last_reset_info,
            "success": True,
        }
        self._register_robotwin_tools()
        initial = self._capture("reset", reset_result, elapsed_s=0.0)
        record = self._state.latest_record()
        if record is not None:
            self._publish_step(record)
        initial_state = initial.get("state")
        if isinstance(initial_state, dict):
            self._latest_status = initial_state.get(
                "episode_status", self._latest_status
            )

    def _register_robotwin_tools(self) -> None:
        self._remove_generic_file_tools()
        self._tools.pop("finish", None)
        self.add_tool(
            "read_text_file",
            self._SPECS["read_text_file"],
            self._resource_reader.read_text_file,
        )
        self.add_tool(
            "list_dir",
            self._SPECS["list_dir"],
            self._resource_reader.list_dir,
        )
        self.add_tool(
            "view_env_state",
            self._SPECS["view_env_state"],
            partial(tools.view_env_state, state=self._state),
        )
        self.add_tool(
            "sample_world_xyz",
            self._SPECS["sample_world_xyz"],
            partial(tools.sample_world_xyz, self._state),
        )
        self.add_tool(
            "query_world_map",
            self._SPECS["query_world_map"],
            partial(tools.query_world_map, self._state),
        )
        self.add_tool("render", self._SPECS["render"], partial(self._step, "render"))
        for name in (
            "lingbot_act",
            "move_to",
            "rotate_wrist",
            "set_gripper",
            "release",
        ):
            self.add_tool(name, self._SPECS[name], partial(self._step, name))
        self.add_tool("finish", self._SPECS["finish"], self._finish)

    @readonly
    def _finish(self, *, status: str, summary: str) -> dict[str, Any]:
        return self._primitives.finish(status=status, summary=summary)

    def _remove_generic_file_tools(self) -> None:
        for name in ("read_text_file", "write_text_file", "list_dir"):
            self._tools.pop(name, None)

    def _capture_full_observation(self) -> dict[str, Any]:
        """Assemble the full observation (rgb + depth + camera_meta + world_xyz).

        This is the dump/recording path consumed by ``tools.dump_observation``
        and the ``sample_world_xyz`` agent tool. It deliberately fetches depth
        and camera_meta so ``world_xyz`` can be back-projected and saved as an
        artifact -- distinct from the rgb-only observation built for LingBot
        inference in ``RoboTwinPrimitives._build_lingbot_observation``.
        """
        env = self._primitives.env
        views: dict[str, dict[str, Any]] = {}
        for camera_name in _CAMERA_NAMES:
            rendered = env.render_camera(camera_name, depth=True)
            if not isinstance(rendered, (list, tuple)) or len(rendered) != 2:
                raise TypeError(
                    "RoboTwin render_camera(depth=True) must return (rgb, depth)"
                )
            rgb, depth = rendered
            camera_meta = env.get_camera_meta(camera_name)
            views[camera_name] = {
                "rgb": np.asarray(rgb),
                "depth": np.asarray(depth, dtype=np.float32),
                "world_xyz": _world_from_depth(depth, camera_meta),
                "camera_meta": camera_meta,
            }
        return {
            "views": views,
            "robot_state": env.last_info["robot_state"],
            "task_name": env.server_meta["task_name"],
            "task_language": env.get_task_language(),
            "depth_unit": "metres",
            "world_frame": "world",
        }

    def _capture(
        self, command: str, result: dict[str, Any], *, elapsed_s: float
    ) -> dict[str, Any]:
        status = self._primitives.status()
        self._latest_status = status
        observation = self._capture_full_observation()
        record = tools.dump_observation(
            observation,
            env_state=self._state,
            status=status,
            log={
                "command": command,
                "result": result,
                "elapsed_s": elapsed_s,
            },
        )
        return tools.view_env_state(record.step_idx, state=self._state)

    def get_env_state(
        self,
        *,
        command: dict[str, Any],
        result: dict[str, Any],
        elapsed_s: float,
    ) -> dict[str, Any]:
        return self._capture(
            str(command.get("action", "unknown")),
            result,
            elapsed_s=elapsed_s,
        )

    def _step(self, name: str, **kwargs) -> dict[str, Any]:
        self._recipe.append({"action": name, **kwargs})
        self.raise_if_cancelled()
        if name == "render":
            return {"success": True}
        return getattr(self._primitives, name)(**kwargs)

    def write_recipe(self, recipe_tag: str) -> str:
        name = f"recipe_{recipe_tag}.jsonl"
        saved = self._state.save(name, self._recipe, step=None)
        if saved is None:
            raise RuntimeError(f"failed to save RoboTwin recipe artifact: {name}")
        return str(self._state.artifact_path(name, step=None))
