# Copyright 2026 The RPent Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""RPent tools for the RLinf RoboTwin robot."""

from __future__ import annotations

from functools import partial
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from robots.robotwin import tools
from robots.robotwin.primitives import RoboTwinPrimitives
from robots.robotwin.robot_spec import ROBOTWIN_CAMERA_NAMES
from rpent.dashboard.events import DashboardEventSink
from rpent.session import EnvState
from rpent.tools.toolkit import Toolkit, readonly
from rpent.utils.logging import get_output_dir

if TYPE_CHECKING:
    from rpent.memory.manager import MemoryManager

# State-advancing RoboTwin primitives eligible for the recipe. ``reset``,
# ``render``, and read-only tools are intentionally excluded so the recipe
# records only commands that actually move the robot.
_RECIPE_ACTIONS = {
    "lingbot_act",
    "move_to",
    "rotate_wrist",
    "set_gripper",
    "release",
}


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
        memory: MemoryManager,
        mode: str = "evaluation",
        attempts_per_session: int = 0,
        state_output_dir: Path | str | None = None,
        recipe_output_dir: Path | str | None = None,
    ):
        if mode not in {"evaluation", "exploration"}:
            raise ValueError(f"unsupported RoboTwin toolkit mode: {mode!r}")
        self._mode = mode
        self._attempts_per_session = max(0, int(attempts_per_session))
        self._session_attempt = 1
        self._recipe_output_dir = Path(recipe_output_dir or get_output_dir())
        state = EnvState(state_output_dir or get_output_dir())
        super().__init__(
            dashboard_events=dashboard_events,
            state=state,
            memory=memory,
        )
        self._latest_status: dict[str, Any] = {}
        self._primitives = RoboTwinPrimitives(
            check_cancelled=self.raise_if_cancelled,
            **primitives_kwargs,
        )
        if self._mode == "exploration":
            self._primitives.reset()
        self._primitives.start_recording()
        self._action_frame_cursor = self._primitives.recorded_frame_count()
        reset_result = {
            **self._primitives.env.last_reset_info,
            "success": True,
        }
        self._register_robotwin_tools()
        initial = self.get_env_state(
            command={"action": "reset"},
            result=reset_result,
            elapsed_s=0.0,
        )
        record = self._state.latest_record()
        if record is not None:
            self._publish_step(record)
        initial_state = initial.get("state")
        if isinstance(initial_state, dict):
            self._latest_status = initial_state.get(
                "episode_status", self._latest_status
            )

    def _register_robotwin_tools(self) -> None:
        self._tools.pop("finish", None)
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
        for name in (
            "render",
            "lingbot_act",
            "move_to",
            "rotate_wrist",
            "set_gripper",
            "release",
        ):
            self.add_tool(name, self._SPECS[name], partial(self._step, name))
        self.add_tool("finish", self._SPECS["finish"], self._finish)

        if self._mode == "exploration":
            self.add_tool("reset", self._SPECS["reset"], self._reset_episode)

    def solved(self) -> bool:
        """Read native status cached by the env client after reset/actions."""
        return self._primitives.status().get("eval_success") is True

    def _reset_episode(self, reason: str) -> dict[str, Any]:
        if self.solved():
            return {
                "error": "reset refused",
                "reason": "Task already solved; call finish.",
            }
        budget = self._attempts_per_session
        if budget and self._session_attempt >= budget:
            return {
                "error": "reset refused",
                "reason": "Attempt budget spent; archive and finish for handoff.",
            }
        # Match LIBERO: a reset invocation consumes an attempt, even if it fails.
        self._session_attempt += 1
        result = self._primitives.reset()
        self._latest_status = {}
        return {
            **result,
            "attempt": self._session_attempt,
            "reason": reason,
            "notice": (
                "Episode reinitialized with the configured exact seed. "
                "Full physical layout determinism has not been verified. "
                "Re-run perception before acting."
            ),
        }

    @readonly
    def _finish(self, *, status: str, summary: str) -> dict[str, Any]:
        budget = self._attempts_per_session
        if (
            self._mode == "exploration"
            and budget
            and not self.solved()
            and self._session_attempt < budget
        ):
            return {
                "error": "finish refused",
                "reason": (
                    f"This session has {budget - self._session_attempt} attempts left. "
                    "Archive this attempt, reset, and try another approach."
                ),
            }
        return self._primitives.finish(status=status, summary=summary)

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
        for camera_name in ROBOTWIN_CAMERA_NAMES:
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

    def get_env_state(
        self,
        *,
        command: dict[str, Any],
        result: dict[str, Any],
        elapsed_s: float,
    ) -> dict[str, Any]:
        frame_start = self._action_frame_cursor
        self._action_frame_cursor = self._primitives.recorded_frame_count()
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
        if self._dashboard_events.enabled:
            frames = self._primitives.frame_slice(frame_start)
            if frames:
                self._state.save(
                    f"action_{command['action']}.mp4",
                    frames,
                    step=record.step_idx,
                    fps=20,
                )
        return tools.view_env_state(record.step_idx, state=self._state)

    def close(self) -> None:
        """Flush the per-step frame buffer into ``episode.mp4`` (LIBERO parity)."""
        frames = self._primitives.stop_recording()
        if frames:
            self._state.save("episode.mp4", frames, step=None, fps=20)

    def _step(self, name: str, **kwargs) -> dict[str, Any]:
        self.raise_if_cancelled()
        if name == "render":
            return {"success": True}
        return getattr(self._primitives, name)(**kwargs)

    def write_recipe(self, recipe_tag: str) -> str:
        """Export state-advancing RoboTwin primitives with no error and no
        explicit ``success=False`` from ``EnvState.records()``."""
        records = self._state.records()
        if self._mode == "exploration":
            if not self.solved():
                return ""
            last_reset = max(
                (
                    r.step_idx
                    for r in records
                    if (r.command or {}).get("action") == "reset"
                    and not (r.result or {}).get("error")
                ),
                default=-1,
            )
            records = [r for r in records if r.step_idx > last_reset]
            if not any(r.terminated for r in records):
                return ""
        recipe = [
            record.command
            for record in records
            if isinstance(record.command, dict)
            and record.command.get("action") in _RECIPE_ACTIONS
            and not (
                isinstance(record.result, dict)
                and (
                    record.result.get("error") or record.result.get("success") is False
                )
            )
        ]
        name = f"{recipe_tag}_recipe.jsonl"
        destination = (
            EnvState(self._recipe_output_dir)
            if self._mode == "exploration"
            else self._state
        )
        saved = destination.save(name, recipe, step=None)
        if saved is None:
            raise RuntimeError(f"failed to save RoboTwin recipe artifact: {name}")
        return str(destination.artifact_path(name, step=None))
