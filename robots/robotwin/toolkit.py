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

"""RoboTwin runtime, tool composition, and observation persistence."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from robots.robotwin import tools
from robots.robotwin.robot_spec import ROBOTWIN_CAMERA_NAMES
from rpent.dashboard.events import DashboardEventSink, StepRecordEvent
from rpent.session import EnvState, StepRecord
from rpent.tools import Toolkit, ToolResult
from rpent.utils.logging import get_logger

if TYPE_CHECKING:
    from robots.robotwin.env_client import RoboTwinEnvClient
    from robots.robotwin.vla_client import LingBotVLAClient
    from rpent.memory import MemoryManager

logger = get_logger("robotwin_toolkit")


class RoboTwinRuntime:
    """Clients, fixed episode configuration, and cumulative action counts."""

    def __init__(
        self,
        *,
        env: RoboTwinEnvClient,
        model: LingBotVLAClient,
        seed: int,
        seed_mode: str = "exact",
    ) -> None:
        if seed_mode != "exact":
            raise ValueError("standard RoboTwin integration requires seed_mode='exact'")
        self.env = env
        self.model = model
        self.seed = int(seed)
        self.policy_actions = 0
        self.native_actions = 0

    def status(self) -> dict[str, Any]:
        return {
            **self.env.last_info["episode_status"],
            "policy_actions": self.policy_actions,
            "native_actions": self.native_actions,
        }


class RoboTwinToolkit(Toolkit[RoboTwinRuntime]):
    """Native tools and observations for one RoboTwin planner session."""

    def __init__(
        self,
        *,
        runtime_kwargs: dict[str, Any],
        output_dir: Path | str,
        dashboard_events: DashboardEventSink,
        memory: MemoryManager,
    ) -> None:
        runtime = RoboTwinRuntime(**runtime_kwargs)
        state = EnvState(output_dir)
        super().__init__(
            state=state,
            memory=memory,
            robot=runtime,
            output_dir=output_dir,
            tools=tools.ROBOTWIN_TOOLS,
            dashboard_events=dashboard_events,
        )
        # The env client is already reset to the requested seed during startup.
        record = dump_state(
            runtime,
            state,
            log={
                "command": {"action": "reset"},
                "result": {**runtime.env.last_reset_info, "success": True},
                "elapsed_s": 0.0,
            },
        )
        try:
            self._dashboard_events.emit(StepRecordEvent(record=record, env_state=state))
        except Exception:
            logger.exception("Dashboard failed to publish step %s", record.step_idx)

    def _capture_observation(
        self,
        *,
        command: dict[str, Any],
        result: ToolResult,
        elapsed_s: float,
    ) -> tuple[dict[str, Any], list[bytes]]:
        logged_result = result.to_dict()
        record = dump_state(
            self._robot,
            self._state,
            log={"command": command, "result": logged_result, "elapsed_s": elapsed_s},
        )
        data, images = build_observation(self._state, record)
        if result.is_error:
            data["log"]["result"] = {
                key: value for key, value in logged_result.items() if key != "error"
            }
        data["agent_elapsed_s"] = elapsed_s
        return data, images

    def solved(self) -> bool:
        """Use recorded native success, independently of the requested finish status."""
        record = self._state.latest_record()
        return bool(
            record is not None
            and record.state["episode_status"].get("eval_success") is True
        )


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


def _artifact_name(view: str, field: str) -> str:
    suffix = {
        "rgb": ".png",
        "depth": ".npy",
        "world_xyz": ".npy",
        "camera_meta": ".json",
    }[field]
    return f"{view}_{field}{suffix}"


def dump_state(
    runtime: RoboTwinRuntime,
    env_state: EnvState,
    log: dict[str, Any],
) -> StepRecord:
    """Persist synchronized RGB, metric depth, and world maps for all three views."""
    env = runtime.env
    views = {}
    for camera in ROBOTWIN_CAMERA_NAMES:
        rgb, depth = env.render_camera(camera, depth=True)
        camera_meta = env.get_camera_meta(camera)
        views[camera] = {
            "rgb": np.asarray(rgb),
            "depth": np.asarray(depth, dtype=np.float32),
            "world_xyz": _world_from_depth(depth, camera_meta),
            "camera_meta": camera_meta,
        }
    step_idx = 0 if env_state.latest_step is None else env_state.latest_step + 1
    state = {
        "step_idx": step_idx,
        "task_name": env.server_meta["task_name"],
        "task_language": env.get_task_language(),
        "robot_state": env.last_info["robot_state"],
        "episode_status": runtime.status(),
        "artifacts": {
            camera: {
                field: str(
                    env_state.artifact_path(
                        _artifact_name(camera, field), step=step_idx
                    )
                )
                for field in view
            }
            for camera, view in views.items()
        },
        "view_specs": {
            camera: {
                "coordinate_space": camera,
                "image_shape": list(view["rgb"].shape[:2]),
                "pixel_order": "row_col",
            }
            for camera, view in views.items()
        },
    }
    with env_state.record_step(
        state=state,
        terminated=env.terminated,
        truncated=env.truncated,
        command=log["command"],
        result=log["result"],
        elapsed_s=log["elapsed_s"],
        extras={"task_language": state["task_language"]},
    ) as recorded_step:
        for camera, view in views.items():
            for field, value in view.items():
                env_state.save(_artifact_name(camera, field), value, step=recorded_step)
    return env_state.get(step_idx)


def build_observation(
    state: EnvState, record: StepRecord
) -> tuple[dict[str, Any], list[bytes]]:
    """Return recorded data and head, left wrist, and right wrist PNGs in order."""
    data = {
        "step": record.step_idx,
        "terminated": record.terminated,
        "truncated": record.truncated,
        "state": record.state,
        "artifacts": sorted(record.artifacts),
        "task_language": record.extras["task_language"],
        "log": {
            "command": record.command,
            "result": record.result,
            "elapsed_s": record.elapsed_s,
        },
    }
    images = []
    for camera in ROBOTWIN_CAMERA_NAMES:
        name = _artifact_name(camera, "rgb")
        if name in record.artifacts:
            try:
                images.append(state.load_bytes(name, step=record.step_idx))
            except FileNotFoundError:
                pass
    return data, images
