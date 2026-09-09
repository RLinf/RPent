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

"""LIBERO tool composition, observations, exploration, and recording lifecycle."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

import numpy as np

from robots.libero import tools as libero_tools
from rpent.dashboard.events import DashboardEventSink, StepRecordEvent
from rpent.memory import MemoryManager
from rpent.session import EnvState, StepRecord
from rpent.tools import Toolkit, ToolResult
from rpent.utils.logging import get_logger

if TYPE_CHECKING:
    from robots.libero.env_client import LiberoEnvClient
    from rpent.robots.components.molmo_client import MolmoClient
    from rpent.robots.components.pi05_vla_client import Pi05VLAClient
    from rpent.robots.components.sam3_client import Sam3Client

logger = get_logger("libero_toolkit")


class LiberoRuntime:
    """Environment clients, cached observations, and session progress."""

    def __init__(
        self,
        env: LiberoEnvClient,
        model: Pi05VLAClient,
        sam3_client: Sam3Client,
        molmo_client: MolmoClient | None = None,
    ):
        self.env = env
        self.model = model
        self._sam3_client = sam3_client
        self.molmo_client = molmo_client
        self._last_obs = None
        self._last_obs_eef_pos = None
        self._last_obs_gripper = None
        self.executed_steps = 0
        self.mode: Literal["evaluation", "exploration"] = "evaluation"
        self.solved = False
        self.attempt = 1
        self.attempts_per_session = 0

    def set_obs(self, obs):
        self._last_obs = obs
        states_arr = np.asarray(obs["states"])
        self._last_obs_eef_pos = np.asarray(states_arr[:3], dtype=np.float32)
        # robosuite 2f85: qpos[6] in [~0, ~0.04], qpos[7] in [~-0.04, ~0.].
        # Use |qpos[6]| + |qpos[7]| ≈ finger separation proxy.
        # When open ≈ 0.08; when closed ≈ 0.
        gp = np.asarray(states_arr[6:8], dtype=np.float32)
        self._last_obs_gripper = float(abs(gp[0]) + abs(gp[1]))

    def reset(self) -> None:
        obs, _ = self.env.reset()
        self.set_obs(obs)


class LiberoToolkit(Toolkit[LiberoRuntime]):
    """Native tools and resources for one LIBERO planner session."""

    def __init__(
        self,
        *,
        runtime_kwargs: dict[str, Any],
        output_dir: Path | str,
        dashboard_events: DashboardEventSink,
        memory: MemoryManager,
        mode: str = "evaluation",
        attempts_per_session: int = 0,
        state_output_dir: Path | str | None = None,
    ) -> None:
        if mode not in {"evaluation", "exploration"}:
            raise ValueError(f"unsupported LIBERO toolkit mode: {mode!r}")
        runtime = LiberoRuntime(**runtime_kwargs)
        runtime.mode = mode
        runtime.attempts_per_session = max(0, int(attempts_per_session))
        state = EnvState(state_output_dir or output_dir)
        tools = libero_tools.LIBERO_TOOLS
        if mode == "evaluation":
            tools = tuple(tool for tool in tools if tool.name != "reset")
        super().__init__(
            state=state,
            memory=memory,
            robot=runtime,
            output_dir=output_dir,
            tools=tools,
            dashboard_events=dashboard_events,
        )
        runtime.reset()
        record = dump_state(runtime, state, log=None)
        try:
            self._dashboard_events.emit(
                StepRecordEvent(record=record, env_state=self._state)
            )
        except Exception:
            logger.exception("Dashboard failed to publish step %s", record.step_idx)

    def _capture_observation(
        self, *, command: dict[str, Any], result: ToolResult, elapsed_s: float
    ) -> tuple[dict[str, Any], list[bytes]]:
        """Save the full action log, then assemble the current observation response."""
        logged_result = result.to_dict()
        record = dump_state(
            self._robot,
            self._state,
            log={
                "command": command,
                "result": logged_result,
                "elapsed_s": elapsed_s,
            },
        )
        self._robot.solved |= record.terminated
        data, images = build_observation(self._state, record)
        if result.is_error:
            # Report the error once in this response; retain it in the saved log
            # so later view_env_state calls can still inspect the failed action.
            data["log"]["result"] = {
                key: value for key, value in logged_result.items() if key != "error"
            }
        data["agent_elapsed_s"] = elapsed_s
        return data, images

    @property
    def molmo_client(self) -> MolmoClient | None:
        """Return the optional point grounder used by task-card replay."""
        return self._robot.molmo_client

    def solved(self) -> bool:
        return self._robot.solved


def dump_state(
    runtime: LiberoRuntime,
    env_state: EnvState,
    log: dict | None = None,
) -> StepRecord:
    """Save one Libero observation through its owned state record."""
    raw = runtime.env.raw_obs()
    state = {
        "robot0_eef_pos": [float(x) for x in raw["robot0_eef_pos"]],
        "robot0_eef_quat": [float(x) for x in raw["robot0_eef_quat"]],
        "robot0_gripper_qpos": [float(x) for x in raw["robot0_gripper_qpos"]],
        "object_names": sorted(
            k[:-4]
            for k in raw
            if k.endswith("_pos") and "robot0" not in k and "to_robot" not in k
        ),
    }
    log = log or {}
    with env_state.record_step(
        state=state,
        terminated=runtime.env.terminated,
        truncated=runtime.env.truncated,
        command=log.get("command"),
        result=log.get("result"),
        elapsed_s=log.get("elapsed_s"),
        extras={"task_language": runtime.env.get_task_language()},
    ) as step_idx:
        _save_observation_artifacts(runtime, env_state, step_idx, raw)
    return env_state.get(step_idx)


def build_observation(
    state: EnvState, record: StepRecord
) -> tuple[dict[str, Any], list[bytes]]:
    """Assemble a recorded observation and its ordered PNG images."""
    nn = record.step_idx
    extras = record.extras
    out: dict = {
        "step": nn,
        "terminated": record.terminated,
        "truncated": record.truncated,
        "state": record.state,
        "artifacts": sorted(record.artifacts),
    }
    out["task_language"] = extras.get("task_language")
    out["log"] = {
        "command": record.command,
        "result": record.result,
        "elapsed_s": record.elapsed_s,
    }
    images: list[bytes] = []
    for names in (
        ("agentview_policy.png",),
        ("agentview_high.png", "agentview.png"),
        ("wrist_high.png", "wrist.png"),
    ):
        name = next((name for name in names if name in record.artifacts), None)
        if name:
            try:
                images.append(state.load_bytes(name, step=nn))
            except FileNotFoundError:
                pass
    return out, images


def _save_observation_artifacts(
    runtime: LiberoRuntime,
    state: EnvState,
    step: int,
    raw: dict[str, Any],
) -> None:
    """Save the policy view, then each camera's calibrated and high-res views."""
    state.save("agentview_policy.png", runtime._last_obs["main_images"], step=step)
    _save_agentview_artifacts(runtime.env, state, step, raw)
    _save_wrist_artifacts(runtime.env, state, step, raw)

    for camera, prefix in (
        ("agentview", "agentview"),
        ("robot0_eye_in_hand", "wrist"),
    ):
        try:
            rgb, depth = runtime.env.render_camera(
                camera_name=camera, height=1024, width=1024, depth=True
            )
            camera_meta = runtime.env.get_camera_meta(camera, 1024, 1024)
            if camera_meta is None:
                raise RuntimeError(f"{camera} camera metadata missing")
            state.save(f"{prefix}_high.png", np.asarray(rgb)[::-1], step=step)
            depth_metric = _metric_depth(depth, camera_meta)[::-1]
            world = _world_from_depth(depth_metric, camera_meta).astype(np.float16)
            state.save(f"{prefix}_world_high.npz", world, step=step)
        except Exception as exc:
            logger.warning("%s high-res dump failed: %s", prefix, exc)


def _save_agentview_artifacts(
    env: LiberoEnvClient, state: EnvState, step: int, raw: dict[str, Any]
) -> None:
    camera_meta = (
        env.get_camera_meta(camera_name="agentview", height=256, width=256) or {}
    )
    if camera_meta:
        metadata = dict(camera_meta)
        metadata["projection"] = (
            "Prefer the back_project(row, col, step=NN) MCP tool; it "
            "uses the 1024x1024 high-resolution world map by default. "
            "Pass resolution='low' only when row/col came from the "
            "256x256 calibration-frame image."
        )
        metadata["note"] = (
            "The agentview_depth.npz observation is aligned with agentview.png. "
            "agentview_policy.png uses the Pi0 orientation and must not supply "
            "pixels for back-projection."
        )
        state.save("agentview_metadata.json", metadata, step=step)

    try:
        image = raw.get("agentview_image")
        if image is not None:
            image = np.asarray(image, dtype=np.uint8)
            state.save("agentview.png", image[::-1], step=step)
    except Exception as exc:
        logger.warning("image_cam dump failed: %s", exc)

    try:
        depth = raw.get("agentview_depth")
        if depth is not None:
            depth_metric = _metric_depth(depth, camera_meta)[::-1]
            state.save(
                "agentview_depth.npz", depth_metric.astype(np.float32), step=step
            )
            world = _world_from_depth(depth_metric, camera_meta).astype(np.float32)
            state.save("agentview_world.npz", world, step=step)
    except Exception as exc:
        logger.warning("depth dump failed: %s", exc)


def _save_wrist_artifacts(
    env: LiberoEnvClient, state: EnvState, step: int, raw: dict[str, Any]
) -> None:
    try:
        image = raw.get("robot0_eye_in_hand_image")
        if image is None:
            logger.warning("wrist image missing from raw_obs")
        else:
            image = np.asarray(image, dtype=np.uint8)
            state.save("wrist.png", image[::-1], step=step)
    except Exception as exc:
        logger.warning("wrist image dump failed: %s", exc)

    try:
        depth = raw.get("robot0_eye_in_hand_depth")
        if depth is None:
            logger.warning("wrist depth missing from raw_obs")
            return
        depth = np.asarray(depth, dtype=np.float32)
        height, width = depth.shape[:2]
        camera_meta = env.get_camera_meta(
            camera_name="robot0_eye_in_hand", height=int(height), width=int(width)
        )
        if camera_meta is None:
            logger.warning("wrist camera meta missing; skipping wrist depth/world")
            return
        depth_metric = _metric_depth(depth, camera_meta)[::-1]
        state.save("wrist_depth.npz", depth_metric.astype(np.float32), step=step)
        world = _world_from_depth(depth_metric, camera_meta).astype(np.float32)
        state.save("wrist_world.npz", world, step=step)
        metadata = dict(camera_meta)
        metadata["note"] = (
            "MOVING camera: extrinsic_cam2world is for THIS step "
            "only. The matching wrist world-map observation gives world "
            "(x,y,z) for that pixel in the same world frame as the "
            "agentview world-map artifact."
        )
        state.save("wrist_metadata.json", metadata, step=step)
    except Exception as exc:
        logger.warning("wrist depth/world dump failed: %s", exc)


def _metric_depth(depth: Any, camera_meta: dict) -> np.ndarray:
    """Convert the raw depth buffer to meters before flipping to the RGB frame."""
    d = np.asarray(depth, dtype=np.float32)
    if d.ndim == 3:
        d = d[..., 0]
    near = camera_meta.get("depth_near")
    far = camera_meta.get("depth_far")
    if near is not None and far is not None:
        d = near / (1.0 - d * (1.0 - near / far))
    return d


def _world_from_depth(depth_metric: np.ndarray, camera_meta: dict) -> np.ndarray:
    """Back-project calibrated image pixels to world XYZ in meters."""
    k_matrix = np.array(camera_meta["intrinsic_K"], dtype=np.float64)
    extrinsic = np.array(camera_meta["extrinsic_cam2world"], dtype=np.float64)
    fx, fy = k_matrix[0, 0], k_matrix[1, 1]
    cx, cy = k_matrix[0, 2], k_matrix[1, 2]
    height, width = depth_metric.shape
    rr, cc = np.mgrid[0:height, 0:width]
    z = depth_metric.astype(np.float64)
    camera_points = np.stack(
        [(cc - cx) * z / fx, (rr - cy) * z / fy, z, np.ones_like(z)],
        axis=-1,
    )
    return (camera_points @ extrinsic.T)[..., :3]
