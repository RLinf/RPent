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

"""RoboCasa runtime, tool composition, and observation persistence."""

from __future__ import annotations

import os
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from robots.robocasa import tools as robocasa_tools
from robots.robocasa.rldx_skill import RLDXSkill
from rpent.dashboard.events import DashboardEventSink, StepRecordEvent
from rpent.session import EnvState, StepRecord
from rpent.tools import Toolkit, ToolResult
from rpent.utils.logging import get_logger

if TYPE_CHECKING:
    from robots.robocasa.env_client import RoboCasaEnvClient
    from robots.robocasa.vla_client import RoboCasaVLAClient
    from rpent.memory import MemoryManager

logger = get_logger("robocasa_toolkit")


class RoboCasaRuntime:
    """Clients, VLA history, controller caches, and fixed run configuration."""

    def __init__(
        self,
        env: RoboCasaEnvClient,
        model: RoboCasaVLAClient,
        hi_res: int | None = None,
    ):
        self.env = env
        self.hi_res = hi_res
        self._keep_heavy = int(os.environ.get("RLDX_KEEP_HEAVY_NPY", "25"))
        self._allow_reset = bool(int(os.environ.get("RLDX_ALLOW_RESET", "0")))
        self._pos_jac = None
        self._fwd_offset = None
        self._cam_meta_cache = {}
        self._rldx = RLDXSkill(env, vla_client=model)
        # Manual steps invalidate VLA history; consecutive VLA calls retain it.
        self._vla_desync = True

    def reset(self) -> None:
        self._vla_desync = True
        self.env.reset()
        self._pos_jac = None
        self._fwd_offset = None
        self._rldx.reset_session()


class RoboCasaToolkit(Toolkit[RoboCasaRuntime]):
    """Native tools and observations for one RoboCasa planner session."""

    def __init__(
        self,
        *,
        runtime_kwargs: dict[str, Any],
        output_dir: Path | str,
        dashboard_events: DashboardEventSink,
        memory: MemoryManager,
    ) -> None:
        runtime = RoboCasaRuntime(**runtime_kwargs)
        state = EnvState(output_dir)
        super().__init__(
            state=state,
            memory=memory,
            robot=runtime,
            output_dir=output_dir,
            tools=robocasa_tools.ROBOCASA_TOOLS,
            dashboard_events=dashboard_events,
        )
        runtime.env.reset()
        record = dump_state(runtime, state, log=None)
        try:
            state.save(
                "success_criteria.md",
                runtime.env.get_success_criteria_text(),
                step=None,
            )
        except Exception as exc:
            logger.warning("failed to save success_criteria.md: %s", exc)
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
        """Read success from the final environment record, independent of finish."""
        record = self._state.latest_record()
        return bool(record is not None and record.extras.get("success", False))


# Heavy npy artifacts pruned after the ``_keep_heavy`` window elapses (the
# agent localizes from the latest frame; old world/depth maps are dead weight
# that once filled the 100GB root and deadlocked everything).
_HEAVY_ARTIFACTS = (
    "agentview_depth.npz",
    "agentview_world.npz",
    "wrist_depth.npz",
    "wrist_world.npz",
    "agentview_world_high.npz",
    "navview_world.npz",
)

# Artifact base names exposed to the agent for each camera (high-res first).
_CAMERA_IMAGE_ARTIFACTS = {
    "agentview": ("agentview_high.png", "agentview.png"),
    "navview": ("navview.png",),
    "wrist": ("wrist_high.png", "wrist.png"),
}


def dump_state(
    runtime: RoboCasaRuntime,
    env_state: EnvState,
    log: dict | None = None,
) -> StepRecord:
    """Record one RoboCasa observation through its owned state record.

    Appends a new :class:`StepRecord` (proprio + task + success + vla_desync)
    via :meth:`EnvState.record_step`, saves the rendered RGB / depth / world
    artifacts for that step, and prunes heavy npy artifacts that fell out of
    the ``_keep_heavy`` window.
    """
    raw = runtime.env.current_raw_obs
    extras = {
        "task_language": raw.get("language", runtime.env.get_task_language()) or "",
        "success": runtime.env.check_success(),
        "task_progress": runtime.env.get_task_progress(),
        "vla_desync": runtime._vla_desync,
    }
    state = {
        "robot0_eef_pos": runtime.env.eef_pos.tolist(),
        "robot0_eef_quat": runtime.env.eef_quat.tolist(),
        "robot0_gripper_qpos": runtime.env.gripper_qpos.tolist(),
        "robot0_base_pos": np.asarray(raw["robot0_base_pos"]).tolist(),
        "robot0_base_quat": np.asarray(raw["robot0_base_quat"]).tolist(),
    }
    log = log or {}
    with env_state.record_step(
        state=state,
        terminated=runtime.env.terminated,
        truncated=False,
        command=log.get("command"),
        result=log.get("result"),
        elapsed_s=log.get("elapsed_s"),
        extras=extras,
    ) as step_idx:
        _save_observation_artifacts(runtime, env_state, step_idx)
        env_state.prune_artifacts(
            _HEAVY_ARTIFACTS, step=step_idx, keep_last=runtime._keep_heavy
        )
    return env_state.get(step_idx)


def _save_observation_artifacts(
    runtime: RoboCasaRuntime,
    env_state: EnvState,
    step_idx: int,
) -> None:
    """Render and save all per-step observation artifacts for ``step_idx``."""
    env = runtime.env
    hi_res = runtime.hi_res

    # ---- agentview + wrist: rgb, depth, world map, camera meta ----
    for cam, image_name, depth_name, world_name, meta_name in (
        (
            "agentview",
            "agentview.png",
            "agentview_depth.npz",
            "agentview_world.npz",
            "agentview_metadata.json",
        ),
        (
            "wrist",
            "wrist.png",
            "wrist_depth.npz",
            "wrist_world.npz",
            "wrist_metadata.json",
        ),
    ):
        rgb, depth = env.render_camera(cam, depth=True)
        env_state.save(image_name, rgb, step=step_idx)
        env_state.save(depth_name, depth.astype(np.float32), step=step_idx)
        env_state.save(world_name, env.world_map(cam).astype(np.float32), step=step_idx)
        if cam not in runtime._cam_meta_cache or cam == "wrist":
            runtime._cam_meta_cache[cam] = env.get_camera_meta(cam)
        env_state.save(meta_name, runtime._cam_meta_cache[cam], step=step_idx)

    # ---- hi-res agentview (SAM grounding / fine localize) ----
    if hi_res:
        hrgb, _ = env.render_camera("agentview", hi_res, hi_res, depth=True)
        env_state.save("agentview_high.png", hrgb, step=step_idx)
        env_state.save(
            "agentview_world_high.npz",
            env.world_map("agentview", hi_res, hi_res).astype(np.float16),
            step=step_idx,
        )

    # ---- navview: base-mounted forward-down floor camera (follows the base) ----
    nrgb, _ = env.render_camera("navview", depth=True)
    nworld = env.world_map("navview").astype(np.float32)
    env_state.save("navview.png", nrgb, step=step_idx)
    env_state.save("navview_world.npz", nworld, step=step_idx)
    floor = (nworld[:, :, 2] < 0.12) & (nworld[:, :, 2] > -0.2)
    overlay = nrgb.copy()
    overlay[floor] = [0, 255, 0]
    env_state.save("navview_floor.png", overlay, step=step_idx)


def build_observation(
    state: EnvState, record: StepRecord
) -> tuple[dict[str, Any], list[bytes]]:
    """Return recorded data and ordered agentview, navigation, and wrist PNGs."""
    nn = record.step_idx
    extras = record.extras
    out: dict = {
        "step": nn,
        "task_progress": extras.get("task_progress", {}),
        "task_language": extras.get("task_language", ""),
        "state": record.state,
        "robocasa_terminated": record.terminated,
        "vla_desync": extras.get("vla_desync", False),
        "success": extras.get("success", False),
        "log": {
            "command": record.command,
            "result": record.result,
            "elapsed_s": record.elapsed_s,
        },
        "images": [],
        "artifacts": sorted(record.artifacts),
    }

    images: list[bytes] = []
    for camera, candidates in _CAMERA_IMAGE_ARTIFACTS.items():
        for name in candidates:
            if name not in record.artifacts:
                continue
            try:
                image = state.load_bytes(name, step=nn)
            except FileNotFoundError:
                continue
            images.append(image)
            out["images"].append(
                {
                    "role": "nav_view" if camera == "navview" else "calibration_frame",
                    "camera": camera,
                    "artifact": name,
                }
            )
            break

    return out, images
