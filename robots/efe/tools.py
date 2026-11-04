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

"""EFE raw tools and canonical state capture.

These tools expose only what the task's standard-mode client exposes: the
capability spec, current frames and proprioception, episode status, and
validated action chunks. They never read simulator internals.
"""

from __future__ import annotations

import io
from collections.abc import Mapping
from typing import Annotated, Any, Literal

import imageio.v2 as imageio
import numpy as np
from pydantic import Field

from robots.efe.env_client import EfeEnvClient
from rpent.session import EnvState, StepRecord
from rpent.tools import ToolResult, tool

# Downsampling factors for the tool's ``resolution`` argument. Camera
# coordinates scale by the same integer, so a model that sees a frame at
# ``low`` must multiply pixel coordinates by four.
_RESOLUTION_FACTORS = {"full": 1, "low": 4}


def _camera_slug(camera: str) -> str:
    """Return a filename-safe form of a camera name."""
    return "".join(
        character if character.isalnum() or character in "_.-" else "_"
        for character in camera
    )


def camera_artifact(camera: str, episode: int, step: int | None) -> str:
    """Return the artifact name that holds one camera frame.

    Frames are session artifacts rather than step artifacts: an observation
    happens before the step record exists, so the episode and simulator step
    are encoded in the name instead of the directory index. A reset restarts
    the step counter, so without the episode index a new episode would
    overwrite the frames of the previous one.
    """
    slug = _camera_slug(camera)
    tag = f"ep{int(episode):03d}"
    if step is None:
        return f"camera_{slug}_{tag}.png"
    return f"camera_{slug}_{tag}_s{int(step):06d}.png"


def depth_artifact(camera: str, episode: int, step: int | None) -> str:
    """Return the artifact name that holds one depth map, in metres.

    Depth follows the frame naming rule, so each observation keeps its own
    file and a returned path stays valid for the rest of the run.
    """
    slug = _camera_slug(camera)
    tag = f"ep{int(episode):03d}"
    if step is None:
        return f"depth_{slug}_{tag}.npy"
    return f"depth_{slug}_{tag}_s{int(step):06d}.npy"


def proprio_map(spec: dict[str, Any], payload: dict[str, Any]) -> dict[str, float]:
    """Return one response's proprioception as ``name -> value``."""
    layout = payload.get("layout") or spec["proprio_layout"]
    values = payload.get("now") or []
    return {name: float(value) for name, value in zip(layout, values, strict=False)}


def summarize_spec(spec: dict[str, Any]) -> dict[str, Any]:
    """Return the capability subset an agent needs to choose commands."""
    kinematics = spec.get("kinematics") or {}
    return {
        "protocol": spec.get("protocol"),
        "robot": spec.get("robot"),
        "control_hz": spec.get("control_hz"),
        "max_chunk_s": spec.get("max_chunk_s"),
        "max_control_steps": spec.get("max_control_steps"),
        "max_frames_per_call": spec.get("max_frames_per_call"),
        "action_types": {
            name: {
                "dim": action.get("dim"),
                "layout": action.get("layout"),
                "limits": action.get("limits"),
                "units": action.get("units"),
                "semantics": action.get("semantics"),
            }
            for name, action in (spec.get("action_types") or {}).items()
            if isinstance(action, dict)
        },
        "cameras": sorted((spec.get("cameras") or {}).keys()),
        "default_cameras": spec.get("default_cameras"),
        "proprio_groups": spec.get("proprio_groups"),
        "tcp_frames": kinematics.get("tcp_frames"),
        "ik_targets": kinematics.get("ik_targets"),
        "joint_names": kinematics.get("joint_names"),
        "budget": spec.get("budget"),
        "success_signal": spec.get("success_signal"),
        "world": spec.get("world"),
    }


def _png_bytes(frame: np.ndarray) -> bytes:
    buffer = io.BytesIO()
    imageio.imwrite(buffer, np.asarray(frame, dtype=np.uint8), format="png")
    return buffer.getvalue()


class EfeTools:
    """Agent-facing access to one EFE episode."""

    def __init__(self, *, client: EfeEnvClient, state: EnvState) -> None:
        self.client = client
        self._state = state
        # The frames saved by the last observation, so a state record binds the
        # images the agent actually saw. A reset restarts the simulator step,
        # so the binding carries the episode index of the capture.
        self._frames: dict[str, str] = {}

    def latest_frames(self) -> dict[str, str]:
        """Return the artifact names saved by the last observation."""
        return dict(self._frames)

    @tool(readonly=True)
    def efe_spec(self) -> ToolResult:
        """Return the robot's action types, limits, cameras, proprioception, and kinematics.

        Call this once before acting. Action layouts differ between EFE tasks.
        """
        return ToolResult(data=summarize_spec(self.client.spec))

    @tool(readonly=True)
    def efe_observe(
        self,
        cameras: list[str] | None = None,
        depth: bool = False,
        resolution: Literal["full", "low"] = "full",
    ) -> ToolResult:
        """Return current camera frames and proprioception. Does not move the robot or advance time.

        Args:
            cameras: Camera names from efe_spec. Default: the task's default cameras.
            depth: Also return the last frame's depth map, in metres, as a .npy artifact.
            resolution: ``full`` returns the service frame; ``low`` shrinks each
                side by four. Multiply pixel coordinates from a low frame by 4
                before using them with camera metadata.
        """
        if resolution not in _RESOLUTION_FACTORS:
            raise ValueError(
                f"Unsupported resolution {resolution!r}; "
                f"use {sorted(_RESOLUTION_FACTORS)}"
            )
        result = self.client.observe(cameras=cameras, depth=depth)
        meta = result.meta
        factor = _RESOLUTION_FACTORS[resolution]
        # Keep the response order so each label stays with its image.
        cameras_shown = list(result.last)
        frames = {name: np.asarray(frame) for name, frame in result.last.items()}
        images = [_png_bytes(frame[::factor, ::factor]) for frame in frames.values()]
        sim_step = meta.get("step")
        data: dict[str, Any] = {
            "step": sim_step,
            "control_steps": meta.get("control_steps"),
            "sim_time_s": meta.get("sim_time_s"),
            "time_left_s": meta.get("time_left_s"),
            "done": meta.get("done"),
            "reason": meta.get("reason"),
            "cameras": cameras_shown,
            "image_downsample": factor,
            "proprio": proprio_map(self.client.spec, meta.get("proprio") or {}),
        }
        # Persist the full-resolution frames so a later view_env_state call,
        # or a post-run review, can inspect what the agent saw. ``resolution``
        # only affects the payload returned to the model.
        episode = self.client.reset_calls
        frame_paths: dict[str, str] = {}
        self._frames = {}
        for name, frame in frames.items():
            artifact = camera_artifact(name, episode, sim_step)
            saved = self._state.save(artifact, frame, step=None)
            if saved is not None:
                self._frames[name] = saved
                frame_paths[name] = str(self._state.artifact_path(saved, step=None))
        data["episode"] = episode
        if frame_paths:
            data["camera_paths"] = frame_paths
        if depth and result.last_depth:
            data["depth_units"] = "metres"
            data["depth_invalid"] = 0
            for name, frame in result.last_depth.items():
                artifact = depth_artifact(name, episode, sim_step)
                saved = self._state.save(artifact, np.asarray(frame), step=None)
                if saved is not None:
                    data.setdefault("depth_paths", {})[name] = str(
                        self._state.artifact_path(saved, step=None)
                    )
        return ToolResult(data=data, images=images)

    @tool(readonly=True)
    def efe_status(self) -> ToolResult:
        """Return the episode's call, control-step, and time counters, and whether it has ended."""
        status = self.client.status()
        data = {
            key: status.get(key)
            for key in (
                "step",
                "calls",
                "control_steps",
                "done",
                "reason",
                "elapsed_s",
            )
        }
        # Tasks with resets or a step limit add these fields; keep them when present.
        for key in ("episode", "resets_left", "step_limit", "steps_left"):
            if key in status:
                data[key] = status[key]
        return ToolResult(data=data)

    @tool(readonly=True)
    def efe_recover(self, timeout_s: int = 600) -> ToolResult:
        """Wait for a crashed simulator to resume the same episode, then report status.

        Use only after a connection error. A call that failed this way may have
        executed; read the status before sending anything else.

        Args:
            timeout_s: Maximum wait in seconds.
        """
        status = self.client.wait_ready(float(timeout_s))
        return ToolResult(
            data={
                key: status.get(key)
                for key in ("done", "reason", "step", "calls", "control_steps")
            }
        )

    @tool
    def efe_submit(
        self,
        action_type: str,
        actions: Annotated[
            list[list[float]],
            Field(description="Rows in the action layout order; [] for done."),
        ],
        dt: Annotated[
            float | None,
            Field(description="Seconds between rows. Default: one control period."),
        ] = None,
    ) -> ToolResult:
        """Send one action chunk to the robot. The service executes it, then returns.

        Prefer the high-level skills when they are available. Invalid actions are
        rejected before the simulator runs, so a rejected call does not count.
        """
        result = self.client.submit(action_type, actions, dt=dt)
        meta = result.meta
        return ToolResult(
            data={
                "action_type": action_type,
                "executed_rows": meta.get("executed_rows", len(actions)),
                "control_steps_chunk": meta.get("control_steps_chunk"),
                "done": meta.get("done"),
                "reason": meta.get("reason"),
                "clipped": meta.get("clipped"),
                "time_left_s": meta.get("time_left_s"),
            }
        )

    @tool
    def efe_reset(self, cameras: list[str] | None = None) -> ToolResult:
        """Abandon the running episode and restart from the frozen start state.

        Available only when efe_spec reports a reset budget. The service refuses
        a reset after success or when the budget is spent. Only the last episode
        is graded, so a reset throws away the current attempt.

        Args:
            cameras: Camera names for the new episode's first frame.
        """
        meta = self.client.reset(cameras=cameras)
        # The new episode restarts at step 0. Earlier frames stay bound to the
        # records that captured them and must not attach to the new episode.
        self._frames = {}
        return ToolResult(
            data={
                key: meta.get(key)
                for key in (
                    "episode",
                    "resets_left",
                    "step",
                    "control_steps",
                    "done",
                    "reason",
                )
            }
        )

    @tool
    def efe_end_episode(self) -> ToolResult:
        """End the episode now; the verifier grades the state it ended in.

        Available only when efe_spec reports ``success_signal="hidden"``: the
        service never reports success, so the agent decides when it is done.
        Ending before the goal is a failure.
        """
        status = self.client.finish()
        return ToolResult(
            data={
                key: status.get(key)
                for key in (
                    "ok",
                    "episode",
                    "resets_left",
                    "step",
                    "control_steps",
                    "done",
                    "reason",
                )
            }
        )


def dump_state(
    client: EfeEnvClient,
    state: EnvState,
    *,
    command: dict[str, Any],
    result: dict[str, Any],
    elapsed_s: float,
    frames: Mapping[str, str] | None = None,
) -> StepRecord:
    """Record the current proprioception without requesting camera frames.

    ``frames`` binds the record to the artifacts saved by the last
    observation, so a later ``view_env_state`` reads the images the agent saw
    at that point instead of searching by simulator step, which resets and
    repeats across episodes.
    """
    meta = client.last_meta
    if not meta.get("proprio") and not client.done:
        meta = client.observe(video="none").meta
    state_data = {
        "proprio": proprio_map(client.spec, meta.get("proprio") or {}),
        "episode": client.reset_calls,
        "step": meta.get("step"),
        "control_steps": meta.get("control_steps"),
        "sim_time_s": meta.get("sim_time_s"),
        "time_left_s": meta.get("time_left_s"),
        "done": meta.get("done"),
        "reason": meta.get("reason"),
        # The camera list and the bound frames tell view_env_state which
        # images belong to this record.
        "cameras": sorted((client.spec.get("cameras") or {}).keys()),
        "frames": dict(frames or {}),
    }
    with state.record_step(
        state=state_data,
        command=command,
        result=result,
        elapsed_s=elapsed_s,
        terminated=bool(meta.get("done")),
    ) as step:
        pass
    return state.get(step)


@tool(readonly=True, exclude=("state",))
def view_env_state(
    step: Annotated[int, Field(json_schema_extra={"default": -1})] = -1,
    *,
    state: EnvState,
) -> ToolResult:
    """Read a recorded state snapshot, including the command that produced it.

    Images are the frames saved by the last observation before the record was
    written, bound by artifact name when the record was captured. A record
    written before any observation has no images.
    """
    record = state.get(step)
    data = record.to_blob()
    images: list[bytes] = []
    frame_paths: dict[str, str] = {}
    for camera, name in (record.state.get("frames") or {}).items():
        if not state.exists(name, step=None):
            continue
        images.append(state.load_bytes(name, step=None))
        frame_paths[camera] = str(state.artifact_path(name, step=None))
    if frame_paths:
        data["camera_images"] = frame_paths
    return ToolResult(data=data, images=images)
