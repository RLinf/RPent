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
from typing import Annotated, Any

import imageio.v2 as imageio
import numpy as np
from pydantic import Field

from robots.efe.env_client import EfeEnvClient
from rpent.session import EnvState, StepRecord
from rpent.tools import ToolResult, tool


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
    ) -> ToolResult:
        """Return current camera frames and proprioception. Does not move the robot or advance time.

        Args:
            cameras: Camera names from efe_spec. Default: the task's default cameras.
            depth: Also return the last frame's depth map, in metres, as a .npy artifact.
        """
        result = self.client.observe(cameras=cameras, depth=depth)
        meta = result.meta
        # Keep the response order so each label stays with its image.
        cameras_shown = list(result.last)
        images = [_png_bytes(frame) for frame in result.last.values()]
        data: dict[str, Any] = {
            "step": meta.get("step"),
            "control_steps": meta.get("control_steps"),
            "sim_time_s": meta.get("sim_time_s"),
            "time_left_s": meta.get("time_left_s"),
            "done": meta.get("done"),
            "reason": meta.get("reason"),
            "cameras": cameras_shown,
            "proprio": proprio_map(self.client.spec, meta.get("proprio") or {}),
        }
        if depth and result.last_depth:
            data["depth_units"] = "metres"
            data["depth_invalid"] = 0
            for name, frame in result.last_depth.items():
                saved = self._state.save(
                    f"depth_{name}.npy", np.asarray(frame), step=None
                )
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
) -> StepRecord:
    """Record the current proprioception without requesting camera frames."""
    meta = client.last_meta
    if not meta.get("proprio") and not client.done:
        meta = client.observe(video="none").meta
    state_data = {
        "proprio": proprio_map(client.spec, meta.get("proprio") or {}),
        "step": meta.get("step"),
        "control_steps": meta.get("control_steps"),
        "sim_time_s": meta.get("sim_time_s"),
        "time_left_s": meta.get("time_left_s"),
        "done": meta.get("done"),
        "reason": meta.get("reason"),
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
    """Read a recorded state snapshot, including the command that produced it."""
    return ToolResult(data=state.get(step).to_blob())
