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

"""VLABench toolkit: the common tools plus pose, gripper and perception tools."""

from __future__ import annotations

from functools import partial
from pathlib import Path
from typing import TYPE_CHECKING, Annotated, Any

import numpy as np
from pydantic import Field

from robots.vlabench.native_tools import VLABenchPrimitives
from robots.vlabench.observations import (
    VLABenchGeometryObservations,
    artifact_names,
)
from robots.vlabench.sim.contracts import VLABENCH_CAMERA_NAMES
from rpent.session import EnvState
from rpent.tools import ToolResult, iter_tools, tool
from rpent.tools.toolkit import Toolkit
from rpent.utils.logging import get_output_dir
from toolkits.back_project import back_project

if TYPE_CHECKING:
    from rpent.dashboard.events import DashboardEventSink
    from rpent.memory.manager import MemoryManager
    from rpent.robots.robot_spec import RunConfig


def _plain(value: Any) -> Any:
    """Convert NumPy values to JSON-friendly lists and scalars."""
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, dict):
        return {key: _plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    return value


class VLABenchToolkit(Toolkit):
    """Common RPent tools plus VLABench motion, grounding and observation tools."""

    def __init__(
        self,
        *,
        runtime_kwargs: dict[str, Any],
        dashboard_events: DashboardEventSink,
        memory: MemoryManager,
        state_output_dir: Path | str | None = None,
    ) -> None:
        state = EnvState(Path(state_output_dir or get_output_dir()))
        super().__init__(dashboard_events=dashboard_events, state=state, memory=memory)
        self._primitives = VLABenchPrimitives(
            env=runtime_kwargs["env"], check_cancelled=self.raise_if_cancelled
        )
        self._observations = VLABenchGeometryObservations(state)
        for definition in iter_tools(self._primitives):
            handler = (
                definition
                if definition.name == "finish"
                else partial(self._step, definition.name)
            )
            self.add_tool(
                definition.with_handler(handler),
                replace=definition.name == "finish",
            )
        self.add_tool(
            back_project.with_handler(
                partial(back_project, observations=self._observations)
            )
        )
        self.add_tool(self.render)
        self.add_tool(self.view_env_state)
        # The environment service already reset the episode on connect.
        self.get_env_state(
            command={"action": "reset"}, result={"success": True}, elapsed_s=0.0
        )
        record = self._state.latest_record()
        if record is not None:
            self._publish_step(record)

    def _step(self, name: str, **kwargs: Any) -> ToolResult:
        self.raise_if_cancelled()
        return getattr(self._primitives, name)(**kwargs)

    @tool
    def render(self) -> ToolResult:
        """Capture a fresh synchronized observation of all four cameras."""
        self.raise_if_cancelled()
        return ToolResult(data={"success": True})

    @tool(readonly=True)
    def view_env_state(
        self, step: Annotated[int, Field(json_schema_extra={"default": -1})] = -1
    ) -> ToolResult:
        """Read one recorded step and embed its camera images. Step -1 selects the latest entry.

        Args:
            step: Step number; 0 = initial, -1 = latest.
        """
        return self._view(step)

    def _view(self, step: int) -> ToolResult:
        try:
            record = self._state.get(step)
        except Exception as error:
            return ToolResult(data={"error": f"state step not available: {error}"})
        images: list[bytes] = []
        for camera in VLABENCH_CAMERA_NAMES:
            rgb_name = artifact_names(camera)[0]
            if rgb_name in record.artifacts:
                images.append(self._state.load_bytes(rgb_name, step=record.step_idx))
        data = {
            "step": record.step_idx,
            "terminated": record.terminated,
            "truncated": record.truncated,
            "state": record.state,
            "artifacts": sorted(record.artifacts),
            "task_language": record.extras.get("task_language"),
            "log": {
                "command": record.command,
                "result": record.result,
                "elapsed_s": record.elapsed_s,
            },
        }
        return ToolResult(data=data, images=images)

    def get_env_state(
        self,
        *,
        command: dict[str, Any],
        result: dict[str, Any],
        elapsed_s: float,
    ) -> ToolResult:
        """Record RGB, metric depth and calibration of every camera as one step."""
        env = self._primitives.env
        snapshot = env.sensor_snapshot(cameras=list(VLABENCH_CAMERA_NAMES))
        status = self._primitives.status()
        latest = self._state.latest_step
        step_idx = 0 if latest is None else latest + 1
        language = env.get_task_language()
        state = {
            "step_idx": step_idx,
            "task_language": language,
            "observation_id": snapshot["observation_id"],
            "robot_state": _plain(snapshot["robot"]["model"]),
            "episode_status": status,
            "cameras": list(VLABENCH_CAMERA_NAMES),
        }
        with self._state.record_step(
            state=state,
            terminated=bool(status["terminated"]),
            truncated=bool(status["truncated"]),
            command=command,
            result=result,
            elapsed_s=elapsed_s,
            extras={"task_language": language},
        ) as recorded:
            for frame in snapshot["cameras"]:
                rgb_name, depth_name, meta_name = artifact_names(frame["camera"])
                self._state.save(rgb_name, frame["rgb_top_down"], step=recorded)
                self._state.save(depth_name, frame["depth_meters"], step=recorded)
                self._state.save(meta_name, _plain(frame["calibration"]), step=recorded)
        return self._view(step_idx)

    def solved(self) -> bool:
        """Return the native VLABench task success flag."""
        return self._primitives.status().get("task_success") is True

    def evaluation_metrics(self, *, success: bool | None) -> dict[str, Any]:
        """Collect native action count and episode budget status for result.json."""
        del success
        metrics: dict[str, Any] = {"native_actions": self._primitives.native_actions}
        try:
            metrics["episode_status"] = self._primitives.status()
        except Exception as exc:
            metrics["status_error"] = f"{type(exc).__name__}: {exc}"
        return metrics


def get_toolkit(
    *,
    runtime_kwargs: dict[str, Any],
    dashboard_events: DashboardEventSink,
    config: RunConfig,
    mode: str = "evaluation",
    attempts_per_session: int = 0,
    state_output_dir: Path | str | None = None,
    **_: Any,
) -> VLABenchToolkit:
    """Return the VLABench toolkit for the current session."""
    from rpent.memory import MemoryManager
    from rpent.utils.config import get_memory_dir

    del attempts_per_session
    if mode != "evaluation":
        raise ValueError("VLABench supports evaluation mode only")
    memory = MemoryManager(
        root=config.prompt_vars.get("memory_dir") or get_memory_dir("vlabench"),
        memory_access="read_only",
    )
    return VLABenchToolkit(
        runtime_kwargs=runtime_kwargs,
        dashboard_events=dashboard_events,
        memory=memory,
        state_output_dir=state_output_dir or config.output_dir,
    )
