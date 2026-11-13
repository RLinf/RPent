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

"""RoboLab observation and policy tools."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated, Any

import numpy as np
from pydantic import Field

from rpent.dashboard.events import DashboardEventSink
from rpent.memory import MemoryManager
from rpent.session import EnvState
from rpent.tools import ToolResult, tool
from rpent.tools.direct_action import direct_action_tool
from rpent.tools.toolkit import Toolkit


class RoboLabToolkit(Toolkit):
    """Expose native RoboLab observation and bounded DROID control."""

    def __init__(
        self,
        *,
        env: Any,
        model: Any | None,
        dashboard_events: DashboardEventSink,
        memory: MemoryManager,
        output_dir: Path,
    ) -> None:
        self.env = env
        self.model = model
        state = EnvState(output_dir)
        super().__init__(dashboard_events=dashboard_events, state=state, memory=memory)
        self.add_tool(self.view_env_state)
        self.add_tool(direct_action_tool(env.action_specs, self._execute_action))
        if model is not None:
            self.add_tool(self.pi05_act)
        self._capture(command=None, result=None, elapsed_s=None)
        self._publish_step(self.state.latest_record())

    def _capture(
        self,
        *,
        command: dict[str, Any] | None,
        result: dict[str, Any] | None,
        elapsed_s: float | None,
    ) -> None:
        obs = self.env.last_obs
        status = self.env.get_status()
        with self.state.record_step(
            state={"robot_state": np.asarray(obs["states"]).tolist(), **status},
            terminated=bool(status["terminated"]),
            truncated=bool(status["truncated"]),
            command=command,
            result=result,
            elapsed_s=elapsed_s,
            extras={"task_language": obs["task_descriptions"]},
        ) as step:
            self.state.save(
                "main.png",
                self.env.render_camera("over_shoulder_left_camera"),
                step=step,
            )
            self.state.save("wrist.png", self.env.render_camera("wrist_cam"), step=step)

    def get_env_state(
        self,
        *,
        command: dict[str, Any],
        result: dict[str, Any],
        elapsed_s: float,
    ) -> ToolResult:
        self._capture(command=command, result=result, elapsed_s=elapsed_s)
        return self.view_env_state(-1)

    @tool(readonly=True)
    def view_env_state(
        self,
        step: Annotated[int, Field(json_schema_extra={"default": -1})] = -1,
    ) -> ToolResult:
        """Read a saved RoboLab state and its external and wrist RGB images.

        Args:
            step: Zero selects the initial state; -1 selects the latest state.
        """
        try:
            record = self.state.get(step)
        except (KeyError, LookupError, ValueError) as exc:
            return ToolResult(data={"error": str(exc)})
        return ToolResult(
            data={
                "step": record.step_idx,
                "task_language": record.extras["task_language"],
                "terminated": record.terminated,
                "truncated": record.truncated,
                "state": record.state,
                "artifacts": sorted(record.artifacts),
            },
            images=[
                self.state.load_bytes(name, step=record.step_idx)
                for name in ("main.png", "wrist.png")
                if name in record.artifacts
            ],
        )

    def _execute_action(self, values: list[float]) -> ToolResult:
        """Execute one action validated against the connected environment."""
        self.raise_if_cancelled()
        action = self.env.validate_action(values)
        _, _, terminated, truncated, info = self.env.step(action)
        return ToolResult(
            data={
                "executed_steps": 1,
                "terminated": terminated,
                "truncated": truncated,
                **info,
            }
        )

    @tool()
    def pi05_act(
        self,
        steps: Annotated[int, Field(ge=1, le=15, json_schema_extra={"default": 5})] = 5,
    ) -> ToolResult:
        """Predict a Pi0.5 DROID action chunk from the latest observation and execute at most the requested steps. Inspect the new state before continuing.

        Args:
            steps: Number of actions to execute from one prediction, from 1 through 15.
        """
        if self.model is None:
            raise RuntimeError("Pi0.5 is not connected")
        self.raise_if_cancelled()
        actions = np.asarray(self.model.predict(self.env.last_obs), dtype=np.float32)
        if (
            actions.ndim != 2
            or actions.shape[1] != 8
            or len(actions) < steps
            or not np.isfinite(actions).all()
        ):
            raise ValueError(f"Pi0.5 must return at least {steps} DROID actions [N,8]")
        actions[:, -1] = (actions[:, -1] > 0.5).astype(np.float32)
        executed = 0
        for action in actions[:steps]:
            self.raise_if_cancelled()
            self.env.step(action)
            executed += 1
            if self.env.terminated or self.env.truncated:
                break
        return ToolResult(data={"executed_steps": executed, **self.env.last_info})

    def solved(self) -> bool:
        """Use the benchmark's native success result."""
        return self.env.get_status()["success"] is True
