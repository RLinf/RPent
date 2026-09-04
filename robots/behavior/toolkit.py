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

"""Standard RPent Toolkit implementation for BEHAVIOR."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from robots.behavior.schemas import behavior_tool_specs_for_task
from robots.behavior.task_specs import get_task_spec
from robots.behavior.tools import BehaviorPrimitives
from rpent.dashboard.events import (
    DashboardEventSink,
    NullDashboardEventSink,
    ToolResultEvent,
)
from rpent.memory import MemoryManager
from rpent.session import EnvState, write_command_recipe_from_states
from rpent.tools import common
from rpent.tools.toolkit import Toolkit, ToolResult
from rpent.utils.templates import substitute


class BehaviorToolkit(Toolkit):
    """Expose BEHAVIOR primitives through the latest standard-main contract."""

    _FRAME_ARTIFACTS = {
        "head": "head_rgb.png",
        "left_wrist": "left_wrist_rgb.png",
        "right_wrist": "right_wrist_rgb.png",
    }

    def __init__(
        self,
        *,
        primitives_kwargs: dict[str, Any],
        dashboard_events: DashboardEventSink | None = None,
        memory: MemoryManager,
        config: Any = None,
        state_output_dir: str | Path | None = None,
    ) -> None:
        values = dict(primitives_kwargs)
        if config is not None:
            prompt_vars = dict(getattr(config, "prompt_vars", {}) or {})
            values.setdefault("task_name", prompt_vars.get("task_name"))
            values.setdefault("public_seed", prompt_vars.get("public_seed"))
            behavior_phase = prompt_vars.get("behavior_phase") or prompt_vars.get(
                "behavior_mode"
            )
            if behavior_phase is not None:
                values.setdefault("behavior_phase", behavior_phase)
            values.setdefault("max_episode_steps", prompt_vars.get("max_episode_steps"))
            values.setdefault("output_dir", getattr(config, "output_dir", None))
        output_dir = Path(
            values.get("output_dir") or getattr(config, "output_dir", Path.cwd())
        )
        self._run_output_dir = Path(getattr(config, "output_dir", output_dir))
        self._state_output_dir = Path(state_output_dir or output_dir)
        values["output_dir"] = self._state_output_dir
        self._recipe_tag = str(
            getattr(config, "recipe_tag", "")
            or get_task_spec(str(values.get("task_name") or "turning_on_radio")).tag(
                int(values.get("public_seed") or 0)
            )
        )

        super().__init__(
            dashboard_events=dashboard_events or NullDashboardEventSink(),
            state=EnvState(self._state_output_dir),
            memory=memory,
        )
        self._run_state = EnvState(self._run_output_dir)
        self._task_spec = get_task_spec(
            str(values.get("task_name") or "turning_on_radio")
        )
        values["episode_video_writer"] = self._state.open_video_writer(
            "episode.mp4",
            step=None,
            fps=20,
            max_frames=2000,
        )
        self._primitives = BehaviorPrimitives(**values)
        for spec in behavior_tool_specs_for_task(self._task_spec):
            if values.get("env") is None:
                continue
            if spec["name"] == "pi0_nav_pick" and values.get("model") is None:
                continue
            self.add_tool(spec["name"], spec, getattr(self._primitives, spec["name"]))
        finish_spec = next(
            spec for spec in common.TOOLS_SPEC if spec["name"] == "finish"
        )
        self.add_tool("finish", finish_spec, self._primitives.finish)

    @property
    def primitives(self) -> BehaviorPrimitives:
        return self._primitives

    def get_tools_spec(self) -> list[dict[str, Any]]:
        return substitute(
            [spec for spec, _ in self._tools.values()],
            variables={"output_dir": str(self._primitives.output_dir)},
        )

    def execute_tool(self, name: str, input_dict: dict[str, Any]) -> ToolResult:
        result = super().execute_tool(name, input_dict)
        if self._dashboard_result_has_frames(result.result):
            try:
                self._dashboard_events.emit(
                    ToolResultEvent(name=name, result=result.result)
                )
            except Exception:
                pass
        if (
            name == "finish"
            and isinstance(result.result, dict)
            and result.result.get("_finish") is True
        ):
            saved = self._state.save("terminal_receipt.json", result.result, step=None)
            if saved is None:
                raise RuntimeError("failed to write terminal_receipt.json")
        return result

    @staticmethod
    def _dashboard_result_has_frames(result: Any) -> bool:
        if not isinstance(result, dict):
            return False
        for key in (
            "_image_bytes",
            "_depth_image_bytes",
            "_image_left_wrist_bytes",
            "_depth_left_wrist_bytes",
            "_image_right_wrist_bytes",
            "_depth_right_wrist_bytes",
            "_frames_bytes",
        ):
            if result.get(key):
                return True
        for key in ("frames", "views", "images", "visual_review"):
            if isinstance(result.get(key), dict):
                return True
        return False

    def _save_observation_images(
        self, observation: dict[str, Any], *, step: int
    ) -> None:
        head = observation.get("main_images")
        wrists = observation.get("wrist_images")
        if head is not None:
            image = np.asarray(head)
            if image.ndim == 3:
                self._state.save("head_rgb.png", image[..., :3], step=step)
        if wrists is not None:
            wrist_array = np.asarray(wrists)
            if wrist_array.ndim == 4 and wrist_array.shape[0] >= 2:
                self._state.save(
                    "left_wrist_rgb.png", wrist_array[0, ..., :3], step=step
                )
                self._state.save(
                    "right_wrist_rgb.png", wrist_array[1, ..., :3], step=step
                )

    def get_env_state(
        self,
        *,
        command: dict[str, Any],
        result: dict[str, Any],
        elapsed_s: float,
    ) -> dict[str, Any]:
        snapshot = self._primitives.snapshot()
        terminated = bool(snapshot.get("task_success"))
        with self._state.record_step(
            state=snapshot,
            terminated=terminated,
            truncated=False,
            command=command,
            result=result,
            elapsed_s=elapsed_s,
        ) as step:
            observation = self._primitives.current_observation
            if isinstance(observation, dict):
                self._save_observation_images(observation, step=step)
            step_idx = step
        record = self._state.get(step_idx)
        return {
            **snapshot,
            "step_idx": step_idx,
            "artifacts": sorted(record.artifacts),
            "command": command,
            "result": result,
        }

    def close(self) -> None:
        """Release clients/transports only; never synthesize task success."""

        self._primitives.shutdown()

    def solved(self) -> bool:
        return self._primitives.solved()

    def write_recipe(self, recipe_tag: str) -> str | None:
        """Write an idempotent public recipe JSONL for a solved session."""

        if not self.solved():
            return None
        if not isinstance(recipe_tag, str) or not recipe_tag.strip():
            recipe_tag = self._task_spec.tag(self._primitives.public_seed)
        return write_command_recipe_from_states(
            self._state,
            recipe_tag.strip(),
            output_state=self._run_state,
        )


__all__ = ["BehaviorToolkit"]
