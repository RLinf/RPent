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
from robots.behavior.tools import _PUBLIC_IMAGE_BYTE_FIELDS, BehaviorPrimitives
from robots.behavior.video import VideoArtifactWriter
from rpent.dashboard.events import (
    DashboardEventSink,
    NullDashboardEventSink,
)
from rpent.memory import MemoryManager
from rpent.robots.runtime import stop_owned_daemons
from rpent.session import EnvState
from rpent.tools import common
from rpent.tools.toolkit import Toolkit, ToolResult
from rpent.utils.daemon import ProcessDaemon
from rpent.utils.templates import substitute


class BehaviorToolkit(Toolkit):
    """Expose BEHAVIOR primitives through the latest standard-main contract."""

    def __init__(
        self,
        *,
        runtime_kwargs: dict[str, Any],
        dashboard_events: DashboardEventSink | None = None,
        memory: MemoryManager,
        config: Any = None,
        state_output_dir: str | Path | None = None,
        owned_daemons: dict[str, ProcessDaemon] | None = None,
    ) -> None:
        values = dict(runtime_kwargs)
        self._owned_daemons = owned_daemons or {}
        if config is not None:
            prompt_vars = dict(getattr(config, "prompt_vars", {}) or {})
            values.setdefault("task_name", prompt_vars.get("task_name"))
            values.setdefault("public_seed", prompt_vars.get("public_seed"))
            behavior_phase = prompt_vars.get("behavior_phase") or prompt_vars.get(
                "mode"
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

        super().__init__(
            dashboard_events=dashboard_events or NullDashboardEventSink(),
            state=EnvState(self._state_output_dir),
            memory=memory,
        )
        self._run_state = EnvState(self._run_output_dir)
        values["memory_dir"] = memory.root
        self._task_spec = get_task_spec(
            str(values.get("task_name") or "turning_on_radio")
        )
        values["episode_video_writer"] = VideoArtifactWriter(
            self._state,
            name="episode.mp4",
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
        if self._primitives.current_observation is not None:
            self.get_env_state(command=None, result={}, elapsed_s=0.0)
            self._publish_step(self._state.latest_record())

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
        if name == "observe" and self._primitives.current_observation is not None:
            self.get_env_state(command=None, result=result.result, elapsed_s=0.0)
            self._publish_step(self._state.latest_record())
        if (
            name == "finish"
            and isinstance(result.result, dict)
            and result.result.get("_finish") is True
        ):
            saved = self._state.save("terminal_receipt.json", result.result, step=None)
            if saved is None:
                raise RuntimeError("failed to write terminal_receipt.json")
        payload = dict(result.result)
        images = [
            payload.pop(key)
            for key in (
                "_image_bytes",
                "_depth_image_bytes",
                "_image_left_wrist_bytes",
                "_depth_left_wrist_bytes",
                "_image_right_wrist_bytes",
                "_depth_right_wrist_bytes",
            )
            if key in payload
        ]
        if images:
            result = ToolResult(name=name, result=payload, call_id=result.call_id)
            for data in images:
                result.content_blocks.extend(
                    ToolResult(name=name, result={"_image_bytes": data}).content_blocks[
                        1:
                    ]
                )
        return result

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
        command: dict[str, Any] | None,
        result: dict[str, Any],
        elapsed_s: float,
    ) -> dict[str, Any]:
        snapshot = self._primitives.snapshot()
        terminated = bool(snapshot.get("task_success"))
        truncated = (
            bool(result.get("truncated")) or result.get("stop_reason") == "truncated"
        )
        result = {**result, "official_success": self.solved()}
        # Image payloads belong to ToolResult blocks, not the JSON step log.
        images = {
            key: result.pop(key) for key in _PUBLIC_IMAGE_BYTE_FIELDS if key in result
        }
        with self._state.record_step(
            state=snapshot,
            terminated=terminated,
            truncated=truncated,
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
            **images,
        }

    def close(self) -> None:
        """Release clients/transports only; never synthesize task success."""

        try:
            self._primitives.shutdown()
        finally:
            stop_owned_daemons(self._owned_daemons, self._dashboard_events)
            self._owned_daemons.clear()

    def solved(self) -> bool:
        return self._primitives.solved()

    def write_recipe(self, recipe_tag: str) -> str | None:
        """Write an idempotent public recipe JSONL for a solved session."""

        if not self.solved():
            return None
        if not isinstance(recipe_tag, str) or not recipe_tag.strip():
            recipe_tag = self._task_spec.tag(self._primitives.public_seed)
        try:
            terminal_result = self._state.load("terminal_receipt.json", step=None)
        except FileNotFoundError as exc:
            raise RuntimeError(
                "solved BEHAVIOR session is missing terminal_receipt.json; "
                "refusing to export recipe"
            ) from exc
        commands = [
            record.command
            for record in self._state.records()
            if isinstance(record.command, dict)
            and record.command.get("action") is not None
            and not (isinstance(record.result, dict) and record.result.get("error"))
        ]
        if not commands:
            return None
        name = f"{recipe_tag.strip()}_recipe.jsonl"
        if self._run_state.save(name, commands, step=None) is None:
            raise RuntimeError(f"failed to write {name}")
        from robots.behavior.dino_v2.index import prepare_evidence_pack

        self.memory.task_artifacts = prepare_evidence_pack(
            self._run_output_dir,
            recipe_tag.strip(),
            self._state,
            terminal_result,
        )
        return str(self._run_state.artifact_path(name, step=None))


__all__ = ["BehaviorToolkit"]
