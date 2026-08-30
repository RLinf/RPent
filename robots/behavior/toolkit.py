"""Standard RPent Toolkit implementation for BEHAVIOR."""

from __future__ import annotations

import json
import os
import tempfile
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
from rpent.session import EnvState
from rpent.tools import common
from rpent.tools.toolkit import Toolkit, ToolResult
from rpent.utils.templates import substitute


class BehaviorToolResult(ToolResult):
    """BEHAVIOR result wrapper.

    The base ``ToolResult`` already supports public PNG byte payloads and finish
    detection.  This subclass exists as a stable BEHAVIOR-facing type.
    """


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
        memory: Any = None,
        config: Any = None,
        video_path: str | Path | None = None,
    ) -> None:
        values = dict(primitives_kwargs)
        if config is not None:
            prompt_vars = dict(getattr(config, "prompt_vars", {}) or {})
            values.setdefault("task_name", prompt_vars.get("task_name"))
            values.setdefault("public_seed", prompt_vars.get("public_seed"))
            values.setdefault(
                "behavior_phase",
                prompt_vars.get("behavior_phase", prompt_vars.get("behavior_mode")),
            )
            values.setdefault("max_episode_steps", prompt_vars.get("max_episode_steps"))
            values.setdefault("output_dir", getattr(config, "output_dir", None))
        output_dir = Path(
            values.get("output_dir") or getattr(config, "output_dir", Path.cwd())
        )
        values["output_dir"] = output_dir
        values["video_path"] = (
            Path(video_path) if video_path is not None else output_dir / "episode.mp4"
        )

        if memory is None:
            from rpent.memory import MemoryManager

            memory = MemoryManager(root=output_dir / "behavior_memory_empty")
        super().__init__(
            dashboard_events=dashboard_events or NullDashboardEventSink(),
            state=EnvState(output_dir),
            memory=memory,
        )
        self._task_spec = get_task_spec(
            str(values.get("task_name") or "turning_on_radio")
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

    def execute_tool(self, name: str, input_dict: dict[str, Any]) -> BehaviorToolResult:
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
            receipt_path = self._primitives.output_dir / "terminal_receipt.json"
            receipt_path.parent.mkdir(parents=True, exist_ok=True)
            fd, temporary_name = tempfile.mkstemp(
                prefix=".terminal_receipt.", suffix=".tmp", dir=receipt_path.parent
            )
            try:
                with os.fdopen(fd, "w", encoding="utf-8") as stream:
                    json.dump(
                        result.result, stream, indent=2, sort_keys=True, default=str
                    )
                    stream.write("\n")
                os.replace(temporary_name, receipt_path)
            finally:
                try:
                    os.unlink(temporary_name)
                except FileNotFoundError:
                    pass
        return BehaviorToolResult(
            name=result.name,
            result=result.result,
            call_id=result.call_id,
        )

    @staticmethod
    def _dashboard_result_has_frames(result: Any) -> bool:
        if not isinstance(result, dict):
            return False
        for key in (
            "_image_bytes",
            "_image_cam_bytes",
            "_image_nav_bytes",
            "_image_wrist_bytes",
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
        """Write an idempotent best-effort public recipe JSONL."""

        if not isinstance(recipe_tag, str) or not recipe_tag.strip():
            recipe_tag = self._task_spec.tag(self._primitives.public_seed)
        records: list[dict[str, Any]] = []
        for record in self._state.records():
            command = record.command or {}
            if command.get("action") in self._tools:
                records.append(
                    {
                        "step_idx": record.step_idx,
                        "command": command,
                        "result": record.result or {},
                        "terminated": record.terminated,
                        "truncated": record.truncated,
                        "elapsed_s": record.elapsed_s,
                    }
                )
        if not records:
            records = self._primitives.recipe_records()
        name = f"recipe_{recipe_tag.strip()}.jsonl"
        self._state.save(name, records, step=None)
        path = self._state.artifact_path(name, step=None)
        if not path.exists():
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(
                "".join(json.dumps(item, default=str) + "\n" for item in records),
                encoding="utf-8",
            )
        return str(path)


__all__ = ["BehaviorToolkit", "BehaviorToolResult"]
