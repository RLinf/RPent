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

"""BEHAVIOR tool registration and observation capture using the RPent toolkit."""

from __future__ import annotations

from dataclasses import replace
from functools import partial
from pathlib import Path
from typing import TYPE_CHECKING, Any

from pydantic import Field, create_model

from robots.behavior import tools
from robots.behavior.primitives import BehaviorPrimitives
from rpent.dashboard.events import DashboardEventSink
from rpent.session import EnvState
from rpent.tools import Tool, ToolResult, iter_tools
from rpent.tools.toolkit import Toolkit
from rpent.utils.logging import get_output_dir

if TYPE_CHECKING:
    from rpent.memory.manager import MemoryManager


class BehaviorToolkit(Toolkit):
    """Register BEHAVIOR tools around clients owned by the runtime."""

    def __init__(
        self,
        *,
        runtime_kwargs: dict[str, Any],
        dashboard_events: DashboardEventSink,
        memory: MemoryManager,
        state_output_dir: Path | str | None = None,
    ) -> None:
        super().__init__(
            dashboard_events=dashboard_events,
            state=EnvState(Path(state_output_dir or get_output_dir())),
            memory=memory,
        )
        primitive_kwargs = dict(runtime_kwargs)
        self._tool_allowlist = primitive_kwargs.pop("tool_allowlist", None)
        sam3_client = primitive_kwargs.pop("sam3_client", None)
        self._segmenter = None
        if sam3_client is not None:
            self._segmenter = tools.SavedImageSegmenter(sam3_client)
        self._primitives = BehaviorPrimitives(
            check_cancelled=self.raise_if_cancelled, **primitive_kwargs
        )
        self._register_behavior_tools()
        self.get_env_state(
            command={"action": "observe"},
            result={},
            elapsed_s=0.0,
        )
        record = self.state.latest_record()
        if record is not None:
            self._publish_step(record)

    def _register_behavior_tools(self) -> None:
        readers = tuple(iter_tools(tools))
        primitives = tuple(iter_tools(self._primitives))
        available = {definition.name for definition in (*readers, *primitives)}
        allowed = self._tool_allowlist
        if allowed is None:
            allowed = {"view_env_state", "vla_act", "finish"}
        if (
            not isinstance(allowed, (list, set, frozenset))
            or any(not isinstance(name, str) for name in allowed)
            or set(allowed) - available
            or not {"view_env_state", "finish"}.issubset(allowed)
        ):
            raise ValueError(
                "tool_allowlist must name available BEHAVIOR tools and retain "
                "view_env_state and finish"
            )
        if "segment" in allowed and self._segmenter is None:
            raise ValueError("segment requires a connected RPent SAM3 client")
        for definition in readers:
            if definition.name not in allowed:
                continue
            resources = {} if definition.name == "finish" else {"state": self.state}
            if definition.name == "segment":
                resources["segmenter"] = self._segmenter
            self.add_tool(
                definition.with_handler(partial(definition, **resources)),
                replace=definition.name == "finish",
            )
        for definition in primitives:
            if definition.name not in allowed:
                continue
            if definition.name == "vla_act":
                definition = self._configure_vla_tool(definition)
            self.add_tool(definition.with_handler(partial(self._step, definition.name)))

    def _configure_vla_tool(self, definition: Tool) -> Tool:
        """Expose the run's action limit in the same schema used for validation."""
        limit = self._primitives.max_chunks_per_call
        if limit is not None:
            args_schema = create_model(
                "BehaviorVLAArgs",
                __base__=definition.args_schema,
                chunks=(
                    int,
                    Field(default=1, ge=1, le=limit),
                ),
            )
            schema = definition.input_schema
            schema["properties"]["chunks"]["maximum"] = limit
            definition = replace(
                definition,
                args_schema=args_schema,
                _input_schema=schema,
                description=definition.description
                + f" This run allows at most {limit} chunks per call.",
            )
        if self._primitives.instruction_mode == "main":
            definition = replace(
                definition,
                description=definition.description
                + " Keep the full configured main-task instruction; omit prompt. "
                "A different prompt is rejected.",
            )
        return definition

    def _step(self, name: str, **kwargs: Any) -> ToolResult:
        self.raise_if_cancelled()
        return getattr(self._primitives, name)(**kwargs)

    def get_env_state(
        self,
        *,
        command: dict[str, Any],
        result: dict[str, Any],
        elapsed_s: float,
    ) -> ToolResult:
        """Capture agent-visible observations through the existing EnvState store."""
        if (
            result.get("session_ended")
            or result.get("observation_fresh") is False
            or result.get("observation_capture_source") == "motion_call_cache"
        ):
            raw = self._primitives.last_observation
            if raw is None:
                raise RuntimeError("ended BEHAVIOR session has no cached observation")
        else:
            raw = self._primitives.env.get_observation()
            self._primitives.last_observation = raw
        snapshot = tools.prepare_behavior_observation(
            raw,
            task_id=self._primitives.task_id,
            task_language=self._primitives.task_language,
        )
        record = tools.save_observation_record(
            snapshot,
            env_state=self.state,
            log={"command": command, "result": result, "elapsed_s": elapsed_s},
        )
        return tools.view_env_state(record.step_idx, state=self.state)

    def solved(self) -> bool:
        """Do not publish successful recipes without an official result channel."""
        return False
