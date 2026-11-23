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

"""One episode conversation with explicit perception and measured action feedback."""

from __future__ import annotations

import os
import threading
import time
from dataclasses import replace
from pathlib import Path
from typing import Any

from robots.rl.contracts import CAMERAS, EPISODE_CONTRACT_VERSION
from robots.rl.images import NATIVE_SIZE
from robots.rl.perception import RecordedPerception
from robots.rl.prompt_bundle import validate_workflow
from rpent.dashboard.events import DashboardEventSink
from rpent.memory import MemoryManager
from rpent.session import EnvState
from rpent.tools import ToolResult
from rpent.tools.toolkit import Toolkit
from rpent.utils.serialization import write_json


def primitive_owner(environment: str, world_cord: str = "base"):
    """Select the same primitive owner class for local and remote execution."""
    if environment == "libero":
        from robots.rl.tools import LiberoRLPrimitives

        return LiberoRLPrimitives
    if environment == "robodojo":
        from robots.rl.robodojo import RoboDojoPrimitives

        return RoboDojoPrimitives
    from robots.rl.backends import (
        RoboCasaGeneralPrimitives,
        RoboTwinGeneralPrimitives,
        RoboTwinPrimitives,
    )

    if environment == "robocasa":
        return RoboCasaGeneralPrimitives
    if environment == "robotwin":
        return (
            RoboTwinGeneralPrimitives if world_cord == "general" else RoboTwinPrimitives
        )
    raise ValueError(f"Unsupported episode environment: {environment}")


def uses_native_1024(environment: str, world_cord: str = "base") -> bool:
    """Return whether the selected environment renders native 1024x1024 RGB-D."""
    return environment == "libero" or (
        environment == "robocasa" and world_cord == "general"
    )


def episode_tools(
    toolkit, environment, world_cord, units, max_actions, perception=None
):
    """Build the same model schema without importing native adapters on the client."""
    from robots.rl.actions import ActionBatch, DualArmBatch, MobileBatch
    from robots.rl.contracts import coordinate_definition, strict_tool
    from robots.rl.units import model_tool

    owner = RecordedPerception if perception is None else perception
    batch_type = {
        "libero": ActionBatch,
        "robocasa": MobileBatch,
        "robotwin": DualArmBatch,
        "robodojo": DualArmBatch,
    }[environment]
    batch = batch_type(toolkit).execute_actions
    schema = batch.input_schema
    schema["properties"]["actions"]["maxItems"] = max_actions
    batch = replace(
        batch,
        _input_schema=schema,
        description=batch.description.replace("1..K", f"1..{max_actions}"),
    )
    for declaration in (owner.view_env, owner.segment, owner.back_project, batch):
        yield model_tool(
            strict_tool(
                coordinate_definition(declaration, world_cord),
                cameras=CAMERAS[environment],
            ),
            units,
        )


class RLToolkit(Toolkit):
    """Expose explicit perception and ordered actions over one simulator scene."""

    planner_builtin_tools = False
    api_environment_context = False

    def __init__(
        self,
        *,
        runtime_kwargs: dict[str, Any],
        dashboard_events: DashboardEventSink,
        memory: MemoryManager,
        observation_mode: str,
        output_dir: Path | str,
        environment: str = "libero",
        workflow: str = "episode",
        image_size: int = 1024,
        action_mode: str = "batch",
        max_actions_per_call: int = 3,
        reasoning_history: int = -1,
        image_history: int = -1,
        perception_mode: str = "single",
        max_observe_calls: int = 8,
        observe_concurrency: int = 8,
        world_cord: str = "base",
        position_unit: str = "m",
        reasoning: bool = True,
    ) -> None:
        if observation_mode != "adaptive":
            raise ValueError("rl supports only adaptive observations")
        validate_workflow(environment, workflow)
        from robots.rl.contracts import (
            validate_coordinates,
            validate_image_size,
            validate_options,
            validate_perception_options,
        )

        validate_coordinates(environment, world_cord)
        from robots.rl.units import PositionUnits

        self.units = PositionUnits(position_unit)
        self.position_unit = position_unit
        self.world_cord, self.reasoning = world_cord, reasoning
        validate_image_size(environment, world_cord, image_size)
        validate_options(
            environment,
            workflow,
            action_mode,
            max_actions_per_call,
            reasoning_history,
            image_history,
            world_cord=world_cord,
        )
        validate_perception_options(
            workflow, perception_mode, max_observe_calls, observe_concurrency
        )
        self.perception_mode = perception_mode
        self.max_observe_calls, self.observe_concurrency = (
            max_observe_calls,
            observe_concurrency,
        )
        self.action_mode, self.max_actions_per_call = action_mode, max_actions_per_call
        self.reasoning_history, self.image_history = reasoning_history, image_history
        self.output_dir = Path(output_dir)
        self.image_size = image_size
        self.workflow_stats = {"perception_seconds": 0.0}
        self._started = time.perf_counter()
        self._native_success_seconds = None
        state = EnvState(output_dir)
        super().__init__(dashboard_events=dashboard_events, state=state, memory=memory)
        self._images_sent = 0
        self._images_lock = threading.Lock()
        self.environment, self.workflow = environment, workflow
        primitive_type = primitive_owner(environment, world_cord)
        native_options = dict(runtime_kwargs)
        if environment in {"libero", "robodojo"}:
            native_options.update(world_cord=world_cord)
        elif environment == "robocasa":
            native_options.update(world_cord=world_cord)
        self._primitives = primitive_type(
            state=state, check_cancelled=self.raise_if_cancelled, **native_options
        )
        if uses_native_1024(environment, world_cord) and image_size != NATIVE_SIZE:
            self._primitives.display_size = image_size
        self.action_error = None
        self._task = self._primitives.env.get_task_language()
        self._publish_step(self._primitives.record_observation())
        self._recorded_env_steps = 0
        self.add_tools(
            episode_tools(
                self,
                environment,
                world_cord,
                self.units,
                max_actions_per_call,
                perception=self._primitives,
            )
        )

    def frame_version(self, *, refresh: bool = False) -> int:
        """Read the measured frame under the episode's exclusive action lock."""
        if refresh and self.environment == "robocasa":
            self._primitives._refresh_coordinates()
        return getattr(getattr(self._primitives, "coordinates", None), "version", 0)

    def _model_feedback(self, result: ToolResult) -> ToolResult:
        if self.world_cord == "general":
            result.data["coordinates"] = self._primitives.coordinates.description()
        if self.position_unit != "m":
            result.data = self.units.payload(result.data)
        result.data["position_unit"] = self.position_unit
        return result

    def _register_common_tools(self) -> None:
        """Register no file tools; rollouts observe only through robot tools."""

    def _robot_state(self) -> dict[str, Any]:
        if self.environment != "libero":
            return {
                **self._primitives.robot_state(),
                "terminated": self._primitives.native_success(),
                "truncated": self._primitives.env_steps
                >= self._primitives._max_episode_steps,
            }
        env = self._primitives.env
        return {
            **self._primitives.measure().state(),
            "terminated": env.terminated,
            "truncated": env.truncated,
        }

    def initial_observation(self) -> ToolResult:
        """Return the task and measured initial state for the opening message."""
        gate = os.environ.get("RPENT_START_GATE")
        if gate:
            write_json(
                self.output_dir / "READY.json",
                {
                    "pid": os.getpid(),
                    "time": time.time(),
                    "gate": gate,
                    "task": self._task,
                },
            )
            deadline = time.monotonic() + float(
                os.environ.get("RPENT_START_GATE_TIMEOUT_S", "1200")
            )
            while not Path(gate).exists():
                if time.monotonic() >= deadline:
                    raise TimeoutError("batch start barrier timed out")
                time.sleep(0.1)
        self._started = time.perf_counter()
        self._started_at_wall = time.time()
        self._state.save(
            "observation-start.json",
            {"started_at": self._started_at_wall, "gate": gate},
            step=None,
        )
        step = self._state.latest_step
        observation = ToolResult(
            data={
                "task": self._task,
                "observation_mode": "adaptive",
                "state": {"step": step, **self._robot_state()},
            },
        )
        observation = self._model_feedback(observation)
        self._count_images(observation)
        self._state.save(
            "planner_audit.json",
            {
                "observation_mode": "adaptive",
                "episode_contract_version": EPISODE_CONTRACT_VERSION,
                "tools": [
                    {
                        "name": definition.name,
                        "description": definition.description,
                        "input_schema": definition.input_schema,
                    }
                    for definition in self.list_tools()
                ],
                "initial_observation": observation.data,
                "initial_images": len(observation.images),
            },
            step=None,
        )
        return observation

    def execute_tool(self, name: str, input_dict: dict[str, Any]) -> ToolResult:
        """Run one tool and count the images returned to the planner."""
        tool_started = time.perf_counter()
        declaration = self._tools.get(name)
        if declaration is None:
            return super().execute_tool(name, input_dict)
        try:
            parameters = declaration.validate_arguments(input_dict)
            arguments = {
                field: getattr(parameters, field)
                for field in type(parameters).model_fields
            }
            from robots.rl.contracts import validate_relations

            validate_relations(name, arguments)
        except (ValueError, TypeError) as exc:
            from robots.rl.contracts import argument_hint

            return ToolResult(
                data={"error": str(exc)},
                model_text=f"Invalid {name} arguments: {exc}. {argument_hint(name, self.environment)} No tool executed; correct the arguments and try again.",
            )
        result = self._execute_validated(declaration, input_dict, arguments)
        if self.solved() and self._native_success_seconds is None:
            self._native_success_seconds = time.perf_counter() - self._started
        elapsed = time.perf_counter() - tool_started
        timings = self.workflow_stats.setdefault("tool_seconds_by_name", {})
        timings[name] = timings.get(name, 0.0) + elapsed
        if name in {"segment", "back_project"}:
            self.workflow_stats["perception_seconds"] += elapsed
        result = self._model_feedback(result)
        self._count_images(result)
        return result

    def _count_images(self, result: ToolResult) -> None:
        with self._images_lock:
            self._images_sent += len(result.images)

    def get_env_state(
        self,
        *,
        command: dict[str, Any],
        result: dict[str, Any],
        elapsed_s: float,
        start_frame_version: int | None = None,
    ) -> ToolResult:
        """Return the measured outcome of a robot action.

        A new observation is recorded only when the action stepped the
        environment; otherwise the latest step number is reused.
        """
        if command["action"] == "execute_actions":
            return ToolResult(data=result)
        if self.infrastructure_error is not None:
            # The native action may have executed before its acknowledgement
            # was lost. Preserve the last observation; do not label a new
            # capture or a partial receipt as an authoritative post-action state.
            return ToolResult(
                data={
                    **result,
                    "status": "error",
                    "error": "native action outcome is uncertain",
                    "state_stale": True,
                    "start_frame_version": start_frame_version,
                    "step": self._state.latest_step,
                }
            )
        advanced = self._primitives.env_steps > self._recorded_env_steps
        step = self._state.latest_step + 1 if advanced else self._state.latest_step
        if "error" in result:
            data: dict[str, Any] = {"status": "error", "error": result["error"]}
        else:
            data = {"status": result["status"]}
            if "reason" in result:
                data["reason"] = result["reason"]
        if self.environment != "libero":
            # Native adapters also report per-arm errors and physical limits.
            data.update(result)
        for key in (
            "position_error_m",
            "rotation_error_rad",
            "opening_error",
            "env_steps",
        ):
            if key in result:
                data[key] = result[key]
        data["step"] = step
        if advanced or data["status"] != "error":
            data.update(self._robot_state())
        data["start_frame_version"] = start_frame_version
        data["frame_version"] = self.frame_version()
        if advanced:
            # Log the payload the planner receives, not the primitive's internals.
            capture_started = time.perf_counter()
            self._primitives.record_observation(
                command=command, result=data, elapsed_s=elapsed_s
            )
            self.workflow_stats["observation_capture_seconds"] = (
                self.workflow_stats.get("observation_capture_seconds", 0.0)
                + time.perf_counter()
                - capture_started
            )
            self._recorded_env_steps = self._primitives.env_steps
        return ToolResult(data=data)

    def solved(self) -> bool:
        """Return the environment's latched success flag."""
        if self.environment != "libero":
            return self._primitives.native_success()
        return self._primitives.env.terminated

    @property
    def infrastructure_error(self) -> dict | None:
        native_error = getattr(self._primitives, "infrastructure_error", None)
        return native_error if native_error is not None else self.action_error

    def completion_result(self) -> dict | None:
        """End the conversation before another request after a native terminal."""
        if self.infrastructure_error is not None:
            return {"status": "error", "reason": "infrastructure_error"}
        if self._primitives.ended:
            success = self.solved()
            return {"status": "completed", "success": success, "truncated": not success}
        return None

    def close(self) -> None:
        """Save the episode video and the rollout's observation counts."""
        super().close()
        self._state.save("episode.mp4", self._primitives.frames, step=None, fps=20)
        self._state.save(
            "rollout.json",
            {
                "observation_mode": "adaptive",
                "workflow": self.workflow,
                "world_cord": self.world_cord,
                "position_unit": self.position_unit,
                "coordinate_contract": "episode_coordinates_v2",
                "episode_contract_version": EPISODE_CONTRACT_VERSION,
                "reasoning": self.reasoning,
                "image_size": self.image_size,
                "action_mode": self.action_mode,
                "perception_mode": self.perception_mode,
                "max_observe_calls": self.max_observe_calls,
                "observe_concurrency": self.observe_concurrency,
                "max_actions_per_call": self.max_actions_per_call,
                "reasoning_history": self.reasoning_history,
                "image_history": self.image_history,
                "native_success_seconds": self._native_success_seconds,
                "started_at_wall": getattr(self, "_started_at_wall", None),
                **self.workflow_stats,
                "images_sent": self._images_sent,
                "images_generated_for_feedback": self._images_sent,
                "env_steps": self._primitives.env_steps,
                "observations": self._state.latest_step + 1,
            },
            step=None,
        )
