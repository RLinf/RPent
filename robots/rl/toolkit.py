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

"""Unified RL toolkit, ordered action batches, and SFT prompt rendering."""

from __future__ import annotations

import threading
import time
from collections.abc import Mapping, Sequence
from dataclasses import replace
from pathlib import Path
from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from robots.rl.envs import primitive_owner
from robots.rl.geometry import NATIVE_SIZE
from robots.rl.perception import RecordedPerception
from robots.rl.units import (
    CAMERAS,
    CENTIMETERS,
    EPISODE_CONTRACT_VERSION,
    argument_hint,
    coordinate_definition,
    model_tool,
    strict_tool,
    validate_relations,
)
from rpent.dashboard.events import DashboardEventSink, NullDashboardEventSink
from rpent.memory import MemoryManager
from rpent.session import EnvState
from rpent.tools import Tool, ToolResult, tool
from rpent.tools.toolkit import ToolCancelled, Toolkit
from rpent.utils.logging import get_logger

logger = get_logger(__name__)

Vector3 = Annotated[list[float], Field(min_length=3, max_length=3)]


class MoveAction(BaseModel):
    """One absolute grasp TCP target; omitted parts keep measured values."""

    model_config = ConfigDict(
        extra="forbid",
        allow_inf_nan=False,
        json_schema_extra={
            "anyOf": [
                {"required": ["xyz"], "properties": {"xyz": {"type": "array"}}},
                {
                    "required": ["orientation_rpy"],
                    "properties": {"orientation_rpy": {"type": "array"}},
                },
            ]
        },
    )
    action: Literal["move_to"]
    xyz: Vector3 | None = Field(
        default=None,
        description="Absolute grasp TCP xyz in the selected world frame, metres.",
    )
    orientation_rpy: Vector3 | None = Field(
        default=None,
        description="Absolute grasp TCP fixed-axis XYZ roll,pitch,yaw, radians; Rz@Ry@Rx.",
    )

    @model_validator(mode="after")
    def require_target(self):
        if self.xyz is None and self.orientation_rpy is None:
            raise ValueError("move_to needs xyz, orientation_rpy, or both")
        return self


class GripperAction(BaseModel):
    """One normalized gripper target."""

    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)
    action: Literal["set_gripper"]
    opening: Annotated[
        float,
        Field(
            ge=0,
            le=1,
            description="0 closed, 1 fully open; contact is not proof of grasp.",
        ),
    ]


class ArmMove(MoveAction):
    arm: Literal["left", "right"]


class ArmGripper(GripperAction):
    arm: Literal["left", "right"]


class BaseMove(BaseModel):
    """Drive the chassis at a bounded local velocity, then stop."""

    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)
    action: Literal["move_base"]
    vx: Annotated[float, Field(ge=-0.35, le=0.35, description="Forward speed in m/s.")]
    vy: Annotated[float, Field(ge=-0.35, le=0.35, description="Leftward speed in m/s.")]
    yaw_rate: Annotated[
        float, Field(ge=-0.5, le=0.5, description="Counterclockwise speed in rad/s.")
    ]
    duration_s: Annotated[float, Field(ge=0.1, le=2)]


class LiftMove(BaseModel):
    """Move only the arm mounting lift."""

    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)
    action: Literal["move_lift"]
    delta_m: Annotated[
        float, Field(ge=-0.25, le=0.25, description="Relative lift height in metres.")
    ]
    duration_s: Annotated[float, Field(ge=0.1, le=5)] = 3


Action = Annotated[MoveAction | GripperAction, Field(discriminator="action")]
ArmAction = Annotated[ArmMove | ArmGripper, Field(discriminator="action")]
MobileAction = Annotated[
    MoveAction | GripperAction | BaseMove | LiftMove, Field(discriminator="action")
]


class ActionBatch:
    """Execute 1..K physical actions strictly in order."""

    def __init__(self, toolkit: RLToolkit | None):
        self.toolkit = toolkit

    @tool
    def execute_actions(
        self, actions: Annotated[list[Action], Field(min_length=1)]
    ) -> ToolResult:
        """Execute 1..K move_to/set_gripper actions strictly in list order. The entire list is validated before any movement. Each item finishes and is measured before the next begins. Stop immediately on native success, truncation, error or partial; remaining items are not executed. Returns every executed result and pose, executed_count, zero-based stopped_at, stop_reason and final_state. Use view_env to see the scene after motion. Example: {"actions":[{"action":"move_to","xyz":[0.1,0,0.3]},{"action":"set_gripper","opening":0}]}.

        Args:
            actions: Ordered list, at most the configured K; each item is move_to with xyz/orientation_rpy or set_gripper with opening. Omitted move targets keep the state measured at that item's start. The list contains only physical actions.
        """
        return self._execute_actions(actions)

    def _execute_actions(self, actions: Sequence[BaseModel]) -> ToolResult:
        tk = self.toolkit
        assert tk is not None
        results, reason, stopped_at = [], None, None
        tk.workflow_stats["batch_calls"] = tk.workflow_stats.get("batch_calls", 0) + 1
        for index, item in enumerate(actions):
            tk.raise_if_cancelled()
            if tk._primitives.ended:
                reason = "terminated" if tk.solved() else "truncated"
                break
            command = item.model_dump(exclude_none=True)
            name = command.pop("action")
            begin = time.perf_counter()
            start_version = tk.frame_version()
            try:
                start_version = tk.frame_version(refresh=True)
                native = getattr(tk._primitives, name)(**command)
                result = tk.get_env_state(
                    command={"action": name, **command},
                    result=native.to_dict(),
                    elapsed_s=time.perf_counter() - begin,
                    start_frame_version=start_version,
                )
            except ToolCancelled:
                raise
            except Exception as exc:
                logger.exception("Native action %s failed", name)
                tk.action_error = {"action": name, "message": str(exc)}
                tk._primitives._save_artifact(
                    "infrastructure_error.json", tk.action_error, None
                )
                result = ToolResult(
                    data={
                        "status": "error",
                        "error": "native action outcome is uncertain",
                        "step": tk._primitives.latest_step,
                        "state_stale": True,
                        "start_frame_version": start_version,
                    }
                )
            tk.workflow_stats["atomic_actions"] = (
                tk.workflow_stats.get("atomic_actions", 0) + 1
            )
            results.append({"index": index, "action": name, **result.data})
            stopped_at = index
            if result.data.get("terminated") or result.data.get("truncated"):
                reason = "terminated" if result.data.get("terminated") else "truncated"
            elif result.data.get("status") in {"partial", "error"}:
                reason = (
                    result.data.get("reason")
                    or result.data.get("error")
                    or result.data["status"]
                )
            if reason is not None:
                break
        if reason is not None:
            tk.workflow_stats["batch_pauses"] = (
                tk.workflow_stats.get("batch_pauses", 0) + 1
            )
        return ToolResult(
            data={
                "status": "completed" if reason is None else "stopped",
                "results": results,
                "executed_count": len(results),
                "stopped_at": stopped_at,
                "stop_reason": reason,
                "final_state": (
                    tk._primitives.latest_snapshot.state
                    if tk.infrastructure_error is not None
                    else tk._robot_state()
                ),
            }
        )


class DualArmBatch(ActionBatch):
    @tool
    def execute_actions(
        self, actions: Annotated[list[ArmAction], Field(min_length=1)]
    ) -> ToolResult:
        """Execute 1..K actions strictly in order. Each move_to or set_gripper names one left or right arm. The other arm holds its last target. Read the measured results; partial, error, native success or truncation stops the remaining list. Use view_env for RGB."""
        return self._execute_actions(actions)


class MobileBatch(ActionBatch):
    @tool
    def execute_actions(
        self, actions: Annotated[list[MobileAction], Field(min_length=1)]
    ) -> ToolResult:
        """Execute 1..K ordered move_base, move_lift, move_to or set_gripper items. Each item controls one component. An arm target uses the selected coordinate convention at the start of its item. In general coordinates the mount follows the base and lift; supply the target for the frame after each preceding item. Partial, error, success or truncation stops the remaining list. Use view_env for RGB."""
        return self._execute_actions(actions)


def system_prompt(variables: Mapping[str, object] | None = None) -> str:
    """Render the SFT robot system prompt in centimetres."""
    variables = variables or {}
    environment = str(variables.get("environment", "libero"))
    prompts = Path(__file__).parent / "prompts"
    names = ("general.md", f"{environment}_general.md", "episode.md")
    sections = [(prompts / name).read_text().strip() for name in names]
    if not variables.get("reasoning", True):
        sections[-1] = sections[-1].replace(
            "Reason briefly about the next decision using observations, not hypothetical completed actions.",
            "Call the next tool directly without reasoning or explanatory text.",
        )
    limit = variables.get("max_actions_per_call", 3)
    sections.append(
        f"Action interface: execute_actions takes one to {limit} physical items in order, including single-item lists. "
        "Read each measured result. A partial result, error or terminal stops the remaining items."
    )
    sections.append(
        "Older reasoning or camera images may be removed from the conversation. "
        "Use a new view when you need pixels absent from the current history."
    )
    return CENTIMETERS.description("\n\n".join(sections))


def user_prompt(variables: Mapping[str, object] | None = None) -> str:
    """Introduce the task and initial measured state."""
    return "The task and the measured initial state follow."


def render_prompts(
    environment: str, *, reasoning: bool = True, max_actions_per_call: int = 3
) -> tuple[str, str]:
    """Return ``(system_prompt, user_message)`` for an RL episode."""
    variables = {
        "environment": environment,
        "reasoning": reasoning,
        "max_actions_per_call": max_actions_per_call,
    }
    return system_prompt(variables), user_prompt(variables)


def episode_tools(
    toolkit: RLToolkit | None,
    environment: str,
    max_actions: int = 3,
    perception: RecordedPerception | None = None,
) -> list[Tool]:
    """Build the centimetre-normalized tool declarations for ``environment``."""
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
    return [
        model_tool(
            strict_tool(coordinate_definition(decl), cameras=CAMERAS[environment])
        )
        for decl in (owner.view_env, owner.segment, owner.back_project, batch)
    ]


def openai_tool_schemas(tools: Sequence[Tool]) -> list[dict[str, Any]]:
    """Serialize tool declarations to OpenAI function-calling schema dicts."""
    return [
        {
            "type": "function",
            "function": {
                "name": t.name,
                "description": t.description,
                "parameters": t.input_schema,
            },
        }
        for t in tools
    ]


class RLToolkit(Toolkit):
    """Unified toolkit over any of the 4 simulators (fast in-memory or disk-audited)."""

    planner_builtin_tools = False
    api_environment_context = False

    def __init__(
        self,
        *,
        runtime_kwargs: dict[str, Any],
        environment: str = "libero",
        output_dir: Path | str | None = None,
        record_artifacts: bool = False,
        dashboard_events: DashboardEventSink | None = None,
        memory: MemoryManager | None = None,
        image_size: int = 1024,
        max_actions_per_call: int = 3,
        reasoning: bool = True,
    ) -> None:
        self.environment = environment
        self.reasoning = reasoning
        self.max_actions_per_call = max_actions_per_call
        self.image_size = image_size
        self.record_artifacts = record_artifacts and output_dir is not None
        self.output_dir = Path(output_dir) if output_dir is not None else None
        self.workflow_stats: dict[str, Any] = {"perception_seconds": 0.0}
        self._started = time.perf_counter()
        self._native_success_seconds: float | None = None
        state = EnvState(self.output_dir) if self.record_artifacts else None
        super().__init__(
            dashboard_events=dashboard_events or NullDashboardEventSink(),
            state=state,
            memory=memory,
        )
        self._images_sent = 0
        self._images_lock = threading.Lock()
        self.action_error: dict[str, Any] | None = None
        self._primitives: RecordedPerception = primitive_owner(environment)(
            state=state,
            check_cancelled=self.raise_if_cancelled,
            record_artifacts=self.record_artifacts,
            **runtime_kwargs,
        )
        if environment in {"libero", "robocasa"} and image_size != NATIVE_SIZE:
            self._primitives.display_size = image_size
        self._task = self._primitives.env.get_task_language()
        record = self._primitives.record_observation()
        if record is not None:
            self._publish_step(record)
        self._recorded_env_steps = 0
        self.add_tools(
            episode_tools(
                self,
                environment,
                max_actions_per_call,
                perception=self._primitives,
            )
        )

    def _register_common_tools(self) -> None:
        """Register no filesystem tools; rollouts use only robot tools."""

    def frame_version(self, *, refresh: bool = False) -> int:
        if refresh and self.environment == "robocasa":
            self._primitives._refresh_coordinates()  # type: ignore[attr-defined]
        return getattr(getattr(self._primitives, "coordinates", None), "version", 0)

    def _model_feedback(self, result: ToolResult) -> ToolResult:
        result.data["coordinates"] = self._primitives.coordinates.description()
        result.data = CENTIMETERS.payload(result.data)
        result.data["position_unit"] = "cm"
        return result

    def _robot_state(self) -> dict[str, Any]:
        if self.environment == "libero":
            env = self._primitives.env
            return {
                **self._primitives.measure().state(),  # type: ignore[attr-defined]
                "terminated": bool(env.terminated),
                "truncated": bool(env.truncated),
            }
        return {
            **self._primitives.robot_state(),
            "terminated": self._primitives.native_success(),
            "truncated": self._primitives.env_steps
            >= self._primitives._max_episode_steps,
        }

    def initial_observation(self) -> ToolResult:
        """Return the task and measured initial state for the opening user message."""
        self._started = time.perf_counter()
        step = self._primitives.latest_step
        obs = ToolResult(
            data={"task": self._task, "state": {"step": step, **self._robot_state()}}
        )
        obs = self._model_feedback(obs)
        self._count_images(obs)
        return obs

    def execute_tool(self, name: str, input_dict: dict[str, Any]) -> ToolResult:
        started = time.perf_counter()
        declaration = self._tools.get(name)
        if declaration is None:
            return super().execute_tool(name, input_dict)
        try:
            parameters = declaration.validate_arguments(input_dict)
            arguments = {
                field: getattr(parameters, field)
                for field in type(parameters).model_fields
            }
            validate_relations(name, arguments)
        except (ValueError, TypeError) as exc:
            return ToolResult(
                data={"error": str(exc)},
                model_text=(
                    f"Invalid {name} arguments: {exc}. "
                    f"{argument_hint(name, self.environment)} "
                    "No tool executed; correct the arguments and try again."
                ),
            )
        result = self._execute_validated(declaration, input_dict, arguments)
        if self.solved() and self._native_success_seconds is None:
            self._native_success_seconds = time.perf_counter() - self._started
        elapsed = time.perf_counter() - started
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
        if command["action"] == "execute_actions":
            return ToolResult(data=result)
        if self.infrastructure_error is not None:
            return ToolResult(
                data={
                    **result,
                    "status": "error",
                    "error": "native action outcome is uncertain",
                    "state_stale": True,
                    "start_frame_version": start_frame_version,
                    "step": self._primitives.latest_step,
                }
            )
        advanced = self._primitives.env_steps > self._recorded_env_steps
        step = (
            self._primitives.latest_step + 1
            if advanced
            else self._primitives.latest_step
        )
        if "error" in result:
            data: dict[str, Any] = {"status": "error", "error": result["error"]}
        else:
            data = {"status": result["status"]}
            if "reason" in result:
                data["reason"] = result["reason"]
        if self.environment != "libero":
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
            capture_started = time.perf_counter()
            record = self._primitives.record_observation(
                command=command, result=data, elapsed_s=elapsed_s
            )
            if record is not None:
                self._publish_step(record)
            self.workflow_stats["observation_capture_seconds"] = (
                self.workflow_stats.get("observation_capture_seconds", 0.0)
                + time.perf_counter()
                - capture_started
            )
            self._recorded_env_steps = self._primitives.env_steps
        return ToolResult(data=data)

    def solved(self) -> bool:
        return self._primitives.native_success()

    @property
    def truncated(self) -> bool:
        return bool(self._primitives.latest_snapshot.truncated)

    @property
    def infrastructure_error(self) -> dict[str, Any] | None:
        return self._primitives.infrastructure_error or self.action_error

    @property
    def sam3_service_errors(self) -> list[dict[str, Any]]:
        return self._primitives.sam3_service_errors

    def completion_result(self) -> dict[str, Any] | None:
        if self.infrastructure_error is not None:
            return {"status": "error", "reason": "infrastructure_error"}
        if self._primitives.ended:
            success = self.solved()
            return {"status": "completed", "success": success, "truncated": not success}
        return None

    def close(self) -> None:
        super().close()
        if self.record_artifacts and self._state is not None:
            if self._primitives.frames:
                self._state.save(
                    "episode.mp4", self._primitives.frames, step=None, fps=20
                )
            self._state.save(
                "rollout.json",
                {
                    "position_unit": "cm",
                    "coordinate_contract": "episode_coordinates_v2",
                    "episode_contract_version": EPISODE_CONTRACT_VERSION,
                    "reasoning": self.reasoning,
                    "image_size": self.image_size,
                    "max_actions_per_call": self.max_actions_per_call,
                    "native_success_seconds": self._native_success_seconds,
                    **self.workflow_stats,
                    "images_sent": self._images_sent,
                    "env_steps": self._primitives.env_steps,
                    "observations": self._primitives.latest_step + 1,
                },
                step=None,
            )
