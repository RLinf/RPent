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

"""Shared adaptive tools with either conversation or snapshot observations."""

from __future__ import annotations

import os
import threading
import time
from dataclasses import replace
from pathlib import Path
from typing import Any

from robots.rl.images import (
    IMAGE_SIZES,
    NATIVE_SIZE,
    native_pixel,
    resize_image,
    scale_box,
)
from robots.rl.prompt_bundle import validate_workflow
from robots.rl.tools import LiberoRLPrimitives
from rpent.dashboard.events import DashboardEventSink
from rpent.memory import MemoryManager
from rpent.session import EnvState
from rpent.tools import ToolResult, iter_tools
from rpent.tools.toolkit import Toolkit
from rpent.utils.serialization import write_json


def primitive_definitions(primitives, *, workflow, action_mode, perception_mode):
    """Select the same native declarations for local and remote execution."""
    for definition in iter_tools(primitives):
        if workflow == "pear" and definition.name not in {"move_to", "set_gripper"}:
            continue
        if definition.name in {"view_env", "segment", "back_project"} and (
            perception_mode == "batch" or workflow != "episode"
        ):
            continue
        if action_mode == "batch" and definition.name in {"move_to", "set_gripper"}:
            continue
        yield definition


def action_batch_definition(toolkit, max_actions_per_call):
    """Publish the configured bound on the single ordered action tool."""
    from robots.rl.actions import ActionBatch

    declaration = ActionBatch(toolkit).execute_actions
    schema = declaration.input_schema
    schema["properties"]["actions"]["maxItems"] = max_actions_per_call
    return replace(
        declaration,
        _input_schema=schema,
        description=declaration.description.replace(
            "1..K", f"1..{max_actions_per_call}"
        ),
    )


class RLToolkit(Toolkit):
    """Expose end-effector, perception, and finish tools.

    Episode uses explicit perception tools; ZAP modes attach automatic perception.
    """

    planner_builtin_tools = False

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
        action_mode: str = "single",
        max_actions_per_call: int = 3,
        reasoning_history: int = -1,
        image_history: int = -1,
        perception_mode: str = "single",
        max_observe_calls: int = 8,
        observe_concurrency: int = 8,
        world_cord: str = "base",
        reasoning: bool = True,
    ) -> None:
        if observation_mode != "adaptive":
            raise ValueError("rl supports only adaptive observations")
        validate_workflow(environment, workflow)
        from robots.rl.contracts import validate_coordinates

        validate_coordinates(environment, world_cord)
        self.world_cord, self.reasoning = world_cord, reasoning
        if image_size not in IMAGE_SIZES:
            raise ValueError(f"unsupported image size {image_size}")
        from robots.rl.contracts import validate_options

        validate_options(
            environment,
            workflow,
            action_mode,
            max_actions_per_call,
            reasoning_history,
            image_history,
        )
        from robots.rl.contracts import validate_perception_options

        validate_perception_options(
            workflow, perception_mode, max_observe_calls, observe_concurrency
        )
        self.perception_mode = perception_mode
        self.max_observe_calls, self.observe_concurrency = (
            max_observe_calls,
            observe_concurrency,
        )
        self.observer = None
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
        primitive_type = LiberoRLPrimitives
        if environment != "libero":
            from robots.rl.backends import RoboCasaPrimitives, RoboTwinPrimitives

            primitive_type = {
                "robocasa": RoboCasaPrimitives,
                "robotwin": RoboTwinPrimitives,
            }[environment]
        native_options = dict(runtime_kwargs)
        if environment == "libero":
            native_options.update(
                world_cord=world_cord, capture_geometry=workflow != "pear"
            )
        self._primitives = primitive_type(
            state=state, check_cancelled=self.raise_if_cancelled, **native_options
        )
        if workflow == "pear" and self._primitives.coordinates.table_height_m is None:
            raise ValueError(
                "pear requires native base and table calibration; use a matching environment server"
            )
        self._task = self._primitives.env.get_task_language()
        self._publish_step(self._primitives.record_observation())
        self._recorded_env_steps = 0
        self.snapshot = None
        if workflow.startswith("zap_"):
            from robots.rl.snapshot import Snapshot

            self.snapshot = Snapshot(
                self._primitives,
                environment=environment,
                task=self._task,
                max_episode_steps=self._primitives.env._client.call("env.get_env_meta")[
                    "max_episode_steps"
                ],
                image_size=image_size,
            )
        if perception_mode == "batch":
            from robots.rl.observe import Observe

            self.observer = Observe(self)
        for definition in primitive_definitions(
            self._primitives,
            workflow=workflow,
            action_mode=action_mode,
            perception_mode=perception_mode,
        ):
            self.add_tool(definition)
        if self.observer is not None:
            declaration = self.observer.observe
            schema = declaration.input_schema
            schema["properties"]["calls"]["maxItems"] = max_observe_calls
            schema["$defs"]["ViewCall"]["properties"]["camera"]["enum"] = [
                *self.observer.cameras,
                "all",
            ]
            if environment == "libero":
                for model, fields in (
                    ("SegmentCall", ("point",)),
                    ("ProjectCall", ("pixel", "box")),
                ):
                    for name in fields:
                        prop = schema["$defs"][model]["properties"][name]
                        for branch in prop.get("anyOf", [prop]):
                            if "array" in branch.get("type", ()):
                                branch["items"]["maximum"] = (
                                    image_size if name == "box" else image_size - 1
                                )
            self.add_tool(
                replace(
                    declaration,
                    _input_schema=schema,
                    description=declaration.description.replace(
                        "1..K", f"1..{max_observe_calls}"
                    ).replace("agentview", self.observer.cameras[0]),
                )
            )
        if workflow == "episode_image":
            from robots.rl.actions import LatestImageTools

            self.add_tools(iter_tools(LatestImageTools(self._primitives)))
        if action_mode == "batch":
            self.add_tool(action_batch_definition(self, max_actions_per_call))
        if self.snapshot is not None:
            self.add_tool(self.snapshot.change_query)
            if workflow == "zap_turn":
                self.add_tool(self.snapshot.update_todo)
            elif workflow == "zap_subtask":
                from robots.rl.snapshot import SubtaskPlan

                self.add_tool(SubtaskPlan(self.snapshot).update_todo)

    def add_tool(self, declaration, *, replace=False) -> None:
        """Expose the same strict input contract at the model and execution boundary."""
        from robots.rl.contracts import coordinate_definition, strict_tool

        primitive = getattr(self, "_primitives", None)
        cameras = getattr(primitive, "cameras", ("agentview", "wrist"))
        declaration = strict_tool(
            coordinate_definition(declaration, self.world_cord),
            image_size=self.image_size,
            latest=self.workflow == "episode_image",
            cameras=cameras,
        )
        super().add_tool(declaration, replace=replace)

    def observe_snapshot(self) -> ToolResult:
        """Return a complete current observation without accumulated history."""
        if self.snapshot is None:
            raise ValueError("snapshot observations require --workflow zap_turn")
        begin = time.perf_counter()
        observation = self.snapshot.observe(self._robot_state())
        if self.world_cord == "general":
            observation.data["coordinates"] = self._primitives.coordinates.description()
        self.workflow_stats["perception_seconds"] += time.perf_counter() - begin
        return observation

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
        if self.world_cord == "general" or self.workflow == "pear":
            observation.data["coordinates"] = self._primitives.coordinates.description()
        if self.workflow in {"zap_episode", "zap_subtask"}:
            observation = self.observe_snapshot()
        elif self.workflow in {"episode_image", "pear"}:
            view = self.current_image_observation()
            observation.data["observation"] = view.data
            observation.images = view.images
        if self.observer is not None:
            observation.data["state"]["image_sources"] = self.observer.publish_current()
        self._count_images(observation)
        self._state.save(
            "planner_audit.json",
            {
                "observation_mode": "adaptive",
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
            if name == "observe":
                self.observer.compile(arguments["calls"])
            if self.environment == "libero" and self.image_size != NATIVE_SIZE:
                for field in ("point", "pixel"):
                    if arguments.get(field) is not None:
                        arguments[field] = native_pixel(
                            arguments[field], self.image_size
                        )
                if arguments.get("box") is not None:
                    box = arguments["box"]
                    if box[0] >= box[2] or box[1] >= box[3]:
                        raise ValueError("box must have positive height and width")
                    arguments["box"] = scale_box(box, self.image_size, NATIVE_SIZE)
        except (ValueError, TypeError) as exc:
            from robots.rl.contracts import argument_hint

            return ToolResult(
                data={"error": str(exc)},
                model_text=f"Invalid {name} arguments: {exc}. {argument_hint(name, self.environment)} No tool executed; correct the arguments and try again.",
            )
        result = self._execute_validated(declaration, input_dict, arguments)
        if self.environment == "libero" and self.image_size != NATIVE_SIZE:
            result.images = [
                resize_image(image, self.image_size) for image in result.images
            ]
            if "box" in result.data:
                result.data["box"] = scale_box(
                    result.data["box"], NATIVE_SIZE, self.image_size
                )
            for image in result.data.get("images", []):
                image["size"] = [self.image_size, self.image_size]
        if self.solved() and self._native_success_seconds is None:
            self._native_success_seconds = time.perf_counter() - self._started
        if self.snapshot is not None:
            # Keep the raw outcome only: never nest an earlier snapshot/history.
            self.snapshot.last_result = {
                "tool": name,
                "arguments": input_dict,
                "result": dict(result.data),
            }
            if self.workflow in {"zap_episode", "zap_subtask"} and name in {
                "move_to",
                "set_gripper",
                "change_query",
            }:
                observation = self.observe_snapshot()
                observation.data.pop("last_result", None)
                result.data["observation"] = observation.data
                result.images.extend(observation.images)
        if name in {"move_to", "set_gripper"}:
            self.workflow_stats["atomic_actions"] = (
                self.workflow_stats.get("atomic_actions", 0) + 1
            )
        if self.workflow in {"episode_image", "pear"} and name in {
            "move_to",
            "set_gripper",
            "execute_actions",
        }:
            view = self.current_image_observation()
            result.data["observation"] = view.data
            result.images = view.images
        elapsed = time.perf_counter() - tool_started
        timings = self.workflow_stats.setdefault("tool_seconds_by_name", {})
        timings[name] = timings.get(name, 0.0) + elapsed
        if name in {"segment", "back_project", "observe"}:
            self.workflow_stats["perception_seconds"] += elapsed
        if self.observer is not None and name == "execute_actions":
            result.data["image_sources"] = self.observer.publish_current()
        self._count_images(result)
        return result

    def current_image_observation(self) -> ToolResult:
        """Return the latest paired RGB with the actual model image dimensions."""
        view = self._primitives.view_env()
        for image in view.data["images"]:
            image["size"] = [self.image_size, self.image_size]
        view.images = [resize_image(image, self.image_size) for image in view.images]
        return view

    def _count_images(self, result: ToolResult) -> None:
        with self._images_lock:
            self._images_sent += len(result.images)

    def get_env_state(
        self,
        *,
        command: dict[str, Any],
        result: dict[str, Any],
        elapsed_s: float,
    ) -> ToolResult:
        """Return the measured outcome of a robot action.

        A new observation is recorded only when the action stepped the
        environment; otherwise the latest step number is reused.
        """
        if command["action"] == "execute_actions":
            return ToolResult(data=result)
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
        if self.observer is not None:
            # Include the exact planner feedback in the recorded state as well.
            data["image_sources"] = self.observer.image_ids(step)
        if advanced or data["status"] != "error":
            data.update(self._robot_state())
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
        if self.observer is not None:
            self.observer.publish_current()
        return ToolResult(data=data)

    def completion_result(self) -> dict[str, Any] | None:
        """Finish pear from native flags, without an extra model request."""
        if self.workflow != "pear" or not self._primitives.ended:
            return None
        return {
            "_finish": True,
            "success": self.solved(),
            "terminated": self._primitives.env.terminated,
            "truncated": self._primitives.env.truncated,
        }

    def solved(self) -> bool:
        """Return the environment's latched success flag."""
        if self.environment != "libero":
            return self._primitives.native_success()
        return self._primitives.env.terminated

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
