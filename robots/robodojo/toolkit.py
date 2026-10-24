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

"""RoboDojo tool registration and observation capture."""

from __future__ import annotations

from dataclasses import replace
from functools import partial
from pathlib import Path
from typing import TYPE_CHECKING, Any

from robots.robodojo.execution import ExecutionSyncError
from robots.robodojo.primitives import RoboDojoPrimitives
from rpent.dashboard.events import DashboardEventSink
from rpent.session import EnvState
from rpent.tools import Toolkit, ToolResult, iter_tools, tool
from rpent.utils.logging import get_logger, get_output_dir

logger = get_logger(__name__)

OBSERVATION_TOOLS = frozenset(
    {"view_env_state", "segment", "back_project", "sample_world_xyz", "query_world_map"}
)
MOTION_TOOLS = frozenset(
    {
        "move_to",
        "move_delta",
        "hold",
        "rotate_wrist",
        "set_gripper",
        "release",
        "vpp2_act",
    }
)

if TYPE_CHECKING:
    from rpent.memory.manager import MemoryManager


class RoboDojoToolkit(Toolkit):
    """Toolkit for the RoboDojo (Isaac Sim) environment.

    Legacy policy code and composites are not exposed. ``eval_fair`` requires a mode-validated
    environment, removes file tools, and disables development perception
    recording. Scoring and safety diagnostics are never planner tools.
    """

    def __init__(
        self,
        *,
        primitives_kwargs: dict[str, Any],
        dashboard_events: DashboardEventSink,
        memory: MemoryManager,
        eval_fair: bool = False,
        state_output_dir: Path | str | None = None,
    ) -> None:
        self.eval_fair = eval_fair
        self.vla_agent = primitives_kwargs.get("vla_agent", False)
        self.registered_tools_only = self.vla_agent
        self._stateful_policy = (
            primitives_kwargs.get("vla_backend", "vpp2") == "awomo05"
        )
        if self.vla_agent and not self._stateful_policy:
            raise ValueError("VLA Agent requires a stateful policy backend")
        if self._stateful_policy and eval_fair:
            raise ValueError("Stateful policy execution is not supported in eval-fair")
        if eval_fair and not getattr(primitives_kwargs.get("env"), "eval_fair", False):
            raise ValueError("eval-fair requires a mode-validated environment client")
        state = EnvState(Path(state_output_dir or get_output_dir()))
        super().__init__(
            dashboard_events=dashboard_events,
            state=state,
            memory=memory,
        )
        names = self._tool_names(primitives_kwargs)
        self.init_primitives(primitives_kwargs=primitives_kwargs)
        try:
            self._register_robodojo_tools(names)
            if self.vla_agent:
                self._configure_memory_tools()
            self._publish_step(self._dump_state())
        except Exception:
            if self._stateful_policy:
                self._primitives.env.release()
            raise

    def _tool_names(self, kwargs) -> set[str]:
        if self.vla_agent:
            groups = set(kwargs.get("agent_tools", ("policy", "motion", "geometry")))
            choices = {
                "policy": {"run_policy"},
                "motion": MOTION_TOOLS - {"vpp2_act"},
                "geometry": {"back_project", "sample_world_xyz", "query_world_map"},
                "sam3": {"segment"},
            }
            if not groups or groups - choices.keys():
                raise ValueError("Unknown or empty Agent tool groups")
            if "sam3" in groups and kwargs.get("sam3_client") is None:
                raise ValueError("The sam3 tool group requires a SAM3 service")
            return {"view_env_state", "render", "finish"}.union(
                *(choices[g] for g in groups)
            )
        if self._stateful_policy:
            return {"view_env_state", "run_policy", "finish"}
        names = OBSERVATION_TOOLS | MOTION_TOOLS | {"finish", "render"}
        return set(
            names if kwargs.get("vla_client") is not None else names - {"vpp2_act"}
        )

    def flash_observation(self) -> dict:
        """Refresh the public observation used by live Flash grounding."""
        from robots.robodojo.access import public_observation

        if not self.eval_fair:
            raise RuntimeError("Flash requires eval-fair toolkit")
        obs = public_observation(self._primitives.env.get_obs())
        self._primitives._last_obs = obs
        return obs

    def init_primitives(self, *, primitives_kwargs: dict[str, Any]) -> None:
        """Start a new trace from the runtime-owned environment."""
        self._pending_perception = []
        env = primitives_kwargs["env"]
        policy = primitives_kwargs.get("vla_client")
        if self._stateful_policy:
            from robots.robodojo.stateful_policy import StatefulPolicyEnv

            if policy is None:
                raise ValueError("Stateful policy execution requires a VLA client")
            env = StatefulPolicyEnv(
                env,
                policy,
                self.raise_if_cancelled,
            )
        self._state.reset()
        self._primitives = RoboDojoPrimitives(
            env=env,
            sam3_client=primitives_kwargs.get("sam3_client"),
            vla_client=None
            if self._stateful_policy
            else primitives_kwargs.get("vla_client"),
            action_type=primitives_kwargs.get("action_type", "joint"),
            check_cancelled=self.raise_if_cancelled,
        )

    def solved(self) -> bool:
        """Read official success for the runner after planner execution."""
        if self.eval_fair:
            return False  # Plan completion is not an official task verdict.
        return self._primitives.env.is_success()

    def _register_robodojo_tools(self, names: set[str]) -> None:
        from robots.robodojo import tools as robodojo_tools

        if self.eval_fair or self.vla_agent:
            common = {"read_text_file", "list_dir"} if self.vla_agent else set()
            self._tools = {
                name: entry for name, entry in self._tools.items() if name in common
            }
        for definition in iter_tools(robodojo_tools):
            if definition.name not in names & OBSERVATION_TOOLS:
                continue
            resources = {"state": self._state}
            if definition.name in {"sample_world_xyz", "query_world_map"}:
                resources = {"env_state": self._state}
            elif definition.name == "segment":
                resources["primitives"] = self._primitives
            self.add_tool(definition.with_handler(partial(definition, **resources)))
        for definition in iter_tools(self._primitives):
            if definition.name not in names:
                continue
            self.add_tool(definition, replace=definition.name == "finish")
        if "run_policy" in names:
            from robots.robodojo.stateful_policy import run_policy

            self.add_tool(
                run_policy.with_handler(
                    partial(run_policy, primitives=self._primitives)
                )
            )
        if "render" in names:
            self.add_tool(self.render)

    @tool
    def render(self) -> ToolResult:
        """Capture fresh RGB and robot state, plus available geometry for enabled tools."""
        self.raise_if_cancelled()
        return ToolResult(data={"success": True})

    def get_env_state(
        self,
        *,
        command: dict[str, Any],
        result: dict[str, Any],
        elapsed_s: float,
    ) -> ToolResult:
        from robots.robodojo import tools as robodojo_tools

        if self.vla_agent and "episode_status" in result:
            # Keep the existing public status boundary; inference metadata is
            # retained in full in the standard execution record.
            result = {
                **result,
                "episode_status": {
                    key: value
                    for key, value in result["episode_status"].items()
                    if key in {"step", "step_limit"}
                },
            }
        record = self._dump_state(
            log={"command": command, "result": result, "elapsed_s": elapsed_s},
            perception=self._pending_perception,
        )
        self._pending_perception = []
        return robodojo_tools.view_env_state(record.step_idx, state=self._state)

    def _dump_state(self, **kwargs):
        from robots.robodojo import tools as robodojo_tools

        return robodojo_tools.dump_state(
            self._primitives,
            self._state,
            geometry="back_project" in self._tools,
            require_rgb=self._stateful_policy,
            **kwargs,
        )

    def close(self) -> None:
        """Drain tools; the runtime owns the environment across planner sessions."""
        drained = False
        try:
            super().close()
            drained = True
        finally:
            try:
                if self._stateful_policy:
                    self._primitives.env.release()
                else:
                    end = getattr(self._primitives.env, "end_execution", None)
                    if end is not None:
                        end()
            except ExecutionSyncError:
                if drained:
                    raise
                logger.exception("Failed to close execution after tool cleanup failed")

    def cancel_active_and_wait(self) -> None:
        super().cancel_active_and_wait()
        pause = getattr(self._primitives.env, "pause_execution", None)
        if pause is not None:
            pause()

    def resume_calls(self) -> None:
        resume = getattr(self._primitives.env, "resume_execution", None)
        if resume is not None:
            resume()
        super().resume_calls()

    def _execute_tool(self, tool, kwargs, input_dict):
        name = tool.name
        before = (
            self._primitives.env.get_execution_state()
            if self._stateful_policy
            else None
        )
        previous_record = self._state.latest_step
        try:
            result = super()._execute_tool(tool, kwargs, input_dict)
        except Exception as exc:
            if not self._stateful_policy:
                raise
            # A state event can fail after motion was confirmed. Do not continue
            # control when its feedback could not be delivered to the planner.
            self._primitives.env.invalidate_execution(str(exc))
            data = {"error": str(exc), "state_delivery_error": str(exc)}
            if self._state.latest_step != previous_record:
                from robots.robodojo.tools import planner_result

                record = self._state.get()
                data.update(
                    step=record.step_idx,
                    artifacts=sorted(record.artifacts),
                    log={
                        "command": record.command,
                        "result": planner_result(record.result),
                        "elapsed_s": record.elapsed_s,
                    },
                )
            result = ToolResult(data=data)
        if self._stateful_policy and result.data.get("state_capture_error"):
            self._primitives.env.invalidate_execution(
                "Planner observation delivery failed"
            )
            result.data["synchronization_trusted"] = False
        if name == "finish" and result.data.get("_finish"):
            end = getattr(self._primitives.env, "end_execution", None)
            if end is not None:
                try:
                    end()
                except ExecutionSyncError as exc:
                    logger.exception("Failed to close execution after finish")
                    result.data.setdefault("error", str(exc))
        if self._stateful_policy:
            from robots.robodojo.tools import planner_result

            after = self._primitives.env.get_execution_state()
            result.data = planner_result(result.data)
            result.data["execution"] = {
                **after,
                "step_before": before["confirmed_step"],
                "step_after": after["confirmed_step"],
                "confirmed_actions": after["confirmed_step"] - before["confirmed_step"],
            }
        if not (self.eval_fair or self._stateful_policy) and name in {
            "segment",
            "back_project",
        }:
            import copy

            if isinstance(result.to_dict(), dict):
                # Read-only queries do not create EnvState steps. Attach them
                # to the next action for Flash grounding; actions themselves
                # are already recorded by the shared Toolkit/EnvState path.
                self._pending_perception.append(
                    copy.deepcopy(
                        {
                            "action": name,
                            "arguments": input_dict,
                            "result": result.to_dict(),
                        }
                    )
                )
        return result

    def _configure_memory_tools(self):
        # Reuse the common memory readers without exposing source or other files.
        for name in ("read_text_file", "list_dir"):
            declaration = self._tools[name]
            self.add_tool(
                replace(
                    declaration.with_handler(partial(self._read_memory, declaration)),
                    description=(
                        "Read a UTF-8 text file from the configured read-only memory."
                        if name == "read_text_file"
                        else "List the configured read-only memory; an empty path selects its root."
                    ),
                ),
                replace=True,
            )

    def _read_memory(self, declaration, path="", **kwargs):
        resolved = self.memory.authorize_read(path or self.memory.root)
        if not resolved.is_relative_to(self.memory.root.resolve()):
            raise PermissionError(
                "Mixed control file access is limited to configured memory"
            )
        return declaration(path=str(resolved), **kwargs)
