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

"""RoboDojo toolkit: shared perception and backend-specific control tools."""

from __future__ import annotations

from functools import partial
from typing import TYPE_CHECKING, Any

from rpent.dashboard.events import DashboardEventSink
from rpent.session import EnvState
from rpent.tools.toolkit import Toolkit
from rpent.utils.logging import get_output_dir

if TYPE_CHECKING:
    from rpent.memory.manager import MemoryManager


class RoboDojoToolkit(Toolkit):
    """Toolkit for the RoboDojo (Isaac Sim) environment.

    ``allowed_tool_groups`` optionally filters robot tool registration, not
    common file tools or automatic state capture. None preserves all tools in
    dev. ``eval_fair`` additionally requires a mode-validated environment,
    removes file tools and task-specific helpers, and disables development
    perception recording. Scoring and safety diagnostics are never planner tools.
    """

    def __init__(
        self,
        *,
        primitives_kwargs: dict[str, Any],
        dashboard_events: DashboardEventSink,
        memory: MemoryManager,
        allowed_tool_groups: frozenset[str] | None = None,
        eval_fair: bool = False,
    ) -> None:
        from robots.robodojo.tools import TOOL_GROUPS

        if allowed_tool_groups is not None:
            unknown = allowed_tool_groups - TOOL_GROUPS.keys()
            if unknown:
                raise ValueError(f"Unknown RoboDojo tool groups: {sorted(unknown)}")
        self.eval_fair = eval_fair
        if eval_fair and not getattr(primitives_kwargs.get("env"), "eval_fair", False):
            raise ValueError("eval-fair requires a mode-validated environment client")
        if eval_fair:
            allowed_tool_groups = frozenset({"general", "mixed"}) & (
                allowed_tool_groups
                if allowed_tool_groups is not None
                else frozenset({"general", "mixed"})
            )
        self._allowed_tool_groups = allowed_tool_groups
        state = EnvState(get_output_dir())
        super().__init__(
            dashboard_events=dashboard_events,
            state=state,
            memory=memory,
        )
        self.init_primitives(primitives_kwargs=primitives_kwargs)
        self._register_robodojo_tools()
        if eval_fair:
            allowed = {
                "view_env_state",
                "segment",
                "back_project",
                "move_to",
                "set_gripper",
                "pi0_pick",
            }
            self._tools = {
                name: entry for name, entry in self._tools.items() if name in allowed
            }

    def execute_tool(self, name, input_dict):
        result = super().execute_tool(name, input_dict)
        if not self.eval_fair and name in {
            "segment",
            "back_project",
        }:
            import copy

            if isinstance(result.result, dict):
                # Read-only queries do not create EnvState steps. Attach them
                # to the next action for Flash grounding; actions themselves
                # are already recorded by the shared Toolkit/EnvState path.
                self._pending_perception.append(
                    copy.deepcopy(
                        {
                            "action": name,
                            "arguments": input_dict,
                            "result": result.result,
                        }
                    )
                )
        return result

    def flash_observation(self) -> dict:
        """Refresh the public observation used by live Flash grounding."""
        from robots.robodojo.access import public_observation

        if not self.eval_fair:
            raise RuntimeError("Flash requires eval-fair toolkit")
        obs = public_observation(self._primitives.env.get_obs())
        self._primitives._last_obs = obs
        return obs

    def init_primitives(self, *, primitives_kwargs: dict[str, Any]) -> None:
        """Wipe stale run artifacts, build the primitives, dump step 0.

        The env server resets the Isaac Sim scene once at boot, so the client
        does not re-reset here; this only establishes the initial state record.
        """
        from robots.robodojo import tools as robodojo_tools

        self._pending_perception = []
        self._state.reset()
        self._primitives = RoboDojoPrimitives(
            env=primitives_kwargs["env"],
            sam3_client=primitives_kwargs.get("sam3_client"),
            vla_client=primitives_kwargs.get("vla_client"),
            action_type=primitives_kwargs.get("action_type", "joint"),
            check_cancelled=self.raise_if_cancelled,
        )
        self._task_name = primitives_kwargs.get("task", "")
        self._publish_step(
            robodojo_tools.dump_state(self._primitives, self._state, log=None)
        )

    def close(self) -> None:
        """Flush the env server's episode videos before the daemon is stopped."""
        self._primitives.env.close()

    def solved(self) -> bool:
        """Read official success for the runner after planner execution."""
        if self.eval_fair:
            return False  # Plan completion is not an official task verdict.
        return self._primitives.env.is_success()

    def _register_robodojo_tools(self) -> None:
        from robots.robodojo import tools as robodojo_tools

        state_handlers = {
            "view_env_state": partial(
                robodojo_tools.view_env_state,
                state=self._state,
            ),
            "back_project": partial(
                robodojo_tools.back_project,
                primitives=self._primitives,
                state=self._state,
            ),
            "segment": partial(
                robodojo_tools.segment,
                primitives=self._primitives,
                state=self._state,
            ),
            "move_to": partial(
                robodojo_tools.move_to,
                primitives=self._primitives,
                state=self._state,
            ),
            "set_gripper": partial(
                robodojo_tools.set_gripper,
                primitives=self._primitives,
                state=self._state,
            ),
            "pi0_pick": partial(
                robodojo_tools.pi0_pick,
                primitives=self._primitives,
                state=self._state,
            ),
            "stabilize": partial(
                robodojo_tools.stabilize,
                primitives=self._primitives,
                state=self._state,
            ),
        }
        # place_in_bin is a put_bottles-specific primitive (bin-mouth pose);
        # hide it for other tasks so the agent does not misuse it.
        if self._task_name == "put_bottles_into_dustbin":
            state_handlers["place_in_bin"] = partial(
                robodojo_tools.place_in_bin,
                primitives=self._primitives,
                state=self._state,
            )
        for spec in robodojo_tools.TOOLS_SPEC:
            name = spec["name"]
            if self._allowed_tool_groups is not None and not any(
                name in robodojo_tools.TOOL_GROUPS[group]
                for group in self._allowed_tool_groups
            ):
                continue
            handler = state_handlers.get(name)
            if handler is None:
                handler = getattr(self._primitives, name, None)
            if handler is None:
                continue
            self.add_tool(name, spec, handler)

    def get_env_state(
        self,
        *,
        command: dict[str, Any],
        result: dict[str, Any],
        elapsed_s: float,
    ) -> dict[str, Any]:
        from robots.robodojo import tools as robodojo_tools

        record = robodojo_tools.dump_state(
            self._primitives,
            self._state,
            log={"command": command, "result": result, "elapsed_s": elapsed_s},
            perception=self._pending_perception,
        )
        self._pending_perception = []
        return robodojo_tools.view_env_state(record.step_idx, state=self._state)


class RoboDojoPrimitives:
    """Primitive implementations backed by the RoboDojo env RPC client."""

    def __init__(self, *, env, sam3_client, vla_client, action_type, check_cancelled):
        self.env = env
        self.sam3_client = sam3_client
        self.vla_client = vla_client
        self.action_type = action_type
        self._check_cancelled = check_cancelled
        self._last_obs = None
