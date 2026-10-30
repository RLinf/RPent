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
from typing import TYPE_CHECKING, Any, Literal

from rpent.dashboard.events import DashboardEventSink
from rpent.session import EnvState
from rpent.session.env_capture import CapturingEnvClient
from rpent.tools import Toolkit, ToolResult, iter_tools
from rpent.utils.logging import get_output_dir

if TYPE_CHECKING:
    from robots.robodojo.env_client import RoboDojoEnvClient
    from rpent.memory.manager import MemoryManager


class RoboDojoToolkit(Toolkit):
    """Toolkit for the RoboDojo (Isaac Sim) environment.

    Every robot tool is registered for every task. ``eval_fair`` additionally
    requires a mode-validated environment, removes file tools, and disables
    development perception recording. Scoring and safety diagnostics are never
    planner tools.
    """

    def __init__(
        self,
        *,
        primitives_kwargs: dict[str, Any],
        dashboard_events: DashboardEventSink,
        memory: MemoryManager,
        eval_fair: bool = False,
    ) -> None:
        self.eval_fair = eval_fair
        self._task_name = primitives_kwargs.get("task", "")
        if eval_fair and not getattr(primitives_kwargs.get("env"), "eval_fair", False):
            raise ValueError("eval-fair requires a mode-validated environment client")
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
        env = primitives_kwargs["env"]
        self._primitives = RoboDojoPrimitives(
            env=CapturingEnvClient(
                env,
                self._state,
                capture_call=partial(_capture_robodojo_call, env),
            ),
            sam3_client=primitives_kwargs.get("sam3_client"),
            vla_client=primitives_kwargs.get("vla_client"),
            action_type=primitives_kwargs.get("action_type", "joint"),
            check_cancelled=self.raise_if_cancelled,
        )
        self._publish_step(
            robodojo_tools.dump_state(self._primitives, self._state, log=None)
        )

    def close(self) -> None:
        """Flush the env server's episode videos before the daemon is stopped."""
        super().close()
        self._primitives.env.close()

    def solved(self) -> bool:
        """Read official success for the runner after planner execution."""
        if self.eval_fair:
            return False  # Plan completion is not an official task verdict.
        return self._primitives.env.is_success()

    def _register_robodojo_tools(self) -> None:
        from robots.robodojo import tools as robodojo_tools

        for definition in iter_tools(robodojo_tools):
            resources = {"state": self._state}
            if definition.name != "view_env_state":
                resources["primitives"] = self._primitives
            self.add_tool(definition.with_handler(partial(definition, **resources)))

    def get_env_state(
        self,
        *,
        command: dict[str, Any],
        result: dict[str, Any],
        elapsed_s: float,
    ) -> ToolResult:
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


def _capture_robodojo_call(
    env: RoboDojoEnvClient,
    method: Literal["step"],
    flat_action: Any,
) -> dict[str, Any]:
    """Execute a RoboDojo step and collect its action and observations."""
    response = getattr(env, method)(flat_action, capture=True)
    result = response["result"]
    obs, reward, done, info = result
    action = response["action"]

    def observation_payload(obs):
        return {
            "format": "robodojo.observation",
            "data": obs,
            "completeness": "complete",
            "inline_array_paths": tuple(f"/state/{key}" for key in obs["state"]),
        }

    return {
        "result": result,
        "before_observation": observation_payload(response["before_observation"]),
        "steps": [
            {
                "action": {
                    "format": f"robodojo.action.{response['action_type']}",
                    "data": action,
                    "completeness": "complete",
                    "inline_array_paths": tuple(f"/{key}" for key in action),
                },
                "after_observation": observation_payload(obs),
                "result": {
                    "format": "robodojo.step_result",
                    "data": {"reward": reward, "done": done, "info": info},
                    "completeness": "complete",
                },
            }
        ],
    }
