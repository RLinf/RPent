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

"""RoboDojo toolkit: common tools + RoboDojo state-viewing tools (M1)."""

from __future__ import annotations

from functools import partial
from typing import Any

from rpent.dashboard.events import DashboardEventSink
from rpent.tools.state import EnvState
from rpent.tools.toolkit import Toolkit
from rpent.utils.logging import get_output_dir


class RoboDojoToolkit(Toolkit):
    """Toolkit for the RoboDojo (Isaac Sim) environment."""

    def __init__(
        self,
        *,
        primitives_kwargs: dict[str, Any],
        dashboard_events: DashboardEventSink,
    ) -> None:
        state = EnvState(get_output_dir())
        super().__init__(dashboard_events=dashboard_events, state=state)
        self._primitives = RoboDojoPrimitives(
            env=primitives_kwargs["env"],
            sam3_client=primitives_kwargs.get("sam3_client"),
            vla_client=primitives_kwargs.get("vla_client"),
            action_type=primitives_kwargs.get("action_type", "joint"),
            check_cancelled=self.raise_if_cancelled,
        )
        self._task_name = primitives_kwargs.get("task", "")
        self._register_robodojo_tools()

    def _register_robodojo_tools(self) -> None:
        from robots.robodojo import tools as robodojo_tools

        state_handlers = {
            "view_env_state": partial(
                robodojo_tools.view_env_state,
                primitives=self._primitives,
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
            "get_reward_details": partial(
                robodojo_tools.get_reward_details,
                primitives=self._primitives,
                state=self._state,
            ),
            "get_safety_status": partial(
                robodojo_tools.get_safety_status,
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

        dump = robodojo_tools.dump_state(
            self._primitives,
            self._state,
            log={"command": command, "result": result, "elapsed_s": elapsed_s},
        )
        # Surface the tool's own result (e.g. move_to diagnostics) to the LLM;
        # the stateful capture would otherwise replace it with the full dump.
        clean = {k: v for k, v in (result or {}).items() if k != "_image_bytes"}
        dump["last_result"] = robodojo_tools._jsonable(clean)
        return dump


class RoboDojoPrimitives:
    """Primitive implementations backed by the RoboDojo env RPC client."""

    def __init__(self, *, env, sam3_client, vla_client, action_type, check_cancelled):
        self.env = env
        self.sam3_client = sam3_client
        self.vla_client = vla_client
        self.action_type = action_type
        self._check_cancelled = check_cancelled
        self._last_obs = None

    def view_env_state(self) -> dict[str, Any]:
        self._check_cancelled()
        self._last_obs = self.env.get_obs()
        return self._last_obs
