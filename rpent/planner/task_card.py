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
"""Planner that replays a recorded plan instead of asking an LLM.

The other backends put a model in the loop and let it decide what to do next.
This one replays a **task card** recorded from an earlier run: the actions,
their order, the prompts given to the policy and the gripper commands are all
fixed before the episode starts. Only perception is live, and it is what makes
the card transferable -- every waypoint is re-expressed against a fresh reading
of the object it was written relative to.

What a card holds, and how it is replayed, is the robot's business: this asks
the robot's :attr:`~rpent.robots.robot_spec.RobotSpec.replay_card` hook to run
the cell and reports what came back.
"""

from __future__ import annotations

import time
from pathlib import Path

from rpent.dashboard.events import DashboardEventSink
from rpent.planner.base import PlannerResult
from rpent.robots.base import get_robot_spec
from rpent.tools.toolkit import Toolkit
from rpent.utils.logging import get_logger

logger = get_logger("task_card")


class TaskCardPlanner:
    """Replay one recorded card against the toolkit the runtime handed over."""

    def __init__(
        self,
        *,
        recipe_tag: str,
        robot_name: str,
        output_dir: str | Path,
        dashboard_events: DashboardEventSink,
    ) -> None:
        """Note which cell this replays; the robot resolves it to a card."""
        self._recipe_tag = recipe_tag
        self._robot_name = robot_name
        self._output_dir = Path(output_dir)
        self._dashboard_events = dashboard_events

    def solve(
        self,
        *,
        system_prompt: str,
        user_message: str,
        toolkit: Toolkit,
        max_turns: int,
        input_queue=None,
        dashboard_interaction=None,
    ) -> PlannerResult:
        """Replay the card for this cell. The prompts and turn budget are unused.

        A card decides the actions before the episode begins, so there is no
        conversation to hold and no turn to spend. The arguments are accepted to
        satisfy the planner protocol.
        """
        replay_card = get_robot_spec(self._robot_name).replay_card
        if replay_card is None:
            return PlannerResult(
                error=f"the {self._robot_name} robot records no task cards, "
                "so --planner task_card has nothing to replay"
            )

        started = time.time()
        try:
            outcome = replay_card(toolkit, self._recipe_tag, logger.info)
        except Exception as exc:
            return PlannerResult(error=f"{type(exc).__name__}: {exc}")
        elapsed = time.time() - started

        solved = bool(outcome.get("done"))
        return PlannerResult(
            finish_result={
                "status": "success" if solved else "failure",
                "summary": (
                    f"replayed the {outcome.get('card', self._recipe_tag)} card: "
                    f"{outcome.get('plan')} actions, "
                    f"{outcome.get('anchors')} anchors re-localized"
                ),
            },
            stats={
                "total_input_tokens": 0,
                "total_output_tokens": 0,
                "turns_used": 0,
                "tool_calls": outcome.get("plan", 0),
                "elapsed_s": round(elapsed, 1),
            },
        )
