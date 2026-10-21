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
"""Planner that runs MT3 instead of asking an LLM.

MT3 (Dreczkowski et al., "Learning a Thousand Tasks in a Day") retrieves the
demonstration of the task whose scene is closest to the live one, moves the arm
to that demonstration's bottleneck pose carried by the object pose change, and
replays the demonstration from there. Nothing is learned and nothing is
reasoned about during the episode.

What a library holds and how it is replayed is the robot's business: this asks
the robot's :attr:`~rpent.robots.robot_spec.RobotSpec.run_mt3` hook to run the
cell and reports what came back.
"""

from __future__ import annotations

import time

from rpent.planner.base import Planner, PlannerResult
from rpent.robots.base import get_robot_spec
from rpent.tools.toolkit import Toolkit
from rpent.utils.logging import get_logger

logger = get_logger("mt3")


class MT3Planner(Planner):
    """Run MT3 for one cell against the toolkit the runtime handed over."""

    def __init__(
        self,
        *,
        recipe_tag: str,
        robot_name: str,
    ) -> None:
        """Note which cell this runs; the robot resolves it to a library."""
        self._recipe_tag = recipe_tag
        self._robot_name = robot_name

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
        """Run the cell. The prompts and turn budget are unused.

        The arguments are accepted to satisfy the planner interface.
        """
        run_mt3 = get_robot_spec(self._robot_name).run_mt3
        if run_mt3 is None:
            return PlannerResult(
                error=f"the {self._robot_name} robot has no MT3 library, "
                "so --planner mt3 has nothing to replay"
            )

        started = time.time()
        try:
            outcome = run_mt3(toolkit, self._recipe_tag, logger.info)
        except Exception as exc:
            return PlannerResult(error=f"{type(exc).__name__}: {exc}")
        elapsed = time.time() - started

        solved = bool(outcome.get("done"))
        return PlannerResult(
            finish_result={
                "status": "success" if solved else "failure",
                "summary": (
                    f"ran {outcome.get('program', 'mt3')}: "
                    f"{outcome.get('plan')} calls, "
                    f"{outcome.get('anchors')} reference bodies"
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
