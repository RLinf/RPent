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

"""Offline contracts for the ``pi0_pick`` success/grasp-status semantics.

``pi0_pick.success`` is a motion heuristic (descent + post-descent ascent +
gripper closure).  These tests replay the issue #243 probe: the real
``pi0_pick`` runs against stubbed VLA/environment boundaries only — no
simulator, no privileged object state anywhere in the payload.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

from robots.libero.prompts import evaluate
from robots.libero.tools import LiberoPrimitives

# The issue-#243 probe table (EEF z in metres, gripper opening in metres):
# descent deep enough, gripper closes on air, then the arm rises above
# lift_thresh — every kinematic condition holds with nothing grasped.
ISSUE_243_TRAJECTORY = [
    (1.00, 0.080),  # initial
    (0.85, 0.002),  # after chunk 1: bottom of descent, fingers closed on air
    (0.91, 0.002),  # after chunk 2: lifted past lift_thresh, still on air
]


class _FakeEnv:
    """LIBERO env boundary stub: scripted observations, no simulator."""

    def __init__(
        self,
        observations: list[dict[str, Any]],
        *,
        terminated: bool = False,
        truncated: bool = False,
    ) -> None:
        self._observations = list(observations)
        self._index = 0
        self.terminated = terminated
        self.truncated = truncated
        self.return_all_frames = False

    def chunk_step(
        self, actions: np.ndarray
    ) -> tuple[dict[str, Any], float, bool, bool, dict[str, Any]]:
        obs = self._observations[min(self._index, len(self._observations) - 1)]
        self._index += 1
        return obs, 0.0, self.terminated, self.truncated, {}

    def step(
        self, action: np.ndarray
    ) -> tuple[dict[str, Any], float, bool, bool, dict[str, Any]]:
        return self.chunk_step(action)

    def validate_action(self, values: list[float]) -> np.ndarray:
        return np.asarray(values)


class _FakeVLA:
    """VLA boundary stub: fixed-size action chunk, observation untouched."""

    def predict(
        self, obs: dict[str, Any], options: dict[str, Any] | None = None
    ) -> np.ndarray:
        return np.zeros((4, 7))


def _observation(z: float, gripper: float) -> dict[str, Any]:
    """An observation whose EEF height is ``z`` and finger separation ``gripper``."""
    states = np.zeros(8, dtype=np.float64)
    states[:3] = [0.5, -0.1, z]
    states[6] = gripper / 2
    states[7] = -gripper / 2
    return {"states": states, "task_descriptions": ["pick up the ketchup bottle"]}


def _primitives(
    observations: list[dict[str, Any]], *, terminated: bool = False
) -> LiberoPrimitives:
    env = _FakeEnv(observations, terminated=terminated)
    return LiberoPrimitives(
        env=env,
        model=_FakeVLA(),
        sam3_client=SimpleNamespace(),
        check_cancelled=lambda: None,
    )


def _trajectory(rows: list[tuple[float, float]]) -> list[dict[str, Any]]:
    """Observations for rows 1..n: what the env returns after each chunk.

    Row 0 of a probe table is the pre-call observation and goes to
    ``set_obs``; the remaining rows are the chunk_step results.
    """
    return [_observation(z, grip) for z, grip in rows[1:]]


def test_empty_grasp_reports_unknown_grasp_status() -> None:
    """The issue-243 probe: closing on air still fires the motion heuristic.

    ``success`` stays the motion-based early-stop signal (unchanged behavior),
    but the structured result must expose an honest ``grasp_status`` — pi0_pick
    has no contact or held-object evidence, so it reports "unknown" instead of
    implying a verified pickup.
    """
    primitives = _primitives(_trajectory(ISSUE_243_TRAJECTORY))
    primitives.set_obs(_observation(*ISSUE_243_TRAJECTORY[0]))

    result = primitives.pi0_pick("pick up the ketchup bottle")

    data = result.data
    # Motion heuristic fires exactly as reported in the issue.
    assert data["success"] is True
    assert data["chunks_used"] == 2
    assert data["peak_lift_m"] == pytest.approx(0.06)
    assert data["min_gripper_opening"] == pytest.approx(0.002)
    # Honest separation: no contact/held-object evidence exists, so the
    # structured grasp status is unknown — never a verified pickup.
    assert data["grasp_status"] == "unknown"


def test_lower_gripper_bound_still_gates_closure() -> None:
    """A set ``gripper_open_thresh`` keeps fully-shut fingers out of 'closed'."""
    primitives = _primitives(_trajectory(ISSUE_243_TRAJECTORY))
    primitives.set_obs(_observation(*ISSUE_243_TRAJECTORY[0]))

    result = primitives.pi0_pick("pick up the ketchup bottle", gripper_open_thresh=0.02)

    data = result.data
    # Gripper opening 0.002 < 0.02 is never "closed", so the heuristic cannot
    # fire; the loop must run to the chunk budget.
    assert data["success"] is False
    assert data["chunks_used"] == data["max_chunks"]
    assert data["grasp_status"] == "unknown"


def test_no_lift_runs_to_budget_and_stays_unknown() -> None:
    """Descent + closure without ascent never reports success."""
    trajectory = [(1.00, 0.080), (0.85, 0.002), (0.85, 0.002)]
    primitives = _primitives(_trajectory(trajectory))
    primitives.set_obs(_observation(*trajectory[0]))

    result = primitives.pi0_pick("pick up the ketchup bottle")

    data = result.data
    assert data["success"] is False
    assert data["chunks_used"] == data["max_chunks"]
    assert data["grasp_status"] == "unknown"


def test_episode_end_path_reports_unknown_grasp_status() -> None:
    """Official termination drives success; grasp evidence is still unknown."""
    trajectory = [(1.00, 0.080), (0.96, 0.080)]
    primitives = _primitives(_trajectory(trajectory), terminated=True)
    primitives.set_obs(_observation(*trajectory[0]))

    result = primitives.pi0_pick("pick up the ketchup bottle")

    data = result.data
    assert data["success"] is True
    assert data["terminated"] is True
    assert data["grasp_status"] == "unknown"


def test_result_payload_exposes_no_privileged_object_state() -> None:
    """The pick payload stays proprioception-only (perception isolation)."""
    primitives = _primitives(_trajectory(ISSUE_243_TRAJECTORY))
    primitives.set_obs(_observation(*ISSUE_243_TRAJECTORY[0]))

    data = primitives.pi0_pick("pick up the ketchup bottle").data

    assert set(data) == {
        "name",
        "instruction",
        "success",
        "grasp_status",
        "chunks_used",
        "max_chunks",
        "peak_lift_m",
        "min_gripper_opening",
        "final_gripper_opening",
        "terminated",
        "truncated",
        "diagnostics",
    }


def test_pick_tool_description_declares_motion_heuristic_only() -> None:
    """The published tool description separates the heuristic from grasp proof."""
    description = LiberoPrimitives.pi0_pick.description.lower()

    assert "grasp_status" in description
    assert "motion heuristic" in description
    assert "not proof" in description


def test_evaluation_prompt_aligns_with_structured_grasp_status() -> None:
    """Rule 1b cites the structured field, not just prose."""
    assert "grasp_status" in evaluate.RULES
    assert "HINT, not" in evaluate.RULES
