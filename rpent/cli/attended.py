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

"""CLI orchestration for one operator-attended attempt."""

from __future__ import annotations

import shutil
from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING, Any

from rpent.evaluation.result import record_operator_outcome
from rpent.session.transcript import _serialize_messages
from rpent.utils.logging import get_logger

logger = get_logger("agent")

if TYPE_CHECKING:
    from rpent.memory import MemoryManager
    from rpent.planner.base import PlannerResult


def _finalize_operator_verdict(
    result: PlannerResult,
    *,
    finalize_verdict: Callable[[], dict[str, Any]],
    state_output_dir: Path,
    session_number: int,
    archive: bool = True,
) -> tuple[dict[str, Any], str | None, str]:
    """Return the finalized outcome, remaining planner error, and next-session handoff."""
    finish = finalize_verdict()
    error = result.error
    if error:
        logger.info("Planner stopped after operator verdict: %s", error)
        result.stats["planner_error_at_operator_verdict"] = error
    verdict = finish["operator_verdict"]
    handoff = ""
    if archive and verdict in {"success", "failure"}:
        evidence = record_operator_outcome(
            session_number=session_number,
            state_output_dir=state_output_dir,
            verdict=finish,
            messages=_serialize_messages(result.messages),
            planner_error=error,
        )
        handoff = (
            f"\nThe operator marked the preceding attempt as {verdict}. "
            f"Read {evidence}, "
            "inspect its evidence and write a grounded outcome summary "
            "to your memory inbox wip notes. Distinguish observations "
            "from hypotheses; never label a failed attempt as successful. "
            "Follow the robot's exploration workflow for the next attempt.\n"
        )
        if verdict == "failure":
            error = None
    return finish, error, handoff


def _publish_success_recipe(
    memory: MemoryManager,
    *,
    recipe_tag: str,
    output_dir: Path,
    state_output_dir: Path,
    auto_merge: bool,
) -> None:
    """Preserve successful-session artifacts before another session overwrites them."""
    for name in (f"{recipe_tag}.json", f"{recipe_tag}_recipe.jsonl"):
        shutil.copy2(Path(output_dir) / name, Path(state_output_dir) / name)
    if auto_merge:
        memory.merge_memory(
            cell_tag=recipe_tag, run_state_dir=state_output_dir, solved=True
        )
