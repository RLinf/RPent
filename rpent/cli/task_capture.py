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

"""Task record lifecycle shared by the CLI and Dashboard runners."""

from __future__ import annotations

import argparse
import asyncio
from dataclasses import asdict

from pydantic_core import to_jsonable_python

from rpent.planner.base import PlannerResult
from rpent.robots.robot_spec import RunConfig
from rpent.session import EnvState, Payload
from rpent.tools.toolkit import Toolkit
from rpent.utils.logging import get_logger

logger = get_logger("task_capture")


def start_api_task(
    toolkit: Toolkit, args: argparse.Namespace, config: RunConfig, instruction: str
) -> EnvState | None:
    """Open API capture on this execution's EnvState."""
    if args.planner != "api" or not args.record_trajectory:
        return None
    state = toolkit.state
    state.start_task(
        instruction=instruction,
        config=Payload(
            format="rpent.task_config",
            format_version="1",
            data=to_jsonable_python({"arguments": vars(args), "run": asdict(config)}),
            completeness="complete",
        ),
    )
    return state


def end_api_task(
    state: EnvState | None,
    result: PlannerResult | None,
    *,
    finish_result: dict | None,
    environment_success: bool | None,
    exception: BaseException | None,
    cancelled_reason: str | None = None,
) -> None:
    """Close capture after toolkit cleanup, keeping outcome separate from success."""
    if state is None:
        return
    error = result.error if result is not None else None
    reason = getattr(result, "stop_reason", None)
    if exception is not None:
        error = f"{type(exception).__name__}: {exception}"
        status = (
            "cancelled"
            if isinstance(exception, (KeyboardInterrupt, asyncio.CancelledError))
            else "failed"
        )
        reason = "operator" if status == "cancelled" else "execution_error"
    elif cancelled_reason is not None:
        status, reason = "cancelled", cancelled_reason
    elif (finish_result or {}).get("operator_aborted"):
        status, reason = "cancelled", "operator"
    elif (finish_result or {}).get("operator_finished"):
        status, reason = "completed", "operator_finished"
    elif reason == "cancelled":
        status = "cancelled"
    elif error is not None or result is None:
        status, reason = "failed", reason or "planner_error"
    else:
        status = "completed"
        reason = reason or ("finish" if finish_result is not None else "model_output")
    state.end_task(
        status=status,
        stop_reason=reason,
        result=Payload(
            format="rpent.task_result",
            format_version="1",
            data={
                "finish_result": finish_result,
                "environment_success": environment_success,
                "stats": result.stats if result is not None else {},
                "error": error,
            },
            completeness="complete",
        ),
    )
    try:
        path = state.save_planner_trajectory()
    except Exception:
        logger.exception("Failed to save planner trajectory")
    else:
        logger.info("Planner trajectory saved to %s", path)
