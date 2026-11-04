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

"""Summarize EFE trials and compare the raw-tools and skills arms.

The report reads Harbor's trial results and the RPent session report in the
same trial directory. It reports the Verifier reward and the agent-side
counters separately; neither replaces the other.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class TrialSummary:
    """One trial's Verifier reward and RPent counters."""

    trial: str
    task: str
    arm: str
    success: int
    infra_error: int
    wall_s: float
    input_tokens: int | None
    output_tokens: int | None
    tool_calls: int | None
    submit_calls: int | None
    control_steps: int | None
    failed_calls: int | None
    skill_calls: int | None

    def row(self) -> dict[str, Any]:
        return {
            "trial": self.trial,
            "arm": self.arm,
            "task": self.task,
            "success": self.success,
            "infra": self.infra_error,
            "wall_s": self.wall_s,
            "in_tokens": self.input_tokens,
            "out_tokens": self.output_tokens,
            "tool_calls": self.tool_calls,
            "submit": self.submit_calls,
            "steps": self.control_steps,
            "failed": self.failed_calls,
            "skills": self.skill_calls,
        }


def _seconds(started: str | None, finished: str | None) -> float:
    if not started or not finished:
        return 0.0
    from datetime import datetime

    try:
        start = datetime.fromisoformat(started.replace("Z", "+00:00"))
        end = datetime.fromisoformat(finished.replace("Z", "+00:00"))
    except ValueError:
        return 0.0
    return round((end - start).total_seconds(), 1)


def _sum_metric(sessions: list[dict[str, Any]], key: str) -> int | None:
    values = [
        session.get("metrics", {}).get(key)
        for session in sessions
        if isinstance(session.get("metrics"), dict)
    ]
    present = [int(value) for value in values if value is not None]
    return sum(present) if present else None


def summarize_trial(result_path: Path) -> TrialSummary | None:
    """Read one Harbor trial. Return ``None`` for a job-level result file."""
    try:
        result = json.loads(result_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(result, dict) or "task_name" not in result:
        return None
    trial_dir = result_path.parent
    session_path = trial_dir / "agent" / "rpent" / "session.json"
    session: dict[str, Any] = {}
    if session_path.is_file():
        try:
            session = json.loads(session_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            session = {}
    reward = (result.get("verifier_result") or {}).get("rewards") or {}
    agent = result.get("agent_result") or {}
    sessions = session.get("sessions") or []
    stats = session.get("stats") or {}
    skills = session.get("skills")
    if skills is None:
        skills = any(
            bool((item.get("metrics") or {}).get("skills_enabled"))
            for item in sessions
            if isinstance(item, dict)
        )
    tool_calls = _sum_metric(sessions, "tool_calls")
    if tool_calls is None:
        tool_calls = stats.get("tool_calls")
    return TrialSummary(
        trial=trial_dir.name,
        task=str(result.get("task_name") or ""),
        arm="rpent-skills" if skills else "rpent-raw",
        success=int(reward.get("success") or 0),
        infra_error=int(reward.get("infra_error") or 0),
        wall_s=_seconds(result.get("started_at"), result.get("finished_at")),
        input_tokens=agent.get("n_input_tokens"),
        output_tokens=agent.get("n_output_tokens"),
        tool_calls=tool_calls,
        submit_calls=_sum_metric(sessions, "efe_submit_calls"),
        control_steps=_sum_metric(sessions, "efe_control_steps"),
        failed_calls=_sum_metric(sessions, "failed_calls"),
        skill_calls=_sum_metric(sessions, "skill_calls"),
    )


def find_trials(root: Path) -> Iterator[TrialSummary]:
    """Yield summaries for every Harbor trial below ``root``."""
    for result_path in sorted(root.rglob("result.json")):
        summary = summarize_trial(result_path)
        if summary is not None:
            yield summary


def _mean(values: list[float]) -> str:
    if not values:
        return "-"
    return f"{sum(values) / len(values):.1f}"


def summarize_arm(summaries: list[TrialSummary]) -> dict[str, dict[str, str]]:
    """Aggregate the arms that appear in ``summaries``."""
    arms: dict[str, list[TrialSummary]] = {}
    for summary in summaries:
        arms.setdefault(summary.arm, []).append(summary)
    output: dict[str, dict[str, str]] = {}
    for arm, items in sorted(arms.items()):
        output[arm] = {
            "trials": str(len(items)),
            "success_rate": f"{sum(item.success for item in items) / len(items):.2f}",
            "mean_wall_s": _mean([item.wall_s for item in items]),
            "mean_input_tokens": _mean(
                [float(item.input_tokens or 0) for item in items]
            ),
            "mean_tool_calls": _mean([float(item.tool_calls or 0) for item in items]),
            "mean_submit_calls": _mean(
                [float(item.submit_calls or 0) for item in items]
            ),
            "mean_control_steps": _mean(
                [float(item.control_steps or 0) for item in items]
            ),
            "mean_failed_calls": _mean(
                [float(item.failed_calls or 0) for item in items]
            ),
        }
    return output


def _format_table(headers: list[str], rows: list[list[str]]) -> str:
    widths = [
        max([len(header), *(len(row[index]) for row in rows)])
        for index, header in enumerate(headers)
    ]

    def line(values: list[str]) -> str:
        return " | ".join(value.ljust(widths[i]) for i, value in enumerate(values))

    divider = "-+-".join("-" * width for width in widths)
    return "\n".join([line(headers), divider, *(line(row) for row in rows)])


def render(summaries: list[TrialSummary]) -> str:
    """Return the per-trial table and the per-arm aggregate."""
    headers = [
        "arm",
        "task",
        "success",
        "infra",
        "wall_s",
        "in_tokens",
        "out_tokens",
        "tool_calls",
        "submit",
        "steps",
        "failed",
        "skills",
    ]
    rows = [
        [
            summary.arm,
            summary.task,
            str(summary.success),
            str(summary.infra_error),
            f"{summary.wall_s:.1f}",
            str(summary.input_tokens if summary.input_tokens is not None else "-"),
            str(summary.output_tokens if summary.output_tokens is not None else "-"),
            str(summary.tool_calls if summary.tool_calls is not None else "-"),
            str(summary.submit_calls if summary.submit_calls is not None else "-"),
            str(summary.control_steps if summary.control_steps is not None else "-"),
            str(summary.failed_calls if summary.failed_calls is not None else "-"),
            str(summary.skill_calls if summary.skill_calls is not None else "-"),
        ]
        for summary in summaries
    ]
    aggregate_rows = [
        [arm, *values.values()] for arm, values in summarize_arm(summaries).items()
    ]
    aggregate_headers = [
        "arm",
        "trials",
        "success_rate",
        "mean_wall_s",
        "mean_input_tokens",
        "mean_tool_calls",
        "mean_submit_calls",
        "mean_control_steps",
        "mean_failed_calls",
    ]
    return "\n\n".join(
        [
            _format_table(headers, rows) if rows else "No trials found.",
            _format_table(aggregate_headers, aggregate_rows) if aggregate_rows else "",
        ]
    ).strip()


def main(argv: list[str] | None = None) -> int:
    """Print a report for one or more Harbor run directories."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("runs", nargs="+", type=Path)
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print the parsed trials and arm aggregates as JSON.",
    )
    args = parser.parse_args(argv)
    summaries: list[TrialSummary] = []
    for run in args.runs:
        if not run.is_dir():
            parser.error(f"Not a directory: {run}")
        summaries.extend(find_trials(run))
    if args.json:
        print(
            json.dumps(
                {
                    "trials": [summary.row() for summary in summaries],
                    "arms": summarize_arm(summaries),
                },
                indent=2,
            )
        )
    else:
        print(render(summaries))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
