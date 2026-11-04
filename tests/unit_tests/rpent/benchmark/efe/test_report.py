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

"""The EFE report joins the Verifier reward with RPent's counters."""

from __future__ import annotations

import json

from rpent.benchmark.efe.report import find_trials, main, summarize_arm


def _trial(
    root,
    name,
    *,
    success,
    skills,
    submit_calls=3,
    control_steps=120,
    tool_calls=7,
    stats_tool_calls=7,
):
    trial = root / name
    agent = trial / "agent" / "rpent"
    agent.mkdir(parents=True)
    (trial / "result.json").write_text(
        json.dumps(
            {
                "task_name": "autobio-pickup-i00-standard",
                "started_at": "2026-11-02T10:00:00Z",
                "finished_at": "2026-11-02T10:02:00Z",
                "agent_result": {"n_input_tokens": 100, "n_output_tokens": 10},
                "verifier_result": {"rewards": {"success": success, "infra_error": 0}},
            }
        )
    )
    (agent / "session.json").write_text(
        json.dumps(
            {
                "skills": skills,
                "stats": {"tool_calls": stats_tool_calls},
                "sessions": [
                    {
                        "metrics": {
                            "tool_calls": tool_calls,
                            "failed_calls": 1,
                            "skill_calls": 2 if skills else 0,
                            "efe_submit_calls": submit_calls,
                            "efe_control_steps": control_steps,
                            "skills_enabled": skills,
                        }
                    }
                ],
            }
        )
    )
    return trial


def test_report_reads_rewards_and_agent_counters(tmp_path):
    _trial(tmp_path, "skills-trial", success=1, skills=True)
    _trial(tmp_path, "raw-trial", success=0, skills=False)

    summaries = {item.trial: item for item in find_trials(tmp_path)}

    assert summaries["skills-trial"].success == 1
    assert summaries["skills-trial"].arm == "rpent-skills"
    assert summaries["skills-trial"].submit_calls == 3
    assert summaries["skills-trial"].control_steps == 120
    assert summaries["skills-trial"].failed_calls == 1
    assert summaries["skills-trial"].wall_s == 120.0
    assert summaries["raw-trial"].arm == "rpent-raw"
    assert summaries["raw-trial"].success == 0


def test_zero_tool_calls_are_reported_not_replaced_by_planner_stats(tmp_path):
    _trial(
        tmp_path,
        "shell-bypass",
        success=1,
        skills=False,
        tool_calls=0,
        stats_tool_calls=15,
    )

    (summary,) = find_trials(tmp_path)

    assert summary.tool_calls == 0


def test_arm_summary_separates_the_two_arms(tmp_path):
    _trial(tmp_path, "skills-a", success=1, skills=True)
    _trial(tmp_path, "skills-b", success=0, skills=True)
    _trial(tmp_path, "raw-a", success=0, skills=False)

    summaries = list(find_trials(tmp_path))
    arms = summarize_arm(summaries)

    assert arms["rpent-skills"]["trials"] == "2"
    assert arms["rpent-skills"]["success_rate"] == "0.50"
    assert arms["rpent-raw"]["success_rate"] == "0.00"


def test_report_ignores_job_level_results(tmp_path):
    (tmp_path / "result.json").write_text(json.dumps({"n_total_trials": 1}))

    assert list(find_trials(tmp_path)) == []


def test_cli_prints_a_table(tmp_path, capsys):
    _trial(tmp_path, "skills-trial", success=1, skills=True)

    assert main([str(tmp_path)]) == 0

    output = capsys.readouterr().out
    assert "rpent-skills" in output
    assert "success_rate" in output
