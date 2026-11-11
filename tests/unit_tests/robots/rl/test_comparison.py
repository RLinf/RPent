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

"""The comparison uses one fixed panel and its declared winner rule."""

import json

from tools.rl.compare_workflows import (
    WORKFLOWS,
    jobs,
    paired_successes,
    select_winner,
    wire_statistics,
)


def test_panel_has_all_forty_tasks_at_seed_zero():
    rows = jobs()
    assert len(rows) == 40
    assert len({(row["suite"], row["task"], row["seed"]) for row in rows}) == 40
    assert all(row["seed"] == 0 for row in rows)


def test_ranking_prioritizes_score_then_batch_time_then_stable_order():
    reports = {name: {"successes": 12, "wall_seconds": 100} for name in WORKFLOWS}
    assert select_winner(reports) == "episode"
    reports["zap_subtask"]["wall_seconds"] = 90
    assert select_winner(reports) == "zap_subtask"
    reports["zap_turn"]["successes"] = 13
    assert select_winner(reports) == "zap_turn"


def test_paired_times_exclude_failures_and_different_tasks():
    def row(task, success, seconds):
        return {
            "task": {"suite": "libero_goal", "task": task, "seed": 0},
            "success": success,
            "elapsed_s": seconds,
        }

    left = {"episodes": [row(0, True, 100), row(1, False, 1)]}
    right = {"episodes": [row(0, True, 80), row(1, True, 70), row(2, True, 60)]}
    assert paired_successes(left, right) == [
        {"task": ("libero_goal", 0, 0), "left_seconds": 100, "right_seconds": 80}
    ]


def test_wire_usage_includes_length_and_reports_missing_usage(tmp_path):
    trace = tmp_path / "api_trace/requests"
    for index in (1, 2):
        directory = trace / f"{index:06d}"
        directory.mkdir(parents=True)
        (directory / "metadata.json").write_text(
            json.dumps({"content_type": "text/event-stream"})
        )
    response = {
        "choices": [{"finish_reason": "length"}],
        "usage": {
            "prompt_tokens": 100,
            "completion_tokens": 32768,
            "prompt_tokens_details": {"cached_tokens": 30},
        },
    }
    (trace / "000001/response.body").write_text(
        "data: " + json.dumps(response) + "\n\ndata: [DONE]\n"
    )
    result = wire_statistics(tmp_path)
    assert result["requests"] == 2 and result["requests_missing_usage"] == 1
    assert result["observed_usage"] == {
        "input_tokens": 100,
        "output_tokens": 32768,
        "cached_input_tokens": 30,
    }
