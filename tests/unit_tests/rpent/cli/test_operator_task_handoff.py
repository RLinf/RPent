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

"""Preserve an edited operator task across fresh simulator planner sessions."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

from rpent.cli import main as cli
from rpent.planner.base import PlannerResult
from rpent.robots import PromptBundle, RobotSpec, RunConfig


@pytest.mark.parametrize(
    ("interactive", "custom"), [(True, True), (True, False), (False, False)]
)
def test_simulator_handoff_keeps_edited_task_once_and_preserves_budgets(
    monkeypatch, tmp_path, interactive, custom
):
    original = "Only validate reset and fresh observations; do not move objects."
    default = "Complete the configured native benchmark task.\n"
    requests = []
    builds = []

    class Planner:
        def solve(self, **kwargs):
            requests.append(kwargs)
            return PlannerResult(
                finish_result={"_finish": True, "status": "failure"},
                stop_info={
                    "reason": "finish",
                    "execution_certain": True,
                    "result_delivery_complete": True,
                    "cleanup_complete": True,
                    "allow_new_session": True,
                },
            )

    class Toolkit:
        memory = None

        def solved(self):
            return False

        def close(self):
            pass

    def arguments(parser, use_dashboard):
        parser.add_argument("--explore-sessions", type=int, default=1)
        parser.add_argument("--explore-attempts-per-session", type=int, default=2)

    def build(*args, **kwargs):
        builds.append(kwargs)
        return Planner()

    def reader(input_queue, **kwargs):
        input_queue.put(original if custom else kwargs["first_prompt_default"].strip())

    def initialize_output(path, **kwargs):
        Path(path).mkdir(parents=True, exist_ok=True)
        return Path(path)

    spec = RobotSpec(
        name="libero",
        prompts=PromptBundle(
            system=lambda variables: "system", user=lambda variables: default
        ),
        add_cli_args=arguments,
        parse_config=lambda args: RunConfig(
            recipe_tag="10_task_t0_s31",
            output_dir=Path(args.output_dir),
            prompt_vars={},
            task_desc={"robot": "libero"},
        ),
        init_runtime=lambda *args: ([], {}),
        supports_exploration=True,
    )
    monkeypatch.setattr(cli, "enumerate_robots", lambda: ("libero",))
    monkeypatch.setattr(cli, "get_robot_spec", lambda name: spec)
    monkeypatch.setattr(cli, "get_toolkit", lambda *args, **kwargs: Toolkit())
    monkeypatch.setattr(cli, "build_planner", build)
    monkeypatch.setattr(cli, "_preflight_deepseek", lambda args: None)
    monkeypatch.setattr(cli, "init_output_dir", initialize_output)
    monkeypatch.setattr(cli, "start_interactive_reader", reader)
    monkeypatch.setattr("rpent.memory.loading.prepare_run_memory", lambda *args: None)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "rpent",
            "--robot",
            "libero",
            "--planner",
            "deepseek_harness",
            "--model",
            "gpt-6.1-sol",
            "--reasoning-effort",
            "xhigh",
            "--output-dir",
            str(tmp_path / "run"),
            "--explore",
            "--explore-sessions",
            "3",
            "--explore-attempts-per-session",
            "2",
            "--max-turns",
            "17",
            "--planner-timeout-s",
            "123",
            *(["--tui"] if interactive else []),
        ],
    )

    assert cli.main() == 0
    assert len(requests) == len(builds) == 3
    expected_initial = (
        original if custom else default.strip() if interactive else default
    )
    assert requests[0]["user_message"] == expected_initial
    for number, request in enumerate(requests[1:], 2):
        message = request["user_message"]
        assert f"agent {number} of up to 3" in message
        assert message.count("Original operator task instruction:") == int(custom)
        assert message.count(original) == int(custom)
        if number == 3:
            assert "agent 2 of up to 3" not in message
    assert all(request["max_turns"] == 17 for request in requests)
    assert all(build["planner_timeout_s"] == 123 for build in builds)
    assert all(
        build["model"] == "gpt-6.1-sol" and build["reasoning_effort"] == "xhigh"
        for build in builds
    )
