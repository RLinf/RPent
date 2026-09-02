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

from __future__ import annotations

from argparse import Namespace

import pytest

from rpent.dashboard.launcher import apply_to_args, defaults_from_args


def test_defaults_from_args_maps_cli_names_to_dashboard_fields() -> None:
    args = Namespace(
        planner="claude_code",
        model=None,
        cuda_device="cuda:1",
        max_turns=12,
        max_episode_steps=34,
        planner_timeout_s=None,
        reasoning_effort="high",
        claude_code_max_budget_usd=1.25,
        no_images=False,
    )

    assert defaults_from_args(args) == {
        "planner": "claude_code",
        "model": None,
        "cuda-device": "cuda:1",
        "max-turns": 12,
        "max-episode-steps": 34,
        "planner-timeout-s": None,
        "reasoning-effort": "high",
        "claude-code-max-budget-usd": 1.25,
        "no-images": False,
    }


def test_apply_to_args_maps_and_converts_complete_claude_payload() -> None:
    args = Namespace(claude_code_max_budget_usd=None)
    payload = {
        "planner": "claude_code",
        "model": "claude-sonnet",
        "cuda-device": "cuda:2",
        "max-turns": "15",
        "max-episode-steps": "120",
        "planner-timeout-s": "45",
        "reasoning-effort": "medium",
        "claude-code-max-budget-usd": "2.75",
        "no-images": True,
    }

    apply_to_args(args, payload)

    assert vars(args) == {
        "planner": "claude_code",
        "model": "claude-sonnet",
        "cuda_device": "cuda:2",
        "max_turns": 15,
        "max_episode_steps": 120,
        "planner_timeout_s": 45,
        "reasoning_effort": "medium",
        "claude_code_max_budget_usd": 2.75,
        "no_images": True,
    }


@pytest.mark.parametrize("empty_value", [None, ""])
def test_apply_to_args_normalizes_empty_optional_values(empty_value: object) -> None:
    args = Namespace(claude_code_max_budget_usd=9.0)

    apply_to_args(
        args,
        {
            "planner": "claude_code",
            "model": empty_value,
            "cuda-device": empty_value,
            "max-turns": 1,
            "max-episode-steps": 2,
            "planner-timeout-s": empty_value,
            "claude-code-max-budget-usd": empty_value,
        },
    )

    assert args.model is None
    assert args.cuda_device is None
    assert args.planner_timeout_s is None
    assert args.claude_code_max_budget_usd is None
    assert args.reasoning_effort == "none"
    assert args.no_images is False


def test_apply_to_args_leaves_claude_budget_unchanged_for_other_planners() -> None:
    args = Namespace(claude_code_max_budget_usd=3.5)

    apply_to_args(
        args,
        {
            "planner": "codex",
            "max-turns": "8",
            "max-episode-steps": "64",
            "planner-timeout-s": 10,
            "claude-code-max-budget-usd": "999",
        },
    )

    assert args.claude_code_max_budget_usd == 3.5
    assert args.model is None
    assert args.cuda_device is None
    assert args.max_turns == 8
    assert args.max_episode_steps == 64
    assert args.planner_timeout_s == 10
    assert args.reasoning_effort == "none"
    assert args.no_images is False
