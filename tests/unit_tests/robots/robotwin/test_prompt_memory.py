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

"""Tests for the curated RoboTwin prompt."""

from __future__ import annotations

import pytest

from robots.robotwin.prompt_bundle import system_prompt, user_prompt


def test_prompts_render_with_memory_relative_paths():
    variables = {
        "task_name": "beat_block_hammer",
        "task_config": "demo_randomized",
        "seed": 100000,
        "memory_dir": "memory/robotwin",
        "reference_tag": "robotwin_beat_block_hammer_demo_randomized_s0",
    }

    system = system_prompt(variables)
    user = user_prompt(variables)

    assert "robots/robotwin/guides/GUIDE_RPENT.md" in system
    assert (
        "memory/robotwin/task-specific/robotwin_beat_block_hammer_demo_randomized_s0.json"
        in system
    )
    assert (
        "memory/robotwin/task-specific/robotwin_beat_block_hammer_demo_randomized_s0_recipe.jsonl"
        in system
    )
    assert "memory/robotwin/MEMORY.md" in system
    assert "task: beat_block_hammer" in user
    assert "seed: 100000" in user
    assert "task_config: demo_randomized" in user
    assert "{{" not in system + user


@pytest.mark.parametrize("mode", ["eval", "explore"])
def test_uniwam_prompt_selects_wam_tool_and_memory_for_mode(mode):
    variables = {
        "policy_backend": "uniwam",
        "policy_kind": "wam",
        "mode": mode,
        "memory_dir": "shared-memory",
        "task_name": "beat_block_hammer",
        "task_config": "demo_randomized",
        "seed": 7,
        "reference_tag": "beat_block_hammer_s0",
        "recipe_tag": "test",
        "session_number": 1,
        "session_max": 3,
        "memory_inbox": "shared-memory/inbox",
        "explore_attempts_per_session": 5,
        "output_dir": "output",
    }
    system = system_prompt(variables)
    user = user_prompt(variables)
    assert "GUIDE_RPENT.md" in system
    assert "{{" not in system + user
    assert "wam_act" in system + user
    assert "lingbot_act" not in system + user
    if mode == "eval":
        assert "shared-memory/MEMORY.md" in system
    else:
        assert "shared-memory/inbox/wip/" in system


@pytest.mark.parametrize(
    "backend", ["lingbot", "uniwam", "cosmos-policy", "cosmos3", "fast-wam"]
)
def test_task_and_backend_switch_use_current_data(backend):
    first = {
        "task_name": "first",
        "task_config": "demo_clean",
        "seed": 1,
        "memory_dir": "/memory/first",
        "reference_tag": "first_s0",
        "policy_backend": "lingbot",
    }
    system_prompt(first)
    values = first | {
        "task_name": "second {{literal}}",
        "seed": 8,
        "task_config": "demo_randomized",
        "memory_dir": "/memory/second",
        "reference_tag": "second_s0",
        "policy_backend": backend,
    }
    system, user = system_prompt(values), user_prompt(values)
    assert "first" not in user
    assert "/memory/first" not in system
    assert "task: second {{literal}}" in user
    assert "seed: 8" in user and "task_config: demo_randomized" in user
    if backend in {"cosmos-policy", "cosmos3", "fast-wam"}:
        assert "wam_act" in system and "GUIDE_RPENT" not in system
        assert "Local memory directory: /memory/second" in user
    else:
        assert "/memory/second/task-specific/second_s0.json" in system
        assert "preserve achieved relations" in system
        assert "preserve achieved relations" not in user
        if backend == "uniwam":
            assert "16 qpos actions" in system
            assert "concrete subtask instruction" in system
            # Preserve the old user-role instruction despite its policy tension.
            assert "complete current task_language unchanged" in system
        else:
            assert "use_length=50" in system
