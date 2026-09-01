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

from robots.robotwin.prompt_bundle import system_prompt, user_prompt
from rpent.prompt.utils import format_prompt


def test_prompts_render_with_repo_relative_resource_paths():
    variables = {
        "task_name": "beat_block_hammer",
        "task_config": "demo_randomized",
        "seed": 100000,
    }

    system = format_prompt(system_prompt(), variables=variables)
    user = format_prompt(user_prompt(), variables=variables)

    assert "robots/robotwin/guides/GUIDE_RPENT.md" in system
    assert "resources/robotwin/recipe/beat_block_hammer_s0.json" in system
    assert "resources/robotwin/memory/MEMORY.md" in system
    assert "task: beat_block_hammer" in user
    assert "seed: 100000" in user
    assert "task_config: demo_randomized" in user
    assert "{{" not in system + user
