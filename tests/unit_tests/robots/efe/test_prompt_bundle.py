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

"""EFE prompt composition contracts."""

from __future__ import annotations

from robots.efe.prompt_bundle import system_prompt
from rpent.prompt.utils import format_prompt

_VARIABLES = {
    "memory_dir": "/memory",
    "memory_inbox": "/memory/_internal/inbox/efe",
    "session_number": 1,
    "session_max": 1,
}


def _render(**kwargs) -> str:
    return format_prompt(system_prompt(_VARIABLES, **kwargs), variables=_VARIABLES)


def test_prompt_guidance_matches_the_run_configuration():
    default = _render()
    restricted = _render(restricted_shell=True, write_notes=True)

    assert "move_to_xyz" in default
    assert "move_to_xyz" not in _render(skills=False)
    assert "no network access" not in default
    assert "no network access" in restricted
    assert "notes-session-" not in default
    assert "notes-session-" in restricted


def test_skill_guidance_follows_the_run_variables():
    variables = {**_VARIABLES, "skills": False}

    text = format_prompt(system_prompt(variables), variables=variables)

    assert "move_to_xyz" not in text


def test_episode_controls_follow_the_task_capabilities():
    assert "efe_reset" not in _render()
    assert "efe_end_episode" not in _render()

    text = _render(reset=True, end_episode=True)

    assert "efe_reset" in text
    assert "efe_end_episode" in text
