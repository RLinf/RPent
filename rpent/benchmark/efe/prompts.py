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

"""RPent-style prompts for the EFE coding session.

The module renders the same RPent prompt primitives, memory protocol, and
session handoff text used by robot runs.
"""

from __future__ import annotations

from collections.abc import Mapping

from rpent.prompt.utils import BulletList, PromptNode, format_prompt

ROLE = """You are an RPent coding agent in an EFE agent container.
Read the supplied task instructions and public workspace documentation.
Use your native shell and file tools to write and execute Python control programs.
Operate the simulator only through the task's documented robot client interfaces.
Do not access simulator internals or verifier files. Do not change task or scoring files.
Harbor owns the execution budget. EFE's verifier determines task success.
Call finish when your work ends. Its status reports your progress, not a reward."""

MEMORY = """RPent memory is a corpus shared across sessions. Use it before acting.

- Read `{{memory_dir}}/global/` and `{{memory_dir}}/task-family/` when they exist.
  Apply a lesson only when this task matches its stated conditions.
- The corpus is read-only in this session for RPent's file tools.
"""

NOTES = """- You may write only under `{{memory_inbox}}/`. Do not create or modify any
  other file under the corpus.
- Write working notes to `{{memory_inbox}}/wip/notes-session-{{session_number}}.md`
  before calling finish. Record what you tried, what the simulator returned, and
  what you ruled out. Do not claim a mechanism you did not observe.
- EFE's verifier runs outside this session, so no note is promoted to the
  published corpus. Later sessions read your notes in `wip/`."""

EXPLORE = """You are session {{session_number}} of up to {{session_max}} on this task.
Earlier sessions archived notes under `{{memory_inbox}}/wip/`.
Read them before acting. Do not repeat an approach that a note reports as failed
unless you can name the parameter that changes the outcome."""
OUTPUT = BulletList(
    [
        "Brief reasoning before each tool call (1-2 sentences): observation -> decision.",
        "Do not re-read a file that is already in this session.",
        "Save artifacts and working notes BEFORE calling finish.",
    ]
)


def system_prompt(
    variables: Mapping[str, object] | None = None,
    *,
    explore: bool = False,
    handoff: bool = False,
) -> str:
    """Render the system prompt for one session of the RPent arm.

    ``explore`` enables the notes protocol. ``handoff`` adds the claim that
    earlier sessions left notes, so it applies only from the second session.
    """
    parts: PromptNode = {
        "ROLE": ROLE,
        "MEMORY": MEMORY + (NOTES if explore else ""),
    }
    if handoff:
        parts["EXPLORE"] = EXPLORE
    parts["OUTPUT"] = OUTPUT
    return format_prompt(parts, variables=variables)


def user_prompt(
    instruction: str,
    variables: Mapping[str, object] | None = None,
) -> str:
    """Render the user message that carries the original EFE instruction."""
    return format_prompt(
        {
            "TASK": """
Complete this EFE task:

{{instruction}}

Workspace: {{workdir}}
Run output: {{output_dir}}
""",
        },
        variables={**(variables or {}), "instruction": instruction.strip()},
    )
