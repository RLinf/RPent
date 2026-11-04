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

"""Prompts for an RPent session that drives an EFE standard-mode task."""

from __future__ import annotations

from collections.abc import Mapping

from rpent.prompt.utils import BulletList, PromptNode, format_prompt

ROLE = """You are an RPent agent controlling a robot in an EFE standard-mode task.
The simulator runs as a service you cannot open. You see only what its client returns:
camera frames, the robot's proprioception, and the task's capability spec.
You get one episode and no reset. The episode ends at success, at the time limit, or on a simulator error."""

TOOLS = """- `efe_spec`: action layouts, limits, cameras, proprioception, kinematics. Read it once before acting.
- `efe_observe`: current frames and proprioception. It never moves the robot.
- `efe_status`: calls, control steps, elapsed time, and whether the episode has ended.
- `efe_submit`: send one action chunk in a layout from `efe_spec`.
- `efe_recover`: wait for a crashed service to resume the same episode."""

SKILL_TOOL = (
    "\n- `move_to_xyz`: move the tool frame to a world position with a closed IK loop."
)

RESET_TOOL = (
    "\n- `efe_reset`: abandon the current episode and restart from the frozen "
    "start state while the task grants resets."
)

END_EPISODE_TOOL = (
    "\n- `efe_end_episode`: end the episode now; the verifier grades the state "
    "it ended in."
)

RULES = BulletList(
    [
        "Look before you move: call `efe_observe` and read the frames.",
        "Use the action layout the service reports. Never assume a fixed width.",
        "A rejected call does not advance the simulator; read the error and correct it.",
        "After a connection error, call `efe_recover` and read the status. Never resend a command that may have executed.",
        "The EFE verifier decides success. `finish` ends your session and is not a score.",
        "If the goal is reached, stop acting: the episode may already be over.",
    ]
)

SHELL_RULE = "Shell commands have no network access. Act on the robot only through the tools above."

MEMORY = """RPent memory is a corpus shared across sessions. Read it before acting:
`{{memory_dir}}/global/` and `{{memory_dir}}/task-family/` when they exist.
Apply a lesson only when this task matches its stated conditions."""

OUTPUT = BulletList(
    [
        "Brief reasoning before each tool call (1-2 sentences): observation -> decision.",
        "Do not repeat an observation you already have in this context.",
    ]
)

NOTES = BulletList(
    [
        "Record working notes under `{{memory_inbox}}/wip/notes-session-{{session_number}}.md` before finish.",
    ]
)

EXPLORE = """You are session {{session_number}} of up to {{session_max}} on this task.
Earlier sessions archived notes under `{{memory_inbox}}/wip/`. Read them first.
Do not repeat a failed approach unless you can name the parameter that changes it."""


def system_prompt(
    variables: Mapping[str, object] | None = None,
    *,
    handoff: bool = False,
    skills: bool | None = None,
    write_notes: bool = False,
    restricted_shell: bool = False,
    reset: bool = False,
    end_episode: bool = False,
) -> PromptNode:
    """Render the system prompt for one session of the RPent arm.

    ``skills`` defaults to the run variables so the registry path matches the
    toolkit's tool set. ``write_notes`` follows the memory policy: an
    evaluation session holds the corpus read-only, so the prompt must not ask
    it to write there. ``restricted_shell`` matches a Codex sandbox that
    cannot reach the robot from shell commands. ``reset`` and ``end_episode``
    follow the service spec, so the prompt lists only the tools the task
    exposes.
    """
    if skills is None:
        skills = bool((variables or {}).get("skills", True))
    rules = RULES
    if restricted_shell:
        rules = BulletList((*RULES.items, SHELL_RULE))
    tools = TOOLS + (SKILL_TOOL if skills else "")
    tools += RESET_TOOL if reset else ""
    tools += END_EPISODE_TOOL if end_episode else ""
    parts: PromptNode = {
        "ROLE": ROLE,
        "TOOLS": tools,
        "RULES": rules,
        "MEMORY": MEMORY,
        "OUTPUT": OUTPUT,
    }
    if write_notes:
        parts["NOTES"] = NOTES
    if handoff:
        parts["EXPLORE"] = EXPLORE
    return format_prompt(parts, variables=variables)


def user_prompt(
    variables: Mapping[str, object] | None = None,
) -> PromptNode:
    """Render the user message that carries the original EFE instruction."""
    values = dict(variables or {})
    instruction = str(values.get("instruction") or "").strip()
    return format_prompt(
        {
            "TASK": """
Complete this EFE task:

{{instruction}}

Workspace: {{workdir}}
Run output: {{output_dir}}
""",
        },
        variables={**values, "instruction": instruction},
    )
