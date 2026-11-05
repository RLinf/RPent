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

from rpent.prompt.utils import BulletList, PromptNode

ROLE_INTRO = """You are an RPent agent controlling a robot in an EFE standard-mode task.
The simulator runs as a service you cannot open. You see only what its client returns:
camera frames, the robot's proprioception, and the task's capability spec."""

ROLE_ONE_EPISODE = "You get one episode and no reset."
ROLE_RESET_BUDGET = "You get one graded episode and a limited number of resets."

EPISODE_END = "The episode ends at success, at the time limit, or on a simulator error."
EPISODE_END_BY_AGENT = (
    "The service never reports success, so `efe_end_episode` ends the episode; "
    "it also ends at the time limit or on a simulator error."
)

TOOLS = """- `efe_spec`: action layouts, limits, cameras, proprioception, kinematics. Read it once before acting.
- `efe_observe`: current frames and proprioception. It never moves the robot.
- `efe_status`: calls, control steps, elapsed time, and whether the episode has ended.
- `efe_recover`: wait for a crashed service to resume the same episode.
- `view_env_state`: read one recorded step: proprioception, counters, the command that produced it, and any saved artifact names. `step` defaults to -1 (latest)."""

SUBMIT_TOOL = "\n- `efe_submit`: send one action chunk in a layout from `efe_spec`."

SKILL_TOOL = (
    "\n- `move_to_xyz`: move the tool frame to a world position with a closed IK loop."
    "\n- `set_gripper`: set the gripper opening in metres while holding the arm pose."
)

RESET_TOOL = (
    "\n- `efe_reset`: abandon the current episode and restart from the frozen "
    "start state while the task grants resets."
)

END_EPISODE_TOOL = (
    "\n- `efe_end_episode`: end the episode now; the verifier grades the state "
    "it ended in."
)

GOAL_REACHED_BY_SERVICE = (
    "If the goal is reached, stop acting: the episode may already be over."
)
GOAL_REACHED_BY_AGENT = (
    "If the goal is reached, call `efe_end_episode`; the verifier grades the "
    "state it ended in."
)

RULES = BulletList(
    [
        "Look before you move: call `efe_observe` and read the frames.",
        "Use the action layout the service reports. Never assume a fixed width.",
        "A rejected call does not advance the simulator; read the error and correct it.",
        "After a connection error, call `efe_recover` and read the status. Never resend a command that may have executed.",
        "The EFE verifier decides success. `finish` ends your session and is not a score.",
        GOAL_REACHED_BY_SERVICE,
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


def role(*, reset: bool = False, end_episode: bool = False) -> str:
    """Render the episode rules that match the task's declared capabilities.

    The service spec decides both facts, so a static line would contradict the
    tools registered for the same task.
    """
    episode = ROLE_RESET_BUDGET if reset else ROLE_ONE_EPISODE
    ending = EPISODE_END_BY_AGENT if end_episode else EPISODE_END
    return f"{ROLE_INTRO}\n{episode} {ending}"


def system_prompt(variables: Mapping[str, object] | None = None) -> PromptNode:
    """Return the ordered prompt sections for one session of the RPent arm.

    ``PromptBundle.render`` fills the variables. ``skills``, ``reset``, and
    ``end_episode`` come from the run variables so the prompt describes the
    tools the session registers. ``write_notes`` follows the memory policy: an
    evaluation session holds the corpus read-only, so the prompt must not ask
    it to write there. ``restricted_shell`` matches a Codex sandbox that
    cannot reach the robot from shell commands.
    """
    values = variables or {}
    skills = bool(values.get("skills", True))
    reset = bool(values.get("reset", False))
    end_episode = bool(values.get("end_episode", False))
    rule_items = list(RULES.items)
    if end_episode:
        rule_items[-1] = GOAL_REACHED_BY_AGENT
    if bool(values.get("restricted_shell", False)):
        rule_items.append(SHELL_RULE)
    rules = BulletList(rule_items)
    tools = TOOLS + (SKILL_TOOL if skills else SUBMIT_TOOL)
    tools += RESET_TOOL if reset else ""
    tools += END_EPISODE_TOOL if end_episode else ""
    parts: dict[str, PromptNode] = {
        "ROLE": role(reset=reset, end_episode=end_episode),
        "TOOLS": tools,
        "RULES": rules,
        "MEMORY": MEMORY,
        "OUTPUT": OUTPUT,
    }
    if bool(values.get("write_notes", False)):
        parts["NOTES"] = NOTES
    if bool(values.get("handoff", False)):
        parts["EXPLORE"] = EXPLORE
    return parts


def user_prompt(variables: Mapping[str, object] | None = None) -> PromptNode:
    """Return the user message sections that carry the EFE instruction."""
    return {
        "TASK": (
            "Complete this EFE task:\n\n"
            "{{instruction}}\n"
            "Workspace: {{workdir}}\n"
            "Run output: {{output_dir}}\n"
        )
    }
