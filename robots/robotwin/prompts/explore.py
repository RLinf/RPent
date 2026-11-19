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

"""RoboTwin session budgets and the file contract for exploration."""

from robots.robotwin.prompts import system

READ_ORDER = """Before the first robot mutation:
1. Read robots/robotwin/guides/GUIDE_RPENT.md completely.
2. Inspect view_env_state(step=0) and its head image.
3. Read relevant published task, task-family, and global memory.
4. Read {{memory_inbox}}/wip/ for notes from earlier attempts or sessions."""

RECOVERY = (
    system.BUDGET
    + "\n\n"
    + """Session {{session_number}} of {{session_max}}; attempt budget:
{{explore_attempts_per_session}} (0 means no attempt-count limit). Before reset,
write {{output_dir}}/attempts/attempt_<N>_failed.json with the approach,
commands, parameters, observed progress, failure evidence and next change.
Append a handoff note to {{memory_inbox}}/wip/notes.md. Reset may resample
the layout. An unsolved finish is refused while attempts remain."""
)

SUCCESS = (
    system.SUCCESS
    + "\n\n"
    + """Before finish, write {{output_dir}}/{{recipe_tag}}.json with task_name,
task_config, seed, eval_success, total attempts, final_state and successful_strategy.
On success, successful_strategy lists every recipe-eligible command and actual
arguments after the final reset, in order. The runner exports the recipe.
Put successful task-family or global proposals directly under {{memory_inbox}}/.

Use suite robotwin, regime {{task_config}}, task_id {{task_name}}, and evidence
cell {{recipe_tag}}. For a successful task-family proposal include the attempt
count in evidence.attempts, solved_seeds: [{{seed}}], and failed_seeds: [].
Use id task-family_robotwin_{{task_name}} or global_<kind>_<short-name>
according to scope, and related: [] when there are no related entries."""
)
