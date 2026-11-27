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

"""Common prompt for a single evaluation episode."""

from rpent.prompt.workflow import Workflow, user_prompt

__all__ = ["system_prompt", "user_prompt"]

system_prompt = Workflow(
    objective="""Solve the current task in one episode without restarting. Follow a focused
sequence grounded in current evidence, avoid broad exploration, and protect
each achieved subgoal. Do not ask for clarification or defer a determined
action. Complete any confirmation explicitly required by the environment.""",
    retries="After the attempt starts, recover within it. If it ends unsolved, report that result without starting another evaluation attempt.",
).system_prompt
