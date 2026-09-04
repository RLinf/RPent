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

"""User prompt sections for one Franka task run."""

TASK = """- task_name: {{task_name}}
- instruction: {{instruction}}
- success_criteria: {{success_criteria}}"""

CONSTRAINTS = """{{constraints}}"""

BEGIN = """Call view_env_state with step 0, inspect both camera views and the TCP
state, then execute the task conservatively."""
