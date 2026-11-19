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

"""User prompt for one RoboTwin run."""

TASK = """- task: {{task_name}}
- seed: {{seed}}
- task_config: {{task_config}}"""

BEGIN = "Follow the RoboTwin startup and memory read order."

WAM_TASK = "Follow the native task_language. Local memory directory: {{memory_dir}}."
WAM_BEGIN = "Inspect view_env_state(step=0) for the native task and current scene."
