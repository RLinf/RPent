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

CELL = """- task: {{task_name}}
- requested_seed: {{seed}}
- task_config: {{task_config}}
- checkpoint: RLinf/LingBot-VLA-RoboTwin-EEF-ckpt1500
"""

BEGIN = """Before acting, read resources/robotwin/memory/MEMORY.md if present, choose only the
memory notes relevant to {{task_name}}, then list resources/robotwin/results and read a successful
{{task_name}} summary and its recipe if available. These files are technique
priors only: never reuse their coordinates. Next inspect view_env_state(step=0)
and the returned images, re-localize the current scene, and use only registered
RoboTwin tools to act. Copy the complete current task_language for every
lingbot_act. The native success predicate and action budget are authoritative."""
