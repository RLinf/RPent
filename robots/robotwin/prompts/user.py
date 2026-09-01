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
- seed: {{seed}}
- task_config: {{task_config}}
- checkpoint: RLinf/LingBot-VLA-RoboTwin-EEF-ckpt1500
"""

BEGIN = """Follow the required read order, bind the current task's targets and
relations from fresh observation, then execute the first unmet recipe phase.
After each action verify its observable gate, preserve achieved relations, and
use the complete current task_language unchanged for every lingbot_act."""
