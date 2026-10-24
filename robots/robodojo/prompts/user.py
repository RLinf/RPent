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

"""User prompt sections for a concrete RoboDojo task cell."""

from __future__ import annotations

TASK = """- task:   {{task}}
- seed group: {{seed}}
- layout: {{layout}}
- env_cfg: {{env_cfg_type}}
- output_dir: {{output_dir}}
- scene (static hint): {{task_summary}}"""

BEGIN = """Call `view_env_state` first to read the instruction and observe the
scene. Localize relevant targets, act through the registered primitives,
inspect execution feedback, and record the outcome before finishing."""


POLICY_BEGIN = """Read the original task instruction with `view_env_state`, then
choose bounded `run_policy` calls and inspect their feedback until the episode
ends or execution is interrupted. Report the observed outcome with `finish`."""

MIXED_BEGIN = """Read `view_env_state` for the original task and current images.
Plan from the observation, act through the configured tools, and check their
actual effects."""
