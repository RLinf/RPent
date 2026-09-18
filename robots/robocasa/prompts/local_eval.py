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

"""Local-memory section for RoboCasa evaluation."""

MEMORY = """
Use the LOCAL exploration corpus for this evaluation. Read every available
layer relevant to the current task:

1. TASK: {{memory_dir}}/task_only/{{reference_tag}}.json and
   {{memory_dir}}/task_only/{{reference_tag}}_recipe.jsonl.
2. SUITE: the matching task and split entry under {{memory_dir}}/suite/.
3. GLOBAL: {{memory_dir}}/MEMORY.md, followed by only the relevant leaves under
   {{memory_dir}}/global/.

Treat all memory as a strategy prior. Current RGB-D, task progress, and
primitive results always take precedence. Recipes provide phase order and
technique, never coordinates: re-ground all xyz, xy, pixels, base poses, and
fixture geometry in the current episode. Historical entries may name vla_act,
use_prompt, or atomic prompts; use the current rldx_skill / rldx_arm tools with
the complete live task_language. Never read {{memory_dir}}/_internal/ during
evaluation. If a layer is absent, continue with the available validated layers.
"""

__all__ = ["MEMORY"]
