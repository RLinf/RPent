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

"""Global prompt definitions shared by robots."""

from __future__ import annotations

from rpent.prompt.utils import BulletList

OUTPUT = BulletList(
    [
        "Brief reasoning before each tool call (1-2 sentences): observation -> decision.",
        "Don't re-read files already in this session. Don't view_env_state right after a primitive tool already returned the state.",
        "Numerical coords in 3 decimals are enough.",
        "Save artifacts BEFORE calling finish. Stop immediately after writing the audit; do not chat further.",
    ]
)

USER = {
    "Task": """
    - suite:   {{suite}}
    - task:    {{task}}
    - seed:    {{seed}}
    - output_dir: {{output_dir}}
    - output:  {{output_dir}}/
      - audit filename:  {{recipe_tag}}.json
    """,
}
