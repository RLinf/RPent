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
- layout: {{layout}}
- env_cfg: {{env_cfg_type}}
- action_type: {{action_type}}
- output_dir: {{output_dir}}
- scene (static hint): {{task_summary}}"""

BEGIN = """Call `view_env_state` first to read the instruction and observe the
scene, then localize targets and execute."""

PUT_BOTTLES_CONTEXT = """For put_bottles_into_dustbin:
- Use `place_in_bin` to carry a held bottle to the bin mouth CENTER, descend
  below the rim, release, and retract. Releasing at mouth height can catch the rim.
- When available, `get_safety_status` reports bottle `rolling` / `off_table`
  alarms. For a rolling bottle, use `stabilize` to place an open gripper in its
  path at table height; reassess before resuming. An off-table bottle may be
  unrecoverable.
- `get_reward_details`, when available, includes per-bottle
  `bottles_on_bin_bottom`, `grippers_open`, `arms_home`, score tiers
  (10/25/40/100), and environment `success`. These are privileged diagnostics,
  not visual estimates."""
