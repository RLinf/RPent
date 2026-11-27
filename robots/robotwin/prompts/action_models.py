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

"""RoboTwin action-model instructions for the shared planner prompt."""

from rpent.robots.components.uniwam import ACTION_CHUNK_SIZE

DEFAULT_WAM_GUIDANCE = """Use wam_act for one chunk and inspect the new state and images.
Omit prompt for the native task or supply a grounded subtask when needed."""

GUIDANCE = {
    "fast-wam": DEFAULT_WAM_GUIDANCE,
    "lingbot": """Call lingbot_act with the exact complete current task_language
and use_length=50. Use the successful same-task recipe's chunk cadence as a prior.""",
    "uniwam": (
        "Omit prompt in wam_act for the native task or supply a grounded subtask "
        f"for this call. Each chunk executes {ACTION_CHUNK_SIZE} qpos actions; "
        "max_chunks defaults to 1. History includes analytic actions. Choose "
        "chunks from current progress; LingBot recipe counts are not UniWAM timing."
    ),
}
