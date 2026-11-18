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

"""Values and checks shared by the EFE robot extension and Harbor adapter."""

from rpent.planner.base import CODEX_SANDBOX_CHOICES

# Codex sandbox modes. An EFE evaluation isolates the robot behind RPent tools;
# ``full-access`` lets the planner's shell reach the simulator directly.
SANDBOX_CHOICES = CODEX_SANDBOX_CHOICES

# Planner errors that mean the session used its time budget instead of
# failing. The Harbor adapter treats such a trial as a graded episode, so the
# same markers decide whether a recorded error is a clean stop.
BUDGET_STOP_MARKERS = (
    "sdk timed out after",
    "planner timed out after",
    "planner timeout expired",
)


def is_budget_stop(error: str | None) -> bool:
    """Whether a planner error records an exhausted time budget."""
    lowered = error.lower() if isinstance(error, str) else ""
    return any(marker in lowered for marker in BUDGET_STOP_MARKERS)
