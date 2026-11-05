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

"""Values shared by the EFE CLI, the Harbor adapter, and the runner."""

# Codex sandbox modes. An EFE evaluation isolates the robot behind RPent tools;
# ``full-access`` lets the planner's shell reach the simulator directly.
SANDBOX_CHOICES = ("read-only", "workspace-write", "full-access")
