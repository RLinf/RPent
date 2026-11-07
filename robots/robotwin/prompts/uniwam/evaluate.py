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

"""UniWAM control guidance for the shared RoboTwin prompts."""

from robots.robotwin.prompts import evaluate as base
from rpent.prompt.utils import PromptNode

TRANSFER = (
    """Transfer roles, phase order, required arms, observable gates,
VLA/analytic division, terminal action, and known failures. Historical LingBot
chunk counts are not UniWAM timing: choose chunks from current progress.
"""
    + base.TRANSFER_GEOMETRY
)

CONTROL = (
    """Every wam_act uses the exact complete current task_language.
Each chunk contains 16 qpos actions; max_chunks defaults to 1. Execution history is
maintained automatically across VLA and analytic tools. """
    + base.CONTROL_GUIDANCE
)


def system_prompt() -> PromptNode:
    """Return the shared evaluation prompt with UniWAM control guidance."""
    return base.system_prompt(transfer=TRANSFER, control=CONTROL)
