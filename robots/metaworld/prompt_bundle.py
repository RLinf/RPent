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

"""MetaWorld prompt bundle assembly."""

from collections.abc import Mapping

from robots.metaworld.prompts.system import SYSTEM
from robots.metaworld.prompts.user import USER


def system_prompt(variables: Mapping[str, object] | None = None) -> str:
    """Return the MetaWorld system prompt."""
    return SYSTEM


def user_prompt(variables: Mapping[str, object] | None = None) -> str:
    """Return the task-specific user prompt template."""
    return USER
