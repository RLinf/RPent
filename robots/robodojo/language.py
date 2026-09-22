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

"""Resolve public task language using RoboDojo's description semantics."""


def validate_instruction(value: str) -> str:
    """Reject empty language and unresolved template markers at the boundary."""
    if not isinstance(value, str) or not value.strip():
        raise ValueError("RoboDojo instruction must be a nonempty string")
    if "<" in value or ">" in value:
        raise ValueError("RoboDojo instruction contains unresolved template markers")
    return value


def resolve_instruction(env) -> str:
    """Read environment 0's language from the initialized description manager."""
    return validate_instruction(env.obs_manager.desc_manager.get_one_description()[0])
