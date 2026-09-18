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

import re

_LABEL = re.compile(r"<([^<>]+)>")


def validate_instruction(value: str) -> str:
    """Reject empty language and unresolved template markers at the boundary."""
    if not isinstance(value, str) or not value.strip():
        raise ValueError("RoboDojo instruction must be a nonempty string")
    if "<" in value or ">" in value:
        raise ValueError("RoboDojo instruction contains unresolved template markers")
    return value


def resolve_instruction(env) -> str:
    """Prefer the official manager; resolve labels explicitly if unavailable.

    The fallback selects the first supplied description deterministically. It
    never substitutes a task name or an invented object description on failure.
    """
    manager_error = None
    try:
        value = env.obs_manager.desc_manager.get_one_description()[0]
        return validate_instruction(value)
    except (
        AttributeError,
        IndexError,
        KeyError,
        TypeError,
        ValueError,
        RuntimeError,
    ) as exc:
        manager_error = exc
    try:
        template = env.gen_instruction(0)[0]
        if not isinstance(template, str):
            raise ValueError("Task template must be a string")
        for label in dict.fromkeys(_LABEL.findall(template)):
            descriptions = env.scene_manager.layout_manager.get_label_descriptions(
                label=label, env_idx=0
            )
            if not len(descriptions):
                raise ValueError(f"No public description for label {label!r}")
            description = validate_instruction(descriptions[0])
            template = template.replace(f"<{label}>", description)
        return validate_instruction(template)
    except (
        AttributeError,
        IndexError,
        KeyError,
        TypeError,
        ValueError,
        RuntimeError,
    ) as exc:
        raise RuntimeError(
            f"Cannot resolve RoboDojo task instruction: {exc}; "
            f"description manager: {manager_error}"
        ) from exc
