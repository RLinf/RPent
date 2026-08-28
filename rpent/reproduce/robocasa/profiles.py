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

"""Frozen planner profiles for RoboCasa reference runs."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:
    from .executor import ExecutorConfig
    from .protocol import Cell


@dataclass(frozen=True)
class PlannerProfile:
    """Immutable planner identity used in result provenance."""

    name: str
    backend: str
    model: str
    reasoning_effort: str


class PlannerAdapter(Protocol):
    """Construct one planner command behind the frozen isolation boundary."""

    backend: str
    audit_id: str
    trusted_symlink_labels: frozenset[str]

    def required_files(self, config: "ExecutorConfig") -> dict[str, Path]:
        """Return backend-owned executable inputs for fail-closed doctor checks."""
        ...

    def build_command(
        self,
        config: "ExecutorConfig",
        *,
        workdir: Path,
        run_id: str,
        cell: "Cell",
    ) -> list[str]:
        """Return an argv vector without embedding planner credentials."""
        ...


PROFILES = {
    "codex-gpt55-xhigh": PlannerProfile(
        name="codex-gpt55-xhigh",
        backend="codex",
        model="gpt-5.5",
        reasoning_effort="xhigh",
    ),
    "claude-opus48": PlannerProfile(
        name="claude-opus48",
        backend="claude_code",
        model="claude-opus-4-8",
        reasoning_effort="high",
    ),
}

# Audit status is release metadata, not part of the frozen scientific profile.
AUDITED_PROFILE_NAMES = frozenset({"codex-gpt55-xhigh"})


def is_audited_profile(profile: PlannerProfile) -> bool:
    """Return whether an exact frozen profile has a formal adapter audit."""
    return (
        profile.name in AUDITED_PROFILE_NAMES and PROFILES.get(profile.name) == profile
    )


def get_profile(name: str) -> PlannerProfile:
    """Resolve one frozen profile without permitting field overrides."""
    try:
        return PROFILES[name]
    except KeyError as exc:
        raise ValueError(
            f"unknown planner profile {name!r}; expected one of {sorted(PROFILES)}"
        ) from exc
