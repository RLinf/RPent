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

"""Model-based corpus selection shared by robot memory integrations."""

from __future__ import annotations

from argparse import Namespace
from collections.abc import Mapping

from rpent.planner.base import resolve_model
from rpent.utils.logging import get_logger

logger = get_logger("memory")


def validate_options(args: Namespace) -> None:
    """Reject HF corpus selectors when a local directory owns the memory."""
    if getattr(args, "memory_version", "auto") != "auto" and (
        getattr(args, "explore", False)
        or getattr(args, "memory_profile", None) == "local"
    ):
        raise ValueError(
            "--memory-version requires --memory-profile hf; "
            "local memory and exploration use --memory-dir"
        )


def select_model_version(
    version: str = "auto",
    *,
    model: str | None = None,
    planner: str = "api",
    model_versions: Mapping[str, str],
    default_version: str,
) -> str:
    """Choose a published corpus without changing the planner's reasoning effort."""
    if version not in {"auto", default_version, *model_versions.values()}:
        raise ValueError(f"Unknown memory version: {version!r}")
    if version != "auto":
        return version
    resolved = resolve_model(planner, model)
    name = resolved.rsplit(":", 1)[-1].lower() if resolved else ""
    if name in model_versions:
        return model_versions[name]
    logger.warning(
        "No memory mapping for model %r (%s); using %s. "
        "Use --memory-version to select explicitly.",
        resolved,
        planner,
        default_version,
    )
    return default_version
