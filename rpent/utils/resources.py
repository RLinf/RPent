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

"""Sync the robot's resources/ payload from its HuggingFace dataset."""

from __future__ import annotations

import os
from pathlib import Path
from typing import TYPE_CHECKING

from rpent.utils.config import get_resources_dir
from rpent.utils.logging import get_logger

if TYPE_CHECKING:
    from rpent.robots.robot_spec import RobotSpec

logger = get_logger("resources")


def _has_local_resources(resources_dir: Path) -> bool:
    return resources_dir.is_dir() and any(
        path.is_file() for path in resources_dir.rglob("*")
    )


def ensure_resources(robot_spec: "RobotSpec") -> Path:
    """Sync a robot's optional resources, or use a pre-downloaded copy."""
    robot_name = robot_spec.name
    repo_id = os.environ.get(
        "RPENT_RESOURCES_HF_REPO",
        robot_spec.resources_repo_id,
    )
    resources_dir = get_resources_dir(robot_name)

    if os.environ.get("HF_HUB_OFFLINE") == "1":
        if not _has_local_resources(resources_dir):
            logger.warning(
                "HF_HUB_OFFLINE=1 but no local resources were found under %s; "
                "curated memory and task references for '%s' will be unavailable. "
                "Download the '%s/**' subtree from dataset '%s' before running "
                "offline.",
                resources_dir,
                robot_name,
                robot_name,
                repo_id,
            )
        return resources_dir

    try:
        from huggingface_hub import snapshot_download

        snapshot_download(
            repo_id=repo_id,
            repo_type="dataset",
            local_dir=str(resources_dir.parent),
            allow_patterns=[f"{robot_name}/**"],
        )
    except Exception as exc:
        if _has_local_resources(resources_dir):
            logger.warning(
                "could not sync '%s' from '%s': %s; continuing with local files "
                "under %s",
                robot_name,
                repo_id,
                exc,
                resources_dir,
            )
        else:
            logger.warning(
                "could not sync '%s' from '%s': %s; no local resources were found "
                "under %s, so curated memory and task references will be unavailable",
                robot_name,
                repo_id,
                exc,
                resources_dir,
            )

    return resources_dir
