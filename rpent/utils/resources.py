# Copyright 2026 RPent Contributors
"""Sync the env's resources/ payload from its HuggingFace dataset."""
from __future__ import annotations

import os
from pathlib import Path

from rpent.utils.config import get_resources_dir
from rpent.utils.logging import get_logger

RESOURCES_HF_REPO = os.environ.get("RPENT_RESOURCES_HF_REPO", "RLinf/RPent-memory")

logger = get_logger("resources")


def _has_local_resources(resources_dir: Path) -> bool:
    return resources_dir.is_dir() and any(path.is_file() for path in resources_dir.rglob("*"))


def ensure_resources(env_name: str) -> Path:
    """Sync an environment's optional resources, or use a pre-downloaded copy."""
    resources_dir = get_resources_dir(env_name)

    if os.environ.get("HF_HUB_OFFLINE") == "1":
        if not _has_local_resources(resources_dir):
            logger.warning(
                "HF_HUB_OFFLINE=1 but no local resources were found under %s; "
                "curated memory and task references for '%s' will be unavailable. "
                "Download the '%s/**' subtree from dataset '%s' before running "
                "offline.",
                resources_dir,
                env_name,
                env_name,
                RESOURCES_HF_REPO,
            )
        return resources_dir

    try:
        from huggingface_hub import snapshot_download

        snapshot_download(
            repo_id=RESOURCES_HF_REPO,
            repo_type="dataset",
            local_dir=str(resources_dir.parent),
            allow_patterns=[f"{env_name}/**"],
        )
    except Exception as exc:
        if _has_local_resources(resources_dir):
            logger.warning(
                "could not sync '%s' from '%s': %s; continuing with local files "
                "under %s",
                env_name,
                RESOURCES_HF_REPO,
                exc,
                resources_dir,
            )
        else:
            logger.warning(
                "could not sync '%s' from '%s': %s; no local resources were found "
                "under %s, so curated memory and task references will be unavailable",
                env_name,
                RESOURCES_HF_REPO,
                exc,
                resources_dir,
            )

    return resources_dir
