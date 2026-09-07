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

"""BEHAVIOR episode-memory index and validation schema."""

from typing import TYPE_CHECKING, Any

from robots.behavior.memory.schema import (
    MemoryValidationError,
    canonical_json_bytes,
    canonical_json_file_bytes,
    require_sha256,
    sha256_bytes,
)

if TYPE_CHECKING:
    from robots.behavior.memory.index import (
        EpisodeExperience,
        EpisodeFrameKey,
        EpisodeMemoryHit,
        EpisodeMemoryIndex,
        empty_episode_memory_index,
        load_current_catalog,
        load_revision_dir,
        write_candidate_revision,
    )

_INDEX_EXPORTS = frozenset(
    {
        "EpisodeExperience",
        "EpisodeFrameKey",
        "EpisodeMemoryHit",
        "EpisodeMemoryIndex",
        "empty_episode_memory_index",
        "load_current_catalog",
        "load_revision_dir",
        "write_candidate_revision",
    }
)


def __getattr__(name: str) -> Any:
    if name not in _INDEX_EXPORTS:
        raise AttributeError(name)
    from robots.behavior.memory import index

    return getattr(index, name)


__all__ = [
    "EpisodeExperience",
    "EpisodeFrameKey",
    "EpisodeMemoryHit",
    "EpisodeMemoryIndex",
    "MemoryValidationError",
    "canonical_json_bytes",
    "canonical_json_file_bytes",
    "empty_episode_memory_index",
    "load_current_catalog",
    "load_revision_dir",
    "require_sha256",
    "sha256_bytes",
    "write_candidate_revision",
]
