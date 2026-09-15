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

"""Pinned HF runs must not claim that an unverified cache is the requested revision."""

import sys
from types import SimpleNamespace

import pytest

from rpent.memory import MemoryManager


@pytest.mark.parametrize("offline", [False, True])
@pytest.mark.parametrize("cached", [False, True])
def test_unverified_pinned_sync_fails_instead_of_relabelling_cache(
    tmp_path, monkeypatch, offline, cached
):
    root = tmp_path / "robocasa"
    if cached:
        root.mkdir()
        (root / "old.md").write_text("Memory from an unknown revision")
    monkeypatch.setenv("HF_HUB_OFFLINE", "1" if offline else "0")

    def fail_download(**kwargs):
        raise OSError("simulated unavailable HF service")

    monkeypatch.setitem(
        sys.modules,
        "huggingface_hub",
        SimpleNamespace(snapshot_download=fail_download),
    )
    with pytest.raises(RuntimeError, match="offline mode|Could not sync pinned"):
        MemoryManager(root).sync(remote_repo="RLinf/RPent-memory", revision="c" * 40)


@pytest.mark.parametrize("offline", [False, True])
def test_unpinned_robots_keep_their_local_cache_fallback(
    tmp_path, monkeypatch, offline
):
    root = tmp_path / "libero"
    root.mkdir()
    (root / "old.md").write_text("Previously synced memory")
    monkeypatch.setenv("HF_HUB_OFFLINE", "1" if offline else "0")

    def fail_download(**kwargs):
        raise OSError("simulated unavailable HF service")

    monkeypatch.setitem(
        sys.modules,
        "huggingface_hub",
        SimpleNamespace(snapshot_download=fail_download),
    )
    assert MemoryManager(root).sync(remote_repo="RLinf/RPent-memory") == root
