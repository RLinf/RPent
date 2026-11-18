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

"""Offline service double for EFE integration contracts."""

from __future__ import annotations

import os
from collections.abc import Iterator

import pytest

from tests.unit_tests.robots.efe.efe_support import FakeEfeClient


@pytest.fixture(autouse=True)
def _restore_efe_environment() -> Iterator[None]:
    """Undo the workspace bindings one CLI run leaves in the environment."""
    names = ("RPENT_REPO_ROOT", "CODEX_HOME")
    original = {name: os.environ.get(name) for name in names}
    try:
        yield
    finally:
        for name, value in original.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value


@pytest.fixture
def efe_client() -> FakeEfeClient:
    """Return a service double with the public spec and proprioception."""
    return FakeEfeClient()
