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

"""Offline service double for the EFE runner contracts."""

from __future__ import annotations

import pytest

from tests.unit_tests.robots.efe.efe_support import FakeEfeClient


@pytest.fixture
def efe_client() -> FakeEfeClient:
    """Return a service double with the public spec and proprioception."""
    return FakeEfeClient()
