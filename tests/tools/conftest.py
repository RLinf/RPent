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

"""Shared fixtures for the toolkit/tool-spec tests."""

from __future__ import annotations

import pytest

from rpent.utils.logging import init_output_dir


@pytest.fixture(scope="session", autouse=True)
def _output_dir(tmp_path_factory) -> None:
    """Point ``get_output_dir()`` at a scratch dir for the whole session."""
    init_output_dir(tmp_path_factory.mktemp("rpent_toolkit_tests"))
