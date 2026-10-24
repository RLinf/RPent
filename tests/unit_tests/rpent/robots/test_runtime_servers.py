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

"""Offline tests for standalone server command construction."""

from __future__ import annotations

import sys

from rpent.robots.runtime import standalone_server_command
from rpent.utils.config import get_repo_root


def test_standalone_server_command_resolves_existing_script() -> None:
    cmd = standalone_server_command("libero", kind="env", extra_args=["--port", "5555"])

    assert cmd == [
        sys.executable,
        str(get_repo_root() / "robots" / "libero" / "env_server.py"),
        "--port",
        "5555",
    ]


def test_standalone_server_command_defaults_to_no_extra_args() -> None:
    cmd = standalone_server_command("libero", kind="env")

    assert cmd == [
        sys.executable,
        str(get_repo_root() / "robots" / "libero" / "env_server.py"),
    ]


def test_standalone_server_command_returns_none_without_script() -> None:
    assert standalone_server_command("mockbot", kind="env") is None
