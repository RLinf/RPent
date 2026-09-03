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

"""Console dispatchers for the source-editable BEHAVIOR plugin."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def _run_source_module(module: str) -> int:
    source_root = Path(__file__).resolve().parents[2]
    behavior_package = source_root / "robots" / "behavior"
    if not behavior_package.is_dir():
        raise RuntimeError(
            "BEHAVIOR commands require an RPent source checkout with "
            "robots/behavior; a regular wheel is not a complete BEHAVIOR runtime"
        )
    return subprocess.run(
        [sys.executable, "-m", module, *sys.argv[1:]],
        cwd=source_root,
        check=False,
    ).returncode


def download_assets() -> int:
    return _run_source_module("robots.behavior.assets_cli")


def install_runtime() -> int:
    return _run_source_module("robots.behavior.install_runtime")
