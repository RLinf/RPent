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

"""Recorded experiment scripts keep their imports and CLI after relocation."""

import importlib
import os
import subprocess
import sys
from pathlib import Path

import pytest


@pytest.mark.parametrize(
    "name",
    [
        "geometry",
        "geometry_agent",
        "reference_geometry",
        "evaluation_scene",
        "localization_checklist",
        "localization_evaluation",
        "localization_experiment",
    ],
)
def test_legacy_import_shares_module_globals_with_tools(name):
    legacy = importlib.import_module(f"robots.libero.{name}")
    current = importlib.import_module(f"tools.libero.{name}")
    assert legacy is current


@pytest.mark.parametrize("package", ["tools.libero", "robots.libero"])
@pytest.mark.parametrize("command", ["--help", "validate-manifest"])
def test_experiment_cli_works_outside_the_checkout(tmp_path, package, command):
    root = Path(__file__).resolve().parents[4]
    result = subprocess.run(
        [sys.executable, "-m", f"{package}.localization_experiment", command],
        cwd=tmp_path,
        env={**os.environ, "PYTHONPATH": str(root)},
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stderr
    output = result.stdout + result.stderr
    if command == "--help":
        assert "collect" in output and "evaluate" in output
    else:
        assert "80 cases, 160 requests" in output
