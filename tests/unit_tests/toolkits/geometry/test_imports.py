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

"""Geometry toolkit imports and its standalone evaluation CLI stay CPU-only."""

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest


def test_geometry_imports_and_checklist_do_not_load_optional_robot_stacks(tmp_path):
    root = Path(__file__).resolve().parents[4]
    code = """
import sys
from toolkits.geometry.geometry_agent import GeometryAgent
from toolkits.geometry.evaluation.experiment import main
assert GeometryAgent.__name__ == "GeometryAgent"
assert GeometryAgent.geometry.name == "geometry"
assert GeometryAgent.geometry.readonly
assert main(["validate-manifest"]) == 0
assert not {"torch", "rlinf", "mujoco", "robosuite", "sam3"}.intersection(sys.modules)
"""
    result = subprocess.run(
        [sys.executable, "-c", code],
        cwd=tmp_path,
        env={**os.environ, "PYTHONPATH": str(root)},
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize("command", ["--help", "validate-manifest", "truth-template"])
def test_experiment_cli_works_outside_the_checkout(tmp_path, command):
    root = Path(__file__).resolve().parents[4]
    output = tmp_path / "truth.json"
    args = [command]
    if command == "truth-template":
        args.extend(["--output", str(output)])
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "toolkits.geometry.evaluation.experiment",
            *args,
        ],
        cwd=tmp_path,
        env={**os.environ, "PYTHONPATH": str(root)},
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stderr
    text = result.stdout + result.stderr
    if command == "--help":
        assert "collect" in text and "evaluate" in text
    elif command == "validate-manifest":
        assert "80 cases, 160 requests" in text
    else:
        assert len(json.loads(output.read_text())["annotations"]) == 160
    assert not (tmp_path / "test-cases.json").exists()
