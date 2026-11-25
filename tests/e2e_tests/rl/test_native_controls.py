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

"""Opt-in native control and RGB-D checks for the shared workflow."""

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest


@pytest.mark.timeout(1800)
@pytest.mark.parametrize(
    "simulator,task,seed",
    [("robocasa", "OpenDrawer", 1), ("robotwin", "adjust_bottle", 100000)],
)
def test_shared_native_controls(tmp_path, simulator, task, seed):
    if os.environ.get("RPENT_RUN_RL_INTEGRATION") != "1":
        pytest.skip(
            "set RPENT_RUN_RL_INTEGRATION=1 with simulator assets and shared SAM3"
        )
    endpoint = os.environ["RPENT_SAM3_ENDPOINT"]
    script = Path(__file__).with_name("scenario.py")
    command = [
        sys.executable,
        str(script),
        "--simulator",
        simulator,
        "--task-name",
        task,
        "--seed",
        str(seed),
        "--cuda-device",
        os.environ.get("RPENT_TEST_GPU", "1"),
        "--sam3-endpoint",
        endpoint,
        "--output-dir",
        str(tmp_path),
    ]
    result = subprocess.run(command, capture_output=True, text=True, timeout=1700)
    assert result.returncode == 0, result.stdout + result.stderr
    evidence = json.loads((tmp_path / "verified.json").read_text())
    assert evidence["passed"] and evidence["env_steps"] > 0
    assert len(evidence["cameras"]) == 3
