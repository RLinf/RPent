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

"""Run the frontend controller against delayed HTTP replies without a browser."""

import shutil
import subprocess
from pathlib import Path

import pytest

from rpent.dashboard import server


def test_interrupt_response_cannot_revert_authoritative_snapshot(tmp_path):
    node = shutil.which("node")
    if node is None:
        pytest.skip("Node.js is required for the JavaScript controller contract")
    assets = Path(server.__file__).with_name("static")
    for name in ("interaction.js", "http.js"):
        shutil.copyfile(assets / name, tmp_path / name)
    (tmp_path / "package.json").write_text('{"type":"module"}')
    script = Path(__file__).with_name("fixtures") / "interrupt_race.mjs"
    shutil.copyfile(script, tmp_path / script.name)
    result = subprocess.run(
        [node, "--test", str(tmp_path / script.name)],
        capture_output=True,
        text=True,
        timeout=20,
    )
    assert result.returncode == 0, result.stdout + result.stderr
