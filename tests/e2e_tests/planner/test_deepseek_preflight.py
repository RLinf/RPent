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

"""Real runtime metadata preflight, without credentials or model requests."""

import pytest

from rpent.planner.deepseek_config import preflight_configuration

deepseek_harness = pytest.importorskip("deepseek_harness")


@pytest.mark.parametrize(
    "effort,error,match",
    [
        ("xhigh", RuntimeError, "rpent-dsh-login"),
        ("not-an-effort", ValueError, "reasoning or image"),
    ],
)
def test_catalog_checks_before_gpu_and_reaps_runtime(
    tmp_path, monkeypatch, effort, error, match
):
    processes = []
    start = deepseek_harness.HarnessClient.start

    def capture(client):
        start(client)
        processes.append(client._proc)

    monkeypatch.setattr(deepseek_harness.HarnessClient, "start", capture)
    home = tmp_path / "empty-home"
    with pytest.raises(error, match=match):
        preflight_configuration(
            dsh_home=home,
            model="gpt-6.1-sol",
            reasoning_effort=effort,
        )
    assert processes and all(process.poll() is not None for process in processes)
    assert not list(home.glob("sessions/**/*.jsonl"))
    assert not (home / ".credentials.yaml").exists()
