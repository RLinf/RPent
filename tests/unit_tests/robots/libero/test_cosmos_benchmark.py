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

"""Offline regression coverage for exact Cosmos benchmark action budgets."""

import json
import sys
from contextlib import contextmanager
from unittest.mock import Mock

import numpy as np
import pytest

from tests.e2e_tests.libero import benchmark_cosmos_policy as benchmark


@pytest.mark.parametrize("success_step", [3, None])
def test_pro_benchmark_records_native_verdict_and_exact_budget(
    tmp_path, monkeypatch, success_step
):
    output = tmp_path / "results"
    env = Mock(terminated=False, truncated=False)
    env.get_task_language.return_value = "perturbed instruction"
    steps = []

    def step(action):
        steps.append(action)
        env.terminated = len(steps) == success_step
        env.truncated = len(steps) == 5

    env.step.side_effect = step
    model = Mock(predict=Mock(return_value=np.zeros((16, 7))))

    @contextmanager
    def runtime(spec, args, output_dir, components):
        assert args.suite == "libero_spatial_task"
        assert args.libero_type == "pro"
        assert components == {"env", "vla"}
        yield {"env": env, "model": model}

    monkeypatch.setattr(benchmark, "runtime_phase", runtime)
    monkeypatch.setattr(benchmark, "save_scene", lambda *args: None)
    monkeypatch.setattr(benchmark, "init_output_dir", lambda *args: None)
    monkeypatch.setattr(benchmark, "version", lambda name: "test")
    monkeypatch.setattr(benchmark.subprocess, "check_output", lambda *a, **k: "test")
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "benchmark",
            "--endpoint",
            "http://localhost:8116",
            "--output-dir",
            str(output),
            "--suite",
            "libero_spatial_task",
            "--tasks",
            "0",
            "--seeds",
            "0",
            "--horizon",
            "5",
            "--warmup",
            "0",
            "--samples",
            "0",
        ],
    )
    env.raw_obs.return_value = {}
    benchmark.main()
    result = json.loads((output / "results.json").read_text())
    episode = result["episodes"][0]
    assert episode["steps"] == (success_step or 5)
    assert episode["success"] == (success_step is not None)
    assert episode["truncated"] == (success_step is None)
    assert episode["instruction"] == "perturbed instruction"
    assert result["successes"] == int(success_step is not None)
    assert result["total_output_tokens"] == 0
    assert result["errors"] == 0
    assert len(episode["rpc_seconds"]) == 1
    assert (
        model.predict.call_args.args[0]["task_descriptions"] == "perturbed instruction"
    )
