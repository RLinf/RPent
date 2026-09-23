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

"""Opt-in real Cosmos Policy component and bounded LIBERO chain checks."""

from __future__ import annotations

import os

import pytest

from robots.libero.robot_spec import get_robot_spec
from tests.e2e_tests.common import (
    ScriptedToolCall,
    parse_runtime_args,
    require_array,
    run_scripted_policy_chain,
    runtime_phase,
)

pytestmark = [
    pytest.mark.skipif(
        not os.environ.get("RPENT_COSMOS_ENDPOINT"),
        reason="requires RPENT_COSMOS_ENDPOINT and standard LIBERO assets",
    ),
    pytest.mark.timeout(1200),
]


def _argv() -> list[str]:
    return [
        "--suite",
        "libero_spatial",
        "--task",
        "0",
        "--seed",
        "0",
        "--libero-type",
        "standard",
        "--max-episode-steps",
        "32",
        "--vla-backend",
        "cosmos-policy",
        "--vla-endpoint",
        os.environ["RPENT_COSMOS_ENDPOINT"],
    ]


def test_cosmos_predicts_from_real_libero_observation(tmp_path) -> None:
    spec = get_robot_spec()
    args = parse_runtime_args(spec, _argv())
    with runtime_phase(spec, args, tmp_path / "component", {"env", "vla"}) as runtime:
        env = runtime["env"]
        env.reset()
        raw = {**env.raw_obs(), "task_descriptions": env.get_task_language()}
        actions = runtime["model"].predict(raw)
        assert require_array(actions, "Cosmos actions", ndim=2, last_dim=7).shape == (
            16,
            7,
        )
        obs, *_ = env.step(actions[0])
        require_array(obs["main_images"], "next image", ndim=3, last_dim=3)


def test_cosmos_policy_chain(tmp_path) -> None:
    result = run_scripted_policy_chain(
        robot="libero",
        robot_argv=_argv(),
        output_dir=tmp_path / "chain",
        action=ScriptedToolCall("cosmos_act", {"max_chunks": 1}),
        action_count_field="chunks",
    )
    assert result["status"] == "passed"
    assert list((tmp_path / "chain" / "agentview_high.png").glob("*.png"))
