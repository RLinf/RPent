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

"""Opt-in APXInf component and public-CLI policy-chain checks on LIBERO-PRO."""

import os
import sys
from collections.abc import Iterator
from pathlib import Path

import numpy as np
import pytest

from rpent.robots.components.pi05_vla_client import Pi05VLAClient
from rpent.utils.daemon import ProcessDaemon, pick_free_port
from rpent.utils.rpc import make_rpc_client, wait_for_ready
from tests.e2e_tests.common import (
    ScriptedToolCall,
    array_summary,
    publish_check,
    require_array,
    run_scripted_policy_chain,
)
from tests.e2e_tests.libero.scenario import LiberoScenario

pytestmark = pytest.mark.skipif(
    os.environ.get("RPENT_RUN_APXINF_INTEGRATION") != "1",
    reason="set RPENT_RUN_APXINF_INTEGRATION=1 with APXInf GPU resources",
)


@pytest.fixture(scope="module")
def apxinf_endpoint(libero_scenario: LiberoScenario) -> Iterator[str]:
    if libero_scenario.variant != "pro":
        pytest.skip("APXInf integration uses LIBERO-PRO")
    model_path = os.environ["APXINF_MODEL_PATH"]
    output = libero_scenario.output_dir / "apxinf"
    output.mkdir(parents=True, exist_ok=True)
    port = pick_free_port()
    command = [
        os.environ.get("APXINF_PYTHON", sys.executable),
        "-m",
        "rpent.robots.components.apxinf_vla_server",
        "--embodiment",
        "libero",
        "--model-path",
        model_path,
        "--transport",
        "http",
        "--host",
        "127.0.0.1",
        "--port",
        str(port),
        "--parent-watch",
    ]
    for variable, flag in (
        ("APXINF_NORM_STATS_PATH", "--norm-stats-path"),
        ("APXINF_TOKENIZER_PATH", "--tokenizer-path"),
    ):
        if os.environ.get(variable):
            command.extend([flag, os.environ[variable]])
    daemon = ProcessDaemon(
        "apxinf_vla",
        command,
        log_path=str(output / "server.log"),
    )
    endpoint = f"http://127.0.0.1:{port}"
    rpc = make_rpc_client(endpoint)
    try:
        daemon.start()
        wait_for_ready(rpc, daemon=daemon, timeout_s=300)
        yield endpoint
    finally:
        rpc.close()
        daemon.stop()
        assert daemon.poll() == 0, "APXInf did not exit cleanly through parent-watch"


def test_apxinf_component(apxinf_endpoint: str, libero_scenario: LiberoScenario):
    observation, _ = libero_scenario.environment
    rpc = make_rpc_client(apxinf_endpoint)
    try:
        client = Pi05VLAClient(rpc, embodiment="libero")
        actions = require_array(
            client.predict(observation),
            "APXInf actions",
            ndim=2,
            last_dim=7,
            dtype=np.float32,
        )
        assert actions.shape == (5, 7)
        assert client.status() == {"action_chunk": 5}
        publish_check(
            "apxinf_component", {"status": "passed", "actions": array_summary(actions)}
        )
    finally:
        rpc.close()


def test_apxinf_policy_chain(apxinf_endpoint: str, libero_scenario: LiberoScenario):
    args = libero_scenario.args
    output = libero_scenario.output_dir / "apxinf-policy-chain"
    result = run_scripted_policy_chain(
        robot="libero",
        robot_argv=[
            "--suite",
            args.suite,
            "--task",
            str(args.task),
            "--seed",
            str(args.seed),
            "--libero-type",
            args.libero_type,
            "--max-episode-steps",
            str(args.max_episode_steps),
            "--cuda-device",
            str(args.cuda_device),
            "--vla-endpoint",
            apxinf_endpoint,
        ],
        output_dir=output,
        action=ScriptedToolCall(
            "pi0_doubled",
            {
                "prompt": "perform one bounded action chunk for the current task",
                "max_chunks": 1,
            },
        ),
        action_count_field="chunks_used",
    )
    # The borrowed endpoint must survive cleanup of the CLI-owned environment.
    rpc = make_rpc_client(apxinf_endpoint)
    try:
        wait_for_ready(rpc, timeout_s=5)
    finally:
        rpc.close()
    assert list(Path(output).rglob("*.png")), "no observation images recorded"
    publish_check("apxinf_policy_chain", result)
