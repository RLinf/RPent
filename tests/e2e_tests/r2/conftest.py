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

"""Run the R2 env server against RLinf's simulated R2 controller."""

from __future__ import annotations

import subprocess
import sys
import time
from collections.abc import Iterator
from pathlib import Path

import pytest
import yaml

from robots.r2.runtime_config import DEFAULT_CONFIG_PATH
from rpent.utils.daemon import pick_free_port
from rpent.utils.rpc import make_rpc_client


@pytest.fixture(scope="module")
def r2_mock_endpoint(tmp_path_factory: pytest.TempPathFactory) -> Iterator[str]:
    """Start ``robots.r2.env_server --mock`` and yield its endpoint."""
    pytest.importorskip("rlinf.envs.real.r2", reason="needs RLinf with R2 support")
    import rlinf

    if not (Path(rlinf.__file__).resolve().parents[1] / "tests/robot_mocks").is_dir():
        pytest.skip("the simulated R2 controller needs RLinf from a source checkout")
    site = yaml.safe_load(DEFAULT_CONFIG_PATH.read_text())
    # The robot's URDF exists only on the robot; camera poses are then absent.
    site["perception"]["urdf"] = None
    workdir = tmp_path_factory.mktemp("r2_mock")
    config = workdir / "site.yaml"
    config.write_text(yaml.safe_dump(site))
    port = pick_free_port()
    endpoint = f"http://127.0.0.1:{port}"
    with (workdir / "env_server.log").open("w") as log:
        server = subprocess.Popen(
            [
                sys.executable,
                "-m",
                "robots.r2.env_server",
                "--mock",
                "--robot-config",
                str(config),
                "--port",
                str(port),
            ],
            cwd=Path(__file__).resolve().parents[3],
            stdin=subprocess.DEVNULL,
            stdout=log,
            stderr=subprocess.STDOUT,
        )
    rpc = make_rpc_client(endpoint)
    try:
        deadline = time.monotonic() + 120.0
        while True:
            if server.poll() is not None:
                raise RuntimeError(
                    f"R2 env server exited with {server.returncode}; "
                    f"see {workdir / 'env_server.log'}"
                )
            try:
                rpc.call("healthz", timeout_s=2.0)
                break
            except Exception:
                if time.monotonic() > deadline:
                    raise
                time.sleep(0.5)
        yield endpoint
    finally:
        try:
            rpc.call("shutdown", timeout_s=30.0)
        except Exception:
            pass
        try:
            server.wait(timeout=60.0)
        except subprocess.TimeoutExpired:
            server.kill()
            server.wait()
