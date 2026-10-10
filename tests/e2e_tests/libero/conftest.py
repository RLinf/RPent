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

import os
from pathlib import Path

import pytest

from tests.e2e_tests.libero.scenario import LiberoScenario


@pytest.fixture(params=["cosmos-policy", "fast-wam"])
def wam_argv(request) -> list[str]:
    """Select a provisioned worker without requiring Pi0.5 or SAM3 weights."""
    from robots.libero.suites import suite_variant

    backend = request.param
    variable = (
        "RPENT_COSMOS_ENDPOINT"
        if backend == "cosmos-policy"
        else "RPENT_FAST_WAM_ENDPOINT"
    )
    endpoint = os.getenv(variable)
    if not endpoint:
        pytest.skip(f"requires {variable} and LIBERO assets")
    suite = os.getenv("RPENT_WAM_SUITE", "libero_spatial")
    return [
        "--suite",
        suite,
        "--task",
        "0",
        "--seed",
        "0",
        "--libero-type",
        suite_variant(suite),
        "--max-episode-steps",
        "32",
        "--wam-backend",
        backend,
        "--wam-endpoint",
        endpoint,
    ]


@pytest.fixture(scope="session")
def libero_scenario() -> LiberoScenario:
    return LiberoScenario(
        variant=os.environ["RPENT_LIBERO_VARIANT"],
        output_dir=Path(os.environ["RPENT_E2E_OUTPUT_DIR"]),
    )
