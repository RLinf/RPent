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

"""Unit tests for fine-grained CI target selection and multi-GPU suite parsing."""

from __future__ import annotations

from pathlib import Path

import pytest

from tests.e2e_tests.suite import component_cache_hash, resolve_value
from tests.tooling.ci_targets import plan_ci

REPO_ROOT = Path(__file__).resolve().parents[3]


def test_docs_only_skips_all_install_and_eval_jobs() -> None:
    plan = plan_ci(
        [
            "docs/source-en/rst_source/development/testing.rst",
            "README.md",
            "tests/unit_tests/rpent/install/test_installer.py",
        ],
        REPO_ROOT,
        environ={},
    )
    assert plan["install_jobs"] == []
    assert plan["eval_jobs"] == []
    assert plan["suites"] == []


def test_single_model_runtime_change_runs_only_matching_eval_case_cached() -> None:
    plan = plan_ci(
        ["components/models/awomo05/server.py"],
        REPO_ROOT,
        environ={},
    )
    assert plan["install_jobs"] == []
    assert [j["id"] for j in plan["eval_jobs"]] == ["robodojo:awomo05"]
    assert plan["eval_jobs"][0]["venv_mode"] == "cached"
    assert plan["eval_jobs"][0]["gpus"] == 2


def test_single_model_manifest_change_runs_component_and_robot_install_plus_eval() -> (
    None
):
    plan = plan_ci(
        ["components/models/rldx/component.toml"],
        REPO_ROOT,
        environ={},
    )
    installs = {j["component"]: j["kind"] for j in plan["install_jobs"]}
    assert installs == {"robot-robocasa": "robot", "model-rldx": "component"}
    assert [j["id"] for j in plan["eval_jobs"]] == ["robocasa:rldx"]
    assert plan["eval_jobs"][0]["venv_mode"] == "run"


def test_shared_perception_model_triggers_only_cases_using_it() -> None:
    plan = plan_ci(
        ["components/models/sam3/server.py"],
        REPO_ROOT,
        environ={},
    )
    assert plan["install_jobs"] == []
    eval_ids = {j["id"] for j in plan["eval_jobs"]}
    assert "libero:pi05" in eval_ids
    assert "libero-pro:pi05" in eval_ids
    assert "robodojo:awomo05" in eval_ids
    assert "robocasa:rldx" not in eval_ids
    assert "robolab:pi05" not in eval_ids
    assert "robotwin:lingbot" not in eval_ids


def test_core_runtime_change_skips_install_and_runs_all_cached_evals() -> None:
    plan = plan_ci(
        ["rpent/planner/api.py"],
        REPO_ROOT,
        environ={},
    )
    assert plan["install_jobs"] == []
    eval_ids = {j["id"] for j in plan["eval_jobs"]}
    assert {"libero:pi05", "robocasa:rldx", "robodojo:awomo05", "robotwin:lingbot"} <= (
        eval_ids
    )
    # Opt-in cosmos cases without endpoints are omitted.
    assert "libero:cosmos-policy" not in eval_ids
    assert all(j["venv_mode"] == "cached" for j in plan["eval_jobs"])


def test_global_install_change_runs_full_install_and_run_mode_evals() -> None:
    plan = plan_ci(
        ["rpent/install/installer.py"],
        REPO_ROOT,
        environ={},
    )
    install_comps = {j["component"] for j in plan["install_jobs"]}
    assert {
        "robot-libero",
        "robot-libero_plus",
        "robot-robocasa",
        "robot-robodojo",
        "robot-robolab",
        "robot-robotwin",
        "robot-franka",
        "infra-dev",
    } <= install_comps
    assert plan["eval_jobs"]
    assert all(j["venv_mode"] == "run" for j in plan["eval_jobs"])


def test_resolve_value_multi_gpu_placeholders_and_cache_hash() -> None:
    assert (
        resolve_value(
            "--env {gpu0} --vla {gpu1} --default {gpu}",
            shared="/shared",
            gpus=["2", "5"],
        )
        == "--env 2 --vla 5 --default 2"
    )
    with pytest.raises(ValueError, match="gpu1"):
        resolve_value("--vla {gpu1}", shared="/shared", gpus=["2"])

    h1 = component_cache_hash("robot-libero", REPO_ROOT)
    h2 = component_cache_hash("robot-libero", REPO_ROOT)
    assert len(h1) == 12
    assert h1 == h2


def test_openpi_manifest_change_installs_all_consuming_e2e_robots() -> None:
    plan = plan_ci(
        ["components/models/openpi/component.toml"],
        REPO_ROOT,
        environ={},
    )
    install_comps = {j["component"] for j in plan["install_jobs"]}
    assert {
        "model-openpi",
        "robot-libero",
        "robot-robodojo",
        "robot-robolab",
    } <= install_comps


def test_component_cache_hash_tracks_cli_install_runner_and_root_suites(
    tmp_path: Path,
) -> None:
    (tmp_path / "robots" / "demo").mkdir(parents=True)
    (tmp_path / "robots" / "demo" / "component.toml").write_text(
        'schema = 2\nname = "robot-demo"\n', encoding="utf-8"
    )
    (tmp_path / "rpent" / "cli").mkdir(parents=True)
    cli_install = tmp_path / "rpent" / "cli" / "install.py"
    cli_install.write_text("# v1\n", encoding="utf-8")
    (tmp_path / "tests" / "e2e_tests" / "suites").mkdir(parents=True)
    runner = tmp_path / "tests" / "e2e_tests" / "run_gpu_suite.sh"
    runner.write_text("#!/bin/bash\n# v1\n", encoding="utf-8")
    suite_toml = tmp_path / "tests" / "e2e_tests" / "suites" / "demo.toml"
    suite_toml.write_text(
        'component = "robot-demo"\n[setup]\nmaterialize_pkgs = ["pkg-a"]\n',
        encoding="utf-8",
    )

    h0 = component_cache_hash("robot-demo", tmp_path)
    cli_install.write_text("# v2\n", encoding="utf-8")
    h1 = component_cache_hash("robot-demo", tmp_path)
    assert h1 != h0

    runner.write_text("#!/bin/bash\n# v2\n", encoding="utf-8")
    h2 = component_cache_hash("robot-demo", tmp_path)
    assert h2 != h1

    suite_toml.write_text(
        'component = "robot-demo"\n[setup]\nmaterialize_pkgs = ["pkg-b"]\n',
        encoding="utf-8",
    )
    h3 = component_cache_hash("robot-demo", tmp_path)
    assert h3 != h2
