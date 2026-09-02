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

"""Contracts for the public RoboCasa Target50 evaluation manifest."""

from __future__ import annotations

import json
import re
from pathlib import Path
from types import SimpleNamespace

from robots.robocasa import robot_spec
from robots.robocasa.eval.result import build_cell_result, write_cell_result
from robots.robocasa.eval.validate_target50 import validate_results

REPO_ROOT = Path(__file__).resolve().parents[4]
MANIFEST_PATH = REPO_ROOT / "robots" / "robocasa" / "eval" / "target50.json"
CONSTRAINTS_PATH = (
    REPO_ROOT / "robots" / "robocasa" / "eval" / "target50-constraints.txt"
)
OVERRIDES_PATH = REPO_ROOT / "robots" / "robocasa" / "eval" / "target50-overrides.txt"
RESULTS_PATH = REPO_ROOT / "robots" / "robocasa" / "eval" / "target50_codex_results.md"
DOCUMENTATION_PATHS = [
    REPO_ROOT / "robots" / "robocasa" / "README.md",
    REPO_ROOT / "docs" / "source-en" / "rst_source" / "usage" / "robocasa.rst",
    REPO_ROOT / "docs" / "source-zh" / "rst_source" / "usage" / "robocasa.rst",
]

EXPECTED_TASKS = {
    "atomic": {
        "CloseBlenderLid",
        "CloseFridge",
        "CloseToasterOvenDoor",
        "CoffeeSetupMug",
        "NavigateKitchen",
        "OpenCabinet",
        "OpenDrawer",
        "OpenStandMixerHead",
        "PickPlaceCounterToCabinet",
        "PickPlaceCounterToStove",
        "PickPlaceDrawerToCounter",
        "PickPlaceSinkToCounter",
        "PickPlaceToasterToCounter",
        "SlideDishwasherRack",
        "TurnOffStove",
        "TurnOnElectricKettle",
        "TurnOnMicrowave",
        "TurnOnSinkFaucet",
    },
    "composite_seen": {
        "DeliverStraw",
        "GetToastedBread",
        "KettleBoiling",
        "LoadDishwasher",
        "PackIdenticalLunches",
        "PrepareCoffee",
        "PreSoakPan",
        "RinseSinkBasin",
        "ScrubCuttingBoard",
        "SearingMeat",
        "SetUpCuttingStation",
        "StackBowlsCabinet",
        "SteamInMicrowave",
        "StirVegetables",
        "StoreLeftoversInBowl",
        "WashLettuce",
    },
    "composite_unseen": {
        "ArrangeBreadBasket",
        "ArrangeTea",
        "BreadSelection",
        "CategorizeCondiments",
        "CuttingToolSelection",
        "GarnishPancake",
        "GatherTableware",
        "HeatKebabSandwich",
        "MakeIceLemonade",
        "PanTransfer",
        "PortionHotDogs",
        "RecycleBottlesByType",
        "SeparateFreezerRack",
        "WaffleReheat",
        "WashFruitColander",
        "WeighIngredients",
    },
}

MEMORYLESS_TASKS = {
    "HeatKebabSandwich",
    "PanTransfer",
    "PortionHotDogs",
    "SeparateFreezerRack",
    "WaffleReheat",
    "WashFruitColander",
    "WeighIngredients",
}


def _manifest() -> dict:
    return json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))


def test_target50_manifest_identity_and_dependencies():
    manifest = _manifest()

    assert manifest["schema_version"] == "1.0"
    assert manifest["protocol_id"] == "robocasa-harness-vla-v1"
    assert manifest["benchmark"] == "RoboCasa365"
    assert manifest["environment_split"] == "target"
    assert manifest["success_source"] == "state.success"
    assert manifest["total_tasks"] == 50
    assert manifest["total_cells"] == 340

    dependencies = manifest["dependencies"]
    assert dependencies["runtime"] == {
        "python": "3.10",
        "cuda": "12.6",
        "constraints_file": "robots/robocasa/eval/target50-constraints.txt",
        "overrides_file": "robots/robocasa/eval/target50-overrides.txt",
        "packages": {
            "mujoco": "3.3.1",
            "numpy": "1.26.4",
            "pydantic": "2.13.5",
            "pydantic-ai-slim": "2.1.0",
            "rlinf-rldx": "1.0.1.post10",
            "rlinf-robocasa365": "1.0.1",
            "torch": "2.7.0",
            "torchvision": "0.22.0",
            "transformers": "4.57.6",
        },
    }
    assert dependencies["robosuite"]["commit"] == (
        "97cfbde4b68d8ec43dad20cf4747297866a6ca2e"
    )
    assert dependencies["rldx_checkpoint"]["revision"] == (
        "587e9ecdcc5e7184fcc17f58713908edff5af041"
    )
    assert dependencies["task_memory"] == {
        "repository": "RLinf/RPent-memory",
        "repository_type": "dataset",
        "revision": "551fc3157b3e56b40a3d3a3b4c7ff81721ebe89b",
        "include_pattern": "robocasa/**",
    }


def test_target50_constraints_match_manifest_packages():
    packages = _manifest()["dependencies"]["runtime"]["packages"]
    constraints = {
        line.strip()
        for line in CONSTRAINTS_PATH.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.startswith("#")
    }

    assert constraints == {f"{name}=={version}" for name, version in packages.items()}


def test_target50_override_freezes_robosuite_source():
    revision = _manifest()["dependencies"]["robosuite"]["commit"]
    overrides = {
        line.strip()
        for line in OVERRIDES_PATH.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.startswith("#")
    }

    assert overrides == {
        f"robosuite @ git+https://github.com/RLinf/robosuite.git@{revision}"
    }


def test_target50_matrix_is_exact_and_has_no_duplicate_cells():
    manifest = _manifest()
    splits = manifest["splits"]

    assert set(splits) == set(EXPECTED_TASKS)
    assert splits["atomic"]["seeds"] == list(range(1, 11))
    assert splits["composite_seen"]["seeds"] == list(range(1, 6))
    assert splits["composite_unseen"]["seeds"] == list(range(1, 6))
    assert splits["atomic"]["timeout_seconds"] == 1800
    assert splits["composite_seen"]["timeout_seconds"] == 3600
    assert splits["composite_unseen"]["timeout_seconds"] == 3600

    tasks: list[str] = []
    cells: list[tuple[str, str, int]] = []
    for split_name, expected_tasks in EXPECTED_TASKS.items():
        split = splits[split_name]
        assert set(split["tasks"]) == expected_tasks
        assert split["task_count"] == len(expected_tasks)
        assert split["cell_count"] == len(expected_tasks) * len(split["seeds"])
        tasks.extend(split["tasks"])
        cells.extend(
            (split_name, task, seed)
            for task in split["tasks"]
            for seed in split["seeds"]
        )

    assert len(tasks) == len(set(tasks)) == manifest["total_tasks"] == 50
    assert len(cells) == len(set(cells)) == manifest["total_cells"] == 340
    assert sum(split["cell_count"] for split in splits.values()) == 340


def test_target50_memory_and_retry_boundaries_are_frozen():
    manifest = _manifest()
    memory = manifest["memory_policy"]

    assert memory["scope"] == "same_task_seed_0"
    assert memory["results_directory"] == "robocasa/results"
    assert memory["required_files"] == [
        "<Task>_s0.json",
        "recipe_<Task>_s0.jsonl",
    ]
    assert memory["optional_files"] == ["<Task>.md"]
    assert memory["use_global_memory"] is False
    assert memory["use_cross_task_memory"] is False
    assert set(memory["tasks_without_memory"]) == MEMORYLESS_TASKS
    assert MEMORYLESS_TASKS < EXPECTED_TASKS["composite_unseen"]

    all_tasks = set().union(*EXPECTED_TASKS.values())
    assert memory["tasks_with_memory"] == len(all_tasks - MEMORYLESS_TASKS) == 43
    assert manifest["retry_policy"] == {
        "retry_infrastructure_failure_without_valid_environment_result": True,
        "retry_valid_task_failure": False,
        "retry_planner_timeout": False,
    }


def test_target50_codex_profile_is_reference_only():
    reference = _manifest()["planner_reference"]

    assert reference == {
        "planner": "codex",
        "model": "gpt-5.5",
        "reasoning_effort": "xhigh",
        "max_turns": 100,
        "runtime_is_planner_agnostic": True,
    }


def test_target50_runtime_protocol_is_frozen():
    assert _manifest()["runtime_protocol"] == {
        "scene_seed_source": "cli_seed_identity",
        "use_reset_seed_override": False,
        "result_schema_version": "1.0",
        "result_filename": "result.json",
        "rldx_max_chunks": 40,
        "rldx_settle_patience": 999,
        "rldx_action_steps_per_chunk": 8,
    }


def test_spawned_environment_uses_cli_seed_and_clears_legacy_override(
    tmp_path, monkeypatch
):
    captured = {}

    class FakeDaemon:
        def __init__(self, **kwargs):
            captured.update(kwargs)

        def start(self):
            captured["started"] = True

    monkeypatch.setattr(robot_spec, "ProcessDaemon", FakeDaemon)
    monkeypatch.setattr(robot_spec, "pick_free_port", lambda: 43210)
    monkeypatch.setattr(robot_spec, "HttpRpcClient", lambda url: url)
    args = SimpleNamespace(
        env_endpoint=None,
        task_name="OpenDrawer",
        split="target",
        seed=7,
        cuda_device=0,
    )

    daemon, client = robot_spec._spawn_env_server(args, tmp_path)

    seed_index = captured["cmd"].index("--seed")
    assert captured["cmd"][seed_index + 1] == "7"
    assert captured["env_overrides"]["RLDX_RESET_SEED"] == ""
    assert captured["started"] is True
    assert daemon is not None
    assert client == "http://127.0.0.1:43210"


def _cell_result(**overrides) -> dict:
    values = {
        "task_name": "OpenDrawer",
        "environment_split": "target",
        "seed": 1,
        "success": False,
        "environment_result_available": True,
        "agent_error": None,
        "elapsed_s": 12.34,
        "planner": "codex",
        "model": "gpt-5.5",
        "reasoning_effort": "xhigh",
        "max_turns": 100,
        "cell_timeout_seconds": 1800,
    }
    values.update(overrides)
    return build_cell_result(**values)


def test_cell_result_uses_environment_success_and_sanitizes_errors(monkeypatch):
    monkeypatch.setenv("RLDX_MAX_CHUNKS", "40")
    monkeypatch.setenv("RLDX_SETTLE_PATIENCE", "999")
    monkeypatch.setenv("RLDX_ACTION_STEPS_PER_CHUNK", "8")

    result = _cell_result(
        success=False,
        agent_error="private provider connection failed at a secret endpoint",
    )

    assert result["protocol_id"] == "robocasa-harness-vla-v1"
    assert result["evaluation_split"] == "atomic"
    assert result["success"] is False
    assert result["success_source"] == "state.success"
    assert result["valid"] is False
    assert result["termination_reason"] == "infrastructure_error"
    assert result["runtime"] == {
        "cell_timeout_seconds": 1800,
        "rldx_max_chunks": 40,
        "rldx_settle_patience": 999,
        "rldx_action_steps_per_chunk": 8,
    }
    assert "secret" not in json.dumps(result)
    assert "finish" not in result


def test_cell_result_counts_planner_timeout_but_not_missing_environment_result():
    timeout = _cell_result(agent_error="Codex SDK timed out after 1800s")
    missing = _cell_result(
        agent_error="Codex SDK timed out after 1800s",
        environment_result_available=False,
    )

    assert timeout["termination_reason"] == "planner_timeout"
    assert timeout["valid"] is True
    assert timeout["success"] is False
    assert missing["valid"] is False


def test_cell_result_writer_is_atomic(tmp_path, monkeypatch):
    monkeypatch.setenv("RLDX_MAX_CHUNKS", "40")

    path = write_cell_result(
        tmp_path,
        **{
            "task_name": "OpenDrawer",
            "environment_split": "target",
            "seed": 1,
            "success": True,
            "environment_result_available": True,
            "agent_error": None,
            "elapsed_s": 12.34,
            "planner": "codex",
            "model": "gpt-5.5",
            "reasoning_effort": "xhigh",
            "max_turns": 100,
            "cell_timeout_seconds": 1800,
        },
    )

    assert path == tmp_path / "result.json"
    assert json.loads(path.read_text(encoding="utf-8"))["success"] is True
    assert not (tmp_path / ".result.json.tmp").exists()


def test_target50_validator_accepts_exactly_all_340_cells(tmp_path, monkeypatch):
    monkeypatch.setenv("RLDX_MAX_CHUNKS", "40")
    manifest = _manifest()

    for split_name, split in manifest["splits"].items():
        for task_name in split["tasks"]:
            for seed in split["seeds"]:
                output_dir = tmp_path / split_name / f"{task_name}_s{seed}"
                output_dir.mkdir(parents=True)
                write_cell_result(
                    output_dir,
                    task_name=task_name,
                    environment_split="target",
                    seed=seed,
                    success=True,
                    environment_result_available=True,
                    agent_error=None,
                    elapsed_s=1.0,
                    planner="codex",
                    model="gpt-5.5",
                    reasoning_effort="xhigh",
                    max_turns=100,
                    cell_timeout_seconds=split["timeout_seconds"],
                )

    summary, errors = validate_results(tmp_path)

    assert errors == []
    assert summary["valid_cells"] == summary["expected_cells"] == 340
    assert summary["overall"] == {
        "metric": "task_weighted_success_rate",
        "tasks": 50,
        "success_rate": 1.0,
    }


def test_published_results_cover_target50_and_match_split_totals():
    manifest = _manifest()
    result_rows = re.findall(
        r"^\| \d+ \| (Atomic|Composite-Seen|Composite-Unseen) "
        r"\| ([A-Za-z0-9]+) \| (\d+)/(\d+) \| (\d+)% \|$",
        RESULTS_PATH.read_text(encoding="utf-8"),
        flags=re.MULTILINE,
    )

    assert len(result_rows) == 50
    by_task = {
        task: {
            "split": split,
            "successes": int(successes),
            "evaluated": int(evaluated),
            "rate": int(rate),
        }
        for split, task, successes, evaluated, rate in result_rows
    }
    assert len(by_task) == 50

    split_labels = {
        "atomic": "Atomic",
        "composite_seen": "Composite-Seen",
        "composite_unseen": "Composite-Unseen",
    }
    expected_successes = {
        "atomic": 163,
        "composite_seen": 49,
        "composite_unseen": 12,
    }
    for split_name, split in manifest["splits"].items():
        rows = [by_task[task] for task in split["tasks"]]
        assert {row["split"] for row in rows} == {split_labels[split_name]}
        assert sum(row["successes"] for row in rows) == expected_successes[split_name]
        assert sum(row["evaluated"] for row in rows) == split["cell_count"]
        for row in rows:
            assert row["evaluated"] == len(split["seeds"])
            assert row["rate"] == 100 * row["successes"] // row["evaluated"]

    assert set(by_task) == set().union(*EXPECTED_TASKS.values())
    task_weighted_rate = sum(
        100 * row["successes"] / row["evaluated"] for row in by_task.values()
    ) / len(by_task)
    assert task_weighted_rate == 57.0
    assert "**Overall (task-weighted)** | **50** | **N/A** | **57.00%**" in (
        RESULTS_PATH.read_text(encoding="utf-8")
    )


def test_target50_documentation_uses_frozen_revisions():
    frozen_values = {
        "robocasa-harness-vla-v1",
        "97cfbde4b68d8ec43dad20cf4747297866a6ca2e",
        "587e9ecdcc5e7184fcc17f58713908edff5af041",
        "551fc3157b3e56b40a3d3a3b4c7ff81721ebe89b",
    }

    for path in DOCUMENTATION_PATHS:
        contents = path.read_text(encoding="utf-8")
        assert all(value in contents for value in frozen_values), path
        assert "target50-constraints.txt" in contents, path
        assert "target50-overrides.txt" in contents, path
        assert "RLDX_MAX_CHUNKS" in contents, path
        assert "RLDX_RESET_SEED" in contents, path
        assert "result.json" in contents, path
