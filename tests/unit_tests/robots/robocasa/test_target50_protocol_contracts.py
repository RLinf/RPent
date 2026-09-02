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

REPO_ROOT = Path(__file__).resolve().parents[4]
MANIFEST_PATH = REPO_ROOT / "robots" / "robocasa" / "eval" / "target50.json"
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
