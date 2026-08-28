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

"""The immutable RoboCasa365 paper-reproduction matrix."""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum
from typing import Any


class Split(str, Enum):
    ATOMIC = "atomic"
    COMPOSITE_SEEN = "composite_seen"
    COMPOSITE_UNSEEN = "composite_unseen"


ALLOWED_ACTIONS = frozenset(
    {
        "navigate_to",
        "move_base",
        "move_to",
        "move_delta",
        "rotate_pitch",
        "set_gripper",
        "release",
        "vla_act",
    }
)

ACTION_PARAMETERS = {
    "navigate_to": (frozenset({"xy"}), frozenset({"tol", "max_steps", "gripper"})),
    "move_base": (
        frozenset(),
        frozenset({"forward", "lateral", "turn", "steps", "gripper"}),
    ),
    "move_to": (
        frozenset({"xyz"}),
        frozenset({"gripper", "step_clip", "max_steps", "tol"}),
    ),
    "move_delta": (
        frozenset({"dxyz"}),
        frozenset({"gripper", "step_clip", "max_steps"}),
    ),
    "rotate_pitch": (
        frozenset(),
        frozenset({"target_pitch", "gripper", "n"}),
    ),
    "set_gripper": (frozenset(), frozenset({"gripper", "steps"})),
    "release": (frozenset(), frozenset({"steps"})),
    "vla_act": (frozenset({"prompt"}), frozenset()),
}


def _finite_number(value: Any) -> bool:
    return (
        not isinstance(value, bool)
        and isinstance(value, (int, float))
        and math.isfinite(value)
    )


def _vector(value: Any, size: int) -> bool:
    return (
        isinstance(value, list)
        and len(value) == size
        and all(_finite_number(item) for item in value)
    )


def _positive_int(value: Any, maximum: int = 10_000) -> bool:
    return type(value) is int and 1 <= value <= maximum


def _gripper(value: Any) -> bool:
    return value == "hold" or _finite_number(value)


def command_problem(command: Any, *, task_language: str | None = None) -> str | None:
    """Return why a formal command is invalid, or ``None`` when it is exact."""
    if not isinstance(command, dict):
        return "command must be an object"
    action = command.get("action")
    if action not in ALLOWED_ACTIONS:
        return "disallowed primitive"
    required, optional = ACTION_PARAMETERS[action]
    keys = set(command) - {"action"}
    missing = required - keys
    extras = keys - required - optional
    if missing:
        return f"missing parameters: {sorted(missing)}"
    if extras:
        return f"unexpected parameters: {sorted(extras)}"

    vector_fields = {"xy": 2, "xyz": 3, "dxyz": 3}
    for name, size in vector_fields.items():
        if name in command and not _vector(command[name], size):
            return f"{name} must be a finite {size}-vector"
    for name in ("max_steps", "steps", "n"):
        if name in command and not _positive_int(command[name]):
            return f"{name} must be an integer in [1, 10000]"
    for name in ("forward", "lateral", "turn"):
        if name in command and not _finite_number(command[name]):
            return f"{name} must be finite"
    if "gripper" in command:
        if action in {"set_gripper", "rotate_pitch"} and not _finite_number(
            command["gripper"]
        ):
            return f"{action} gripper must be finite"
        if action not in {"set_gripper", "rotate_pitch"} and not _gripper(
            command["gripper"]
        ):
            return "gripper must be finite or 'hold'"
    if "step_clip" in command and not (
        _finite_number(command["step_clip"]) and 0 < command["step_clip"] <= 0.30
    ):
        return "step_clip must be finite and in (0, 0.30]"
    if "tol" in command:
        maximum = 5.0 if action == "navigate_to" else 1.0
        if not (_finite_number(command["tol"]) and 0 < command["tol"] <= maximum):
            return f"{action} tol must be finite and in (0, {maximum}]"
    if "target_pitch" in command and not (
        _finite_number(command["target_pitch"])
        and -1.5 <= command["target_pitch"] <= 1.5
    ):
        return "target_pitch must be finite and in [-1.5, 1.5]"
    if action == "vla_act":
        prompt = command.get("prompt")
        if not isinstance(prompt, str) or not prompt.strip():
            return "vla_act prompt must be a non-empty string"
        if task_language is not None and prompt != task_language:
            return "vla_act prompt differs from task_language"
    return None


ATOMIC_TASKS = (
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
)
COMPOSITE_SEEN_TASKS = (
    "ScrubCuttingBoard",
    "StackBowlsCabinet",
    "WashLettuce",
    "RinseSinkBasin",
    "PreSoakPan",
    "StirVegetables",
    "LoadDishwasher",
    "SteamInMicrowave",
    "SetUpCuttingStation",
    "GetToastedBread",
    "DeliverStraw",
    "KettleBoiling",
    "PrepareCoffee",
    "StoreLeftoversInBowl",
    "SearingMeat",
    "PackIdenticalLunches",
)
COMPOSITE_UNSEEN_TASKS = (
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
)
EMPTY_MEMORY_TASKS = frozenset(
    {
        "HeatKebabSandwich",
        "PanTransfer",
        "PortionHotDogs",
        "SeparateFreezerRack",
        "WaffleReheat",
        "WashFruitColander",
        "WeighIngredients",
    }
)


@dataclass(frozen=True)
class SplitSpec:
    tasks: tuple[str, ...]
    seeds: tuple[int, ...]
    timeout_seconds: int


@dataclass(frozen=True, order=True)
class Cell:
    split: Split
    task: str
    seed: int

    @property
    def tag(self) -> str:
        return f"{self.task}_s{self.seed}"


SPLITS = {
    Split.ATOMIC: SplitSpec(ATOMIC_TASKS, tuple(range(1, 11)), 1800),
    Split.COMPOSITE_SEEN: SplitSpec(COMPOSITE_SEEN_TASKS, tuple(range(1, 6)), 3600),
    Split.COMPOSITE_UNSEEN: SplitSpec(COMPOSITE_UNSEEN_TASKS, tuple(range(1, 6)), 3600),
}
CELLS = tuple(
    Cell(split, task, seed)
    for split, spec in SPLITS.items()
    for seed in spec.seeds
    for task in spec.tasks
)
EXPECTED_ROLLOUTS = 340
PROTOCOL_ID = "robocasa-harness-vla-v1"
PAPER_REFERENCES = {
    "RLDX-1": {"atomic": 60.0, "composite_seen": 21.3, "composite_unseen": 5.0},
    "Harness VLA (Codex)": {
        "atomic": 91.6,
        "composite_seen": 56.3,
        "composite_unseen": 13.8,
    },
    "Harness VLA (CC)": {
        "atomic": 79.4,
        "composite_seen": 47.5,
        "composite_unseen": 15.0,
    },
}
MEMORY_PAIR_TASKS = frozenset(
    set(ATOMIC_TASKS)
    | set(COMPOSITE_SEEN_TASKS)
    | (set(COMPOSITE_UNSEEN_TASKS) - set(EMPTY_MEMORY_TASKS))
)

assert len(ALLOWED_ACTIONS) == 8
assert set(ACTION_PARAMETERS) == set(ALLOWED_ACTIONS)
assert len(CELLS) == EXPECTED_ROLLOUTS
assert len(MEMORY_PAIR_TASKS) == 43
assert len(EMPTY_MEMORY_TASKS) == 7


def cell_for(split: Split | str, task: str, seed: int) -> Cell:
    """Return a cell only when it belongs to the frozen matrix."""
    split = Split(split)
    spec = SPLITS[split]
    if task not in spec.tasks or type(seed) is not int or seed not in spec.seeds:
        raise ValueError(
            f"cell is outside the frozen protocol: {split.value}/{task}/s{seed}"
        )
    return Cell(split, task, seed)
