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
"""Offline regressions for LIBERO task language and candidate selection."""

from __future__ import annotations

import numpy as np
import pytest

from robots.libero.onejev.candidates import (
    Progress,
    generate_candidates,
    generate_contact_candidates,
)
from robots.libero.onejev.config import OneJevConfig
from robots.libero.onejev.task_parser import DrawerTask, PickPlaceTask, parse_task


@pytest.mark.parametrize(
    "instruction,verb",
    [
        ("open the cabinet’s middle drawer", "open"),
        ("open the bottom drawer of the cabinet", "open"),
        ("Open the middle layer of the drawer", "open"),
        ("close the top drawer of the cabinet", "close"),
    ],
)
def test_drawer_instructions_offer_bounded_contact(instruction: str, verb: str) -> None:
    task = parse_task(instruction)
    assert isinstance(task, DrawerTask)
    assert task.verb == verb
    progress = Progress()
    config = OneJevConfig()

    actions = generate_contact_candidates(task, progress, config)
    assert [action.tool_name for action in actions] == [
        "pi0_doubled",
        "pi0_doubled",
        "finish",
    ]
    assert actions[0].arguments["prompt"] == instruction

    progress.contact_attempts = config.max_contact_attempts
    assert not generate_contact_candidates(task, progress, config)


def test_two_objects_have_independent_grasp_candidates() -> None:
    task = parse_task("place alphabet soup and tomato sauce into basket")
    assert isinstance(task, PickPlaceTask)
    assert task.object_phrases == ("alphabet soup", "tomato sauce")
    progress = Progress(
        pick_attempts_by_object=dict.fromkeys(task.object_phrases, 0),
        placement_attempts_by_object=dict.fromkeys(task.object_phrases, 0),
    )
    options = generate_candidates(
        task,
        eef=np.array([0.0, 0.0, 0.7]),
        gripper_opening=0.08,
        regions=[],
        progress=progress,
        config=OneJevConfig(),
    )
    assert set(options.pick_objects.values()) == set(task.object_phrases)
    assert {
        action.arguments["prompt"]
        for action in options.actions
        if action.tool_name == "pi0_pick"
        and action.arguments["prompt"].startswith("pick up")
    } == {"pick up alphabet soup", "pick up tomato sauce"}

    progress.placement_attempts_by_object["alphabet soup"] = 1
    options = generate_candidates(
        task,
        eef=np.array([0.0, 0.0, 0.7]),
        gripper_opening=0.08,
        regions=[],
        progress=progress,
        config=OneJevConfig(),
    )
    assert set(options.pick_objects.values()) == {"tomato sauce"}


def test_spatial_relation_names_one_object() -> None:
    task = parse_task(
        "lift the black bowl between the plate and ramekin and set it on the plate"
    )
    assert isinstance(task, PickPlaceTask)
    assert task.object_phrases == ("the black bowl between the plate and ramekin",)
    assert task.receptacle == "plate"
