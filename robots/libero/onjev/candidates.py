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
"""Compute admissible primitive arguments from public robot and RGB-D state."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from robots.libero.onjev.config import OneJevConfig
from robots.libero.onjev.geometry import Region, in_workspace
from robots.libero.onjev.task_parser import DrawerTask, PickPlaceTask
from rpent.planner.onejev_types import ActionCandidate


@dataclass
class Progress:
    """Episode-local hypotheses derived from primitive and observation evidence."""

    held: bool = False
    pick_attempts: int = 0
    placement_attempts: int = 0
    no_progress: int = 0
    destination_xyz: tuple[float, float, float] | None = None
    active_object: str | None = None
    pick_attempts_by_object: dict[str, int] = field(default_factory=dict)
    placement_attempts_by_object: dict[str, int] = field(default_factory=dict)
    contact_attempts: int = 0


@dataclass(frozen=True)
class CandidateSet:
    """Fully parameterized actions and their robot-owned object/region context."""

    actions: tuple[ActionCandidate, ...]
    destinations: dict[str, Region]
    pick_objects: dict[str, str]


def generate_contact_candidates(
    task: DrawerTask, progress: Progress, config: OneJevConfig
) -> tuple[ActionCandidate, ...]:
    """Offer bounded Pi0.5 drawer-contact actions from the public instruction."""
    if progress.contact_attempts >= config.max_contact_attempts:
        return ()
    alternate = (
        f"pull open {task.target_phrase}"
        if task.verb == "open"
        else f"push {task.target_phrase} closed"
    )
    actions = [
        ActionCandidate(
            f"c{index}",
            "pi0_doubled",
            {"prompt": prompt, "max_chunks": config.contact_max_chunks},
            f"Use Pi0.5 contact control to {task.verb} the named drawer; inspect the new RGB views before another attempt.",
        )
        for index, prompt in enumerate((task.instruction, alternate))
    ]
    actions.append(
        ActionCandidate(
            f"c{len(actions)}",
            "finish",
            {
                "status": "stuck",
                "summary": "The public views do not support another drawer-contact attempt",
            },
            "Stop if the named drawer cannot be identified or contact is unsafe.",
        )
    )
    return tuple(actions)


def generate_candidates(
    task: PickPlaceTask,
    *,
    eef: np.ndarray,
    gripper_opening: float,
    regions: list[Region],
    progress: Progress,
    config: OneJevConfig,
) -> CandidateSet:
    """Generate complete grasp, carry, lower and release actions for this step."""
    actions: list[ActionCandidate] = []
    destinations: dict[str, Region] = {}
    pick_objects: dict[str, str] = {}

    def add(
        name: str,
        arguments: dict,
        description: str,
        region: Region | None = None,
        object_phrase: str | None = None,
    ) -> None:
        candidate_id = f"c{len(actions)}"
        actions.append(ActionCandidate(candidate_id, name, arguments, description))
        if region is not None:
            destinations[candidate_id] = region
        if object_phrase is not None:
            pick_objects[candidate_id] = object_phrase

    def move(
        target: np.ndarray,
        description: str,
        region: Region | None = None,
        *,
        gripper: float = 1.0,
    ) -> None:
        if not in_workspace(target, config):
            return
        difference = target[:2] - eef[:2]
        distance = float(np.linalg.norm(difference))
        if distance > config.max_xy_step:
            target = target.copy()
            target[:2] = eef[:2] + difference * config.max_xy_step / distance
        if np.linalg.norm(target - eef) < config.move_tolerance:
            return
        add(
            "move_to",
            {
                "xyz": [float(value) for value in target],
                "gripper": gripper,
                "max_steps": config.move_max_steps,
                "step_clip": 0.025,
                "tol": config.move_tolerance,
                "action_scale": 0.05,
                "target_yaw": None,
                "yaw_step_clip": 0.10,
            },
            description,
            region,
        )

    if not progress.held:
        eligible = [
            phrase
            for phrase in task.object_phrases
            if progress.pick_attempts_by_object.get(phrase, 0) < config.max_pick_attempts
            and progress.placement_attempts_by_object.get(phrase, 0)
            < config.max_placement_attempts
        ]
        if not eligible:
            return CandidateSet((), {}, {})
        # Give each requested object a placement attempt before offering retries.
        fewest_placements = min(
            progress.placement_attempts_by_object.get(phrase, 0) for phrase in eligible
        )
        retreat_z = (
            max(region.rim_z for region in regions)
            + config.eef_object_offset
            + config.carry_clearance
            if regions
            else eef[2]
        )
        if progress.placement_attempts and eef[2] < retreat_z - config.move_tolerance:
            move(
                np.array([eef[0], eef[1], retreat_z]),
                "Lift the open gripper vertically clear of the observed receptacle before grasping another object.",
                gripper=-1.0,
            )
        elif gripper_opening < config.min_gripper_opening:
            add(
                "set_gripper",
                {"gripper": -1.0, "steps": 5},
                "Open the empty closed gripper before another grasp attempt.",
            )
        else:
            for object_phrase in eligible:
                if (
                    progress.placement_attempts_by_object.get(object_phrase, 0)
                    != fewest_placements
                ):
                    continue
                for prompt in (
                    f"pick up {object_phrase}",
                    f"grasp {object_phrase} and lift it",
                ):
                    add(
                        "pi0_pick",
                        {
                            "prompt": prompt,
                            "max_chunks": config.pick_max_chunks,
                            "lift_thresh": 0.05,
                            "gripper_closed_thresh": 0.06,
                            "gripper_open_thresh": config.min_gripper_opening,
                            "descent_thresh": 0.10,
                        },
                        f"Use Pi0.5 to grasp '{object_phrase}' only if it still needs placement in '{task.destination_phrase}' according to RGB; inspect the new views after this bounded grasp.",
                        object_phrase=object_phrase,
                    )
    else:
        if (
            not config.min_gripper_opening
            <= gripper_opening
            <= config.max_gripper_opening
        ):
            return CandidateSet((), {}, {})
        if not regions:
            return CandidateSet((), {}, {})
        carry_z = max(
            eef[2],
            max(region.rim_z for region in regions)
            + config.eef_object_offset
            + config.carry_clearance,
        )
        selected_region = None
        if progress.destination_xyz is not None:
            previous = np.asarray(progress.destination_xyz)
            nearest = min(
                regions,
                key=lambda item: np.linalg.norm(
                    np.asarray(item.xyz[:2]) - previous[:2]
                ),
            )
            if np.linalg.norm(np.asarray(nearest.xyz[:2]) - previous[:2]) < 0.06:
                selected_region = nearest
        at_destination = (
            selected_region is not None
            and np.linalg.norm(np.asarray(selected_region.xyz[:2]) - eef[:2])
            <= config.move_tolerance
        )
        if eef[2] < carry_z - config.move_tolerance and not at_destination:
            move(
                np.array([eef[0], eef[1], carry_z]),
                f"Lift the held-object hypothesis '{progress.active_object}' vertically above observed obstacles before lateral transport.",
            )
        else:
            for region in regions:
                if (
                    at_destination
                    and eef[2] < carry_z - config.move_tolerance
                    and region is not selected_region
                ):
                    continue
                xy_distance = float(
                    np.linalg.norm(np.asarray(region.xyz[:2]) - eef[:2])
                )
                target = np.asarray(region.xyz, dtype=float)
                placement_z = max(
                    region.xyz[2]
                    + config.eef_object_offset
                    + config.placement_clearance,
                    region.rim_z + config.eef_object_offset * 0.55,
                )
                if xy_distance > config.move_tolerance:
                    target[2] = carry_z
                    move(
                        target,
                        f"Carry the held-object hypothesis '{progress.active_object}' toward marked RGB region {region.id} at pixel {region.pixel} as a hypothesis for '{task.destination_phrase}'. The route is split into bounded waypoints.",
                        region,
                    )
                else:
                    target[2] = placement_z
                    if eef[2] > placement_z + config.move_tolerance:
                        move(
                            target,
                            f"Lower above the measured interior of region {region.id} for '{task.destination_phrase}', keeping the gripper closed. The EEF/object offset is an explicit control prior.",
                            region,
                        )
                    elif abs(eef[2] - placement_z) <= config.move_tolerance * 2:
                        add(
                            "release",
                            {"max_steps": 20},
                            f"Release over marked region {region.id} ONLY if RGB confirms it is '{task.destination_phrase}' and '{progress.active_object}' is held.",
                            region,
                        )
    if actions:
        add(
            "finish",
            {
                "status": "stuck",
                "summary": "OneJev could not identify a reliable next action from public observations",
            },
            "Stop if the views do not support any proposed grasp, destination or holding hypothesis; do not guess.",
        )
    return CandidateSet(tuple(actions), destinations, pick_objects)
