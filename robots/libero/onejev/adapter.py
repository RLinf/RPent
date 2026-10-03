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
"""Read public LIBERO artifacts and maintain a bounded OneJev action episode."""

from __future__ import annotations

import base64
from dataclasses import asdict
from typing import TYPE_CHECKING

import numpy as np

from robots.libero.onejev.candidates import (
    Progress,
    generate_candidates,
    generate_contact_candidates,
)
from robots.libero.onejev.config import OneJevConfig
from robots.libero.onejev.geometry import mark_regions, propose_regions
from robots.libero.onejev.prompts import CONTACT_INSTRUCTIONS, INSTRUCTIONS
from robots.libero.onejev.task_parser import DrawerTask, PickPlaceTask, parse_task
from rpent.planner.onejev_types import ActionCandidate, DecisionContext
from rpent.tools.toolkit import ToolResult

if TYPE_CHECKING:
    from robots.libero.toolkit import LiberoToolkit


class LiberoOneJevAdapter:
    """Generate actions without simulator poses, contact flags or segmentation."""

    def __init__(self, toolkit: LiberoToolkit) -> None:
        """Bind the toolkit's public-only observation contract and control settings."""
        if not toolkit.primitives.env.public_observations_only:
            raise ValueError("OneJev requires a public-observations-only LIBERO server")
        self._toolkit = toolkit
        self._config: OneJevConfig = toolkit.onejev_config
        record = toolkit.state.latest_record()
        if record is None:
            raise ValueError("LIBERO has no initial public observation")
        instruction = record.extras.get("task_language")
        if not isinstance(instruction, str) or not instruction.strip():
            raise ValueError("LIBERO did not provide a public task instruction")
        self._task = parse_task(instruction)
        objects = (
            self._task.object_phrases if isinstance(self._task, PickPlaceTask) else ()
        )
        self._progress = Progress(
            pick_attempts_by_object=dict.fromkeys(objects, 0),
            placement_attempts_by_object=dict.fromkeys(objects, 0),
        )
        self._destinations = {}
        self._pick_objects: dict[str, str] = {}
        self._recent: list[dict] = []
        self._decision_step = record.step_idx

    def prepare(self) -> DecisionContext:
        """Build current RGB, measured geometry and feasible action alternatives."""
        store = self._toolkit.state
        record = store.latest_record()
        if record is None:
            raise RuntimeError("LIBERO lost its public observation record")
        summary = {
            "task": self._task.instruction,
            "task_type": "drawer"
            if isinstance(self._task, DrawerTask)
            else "placement",
            "step_idx": record.step_idx,
        }
        if isinstance(self._task, PickPlaceTask):
            summary["requested_objects"] = list(self._task.object_phrases)
        if record.terminated or self._toolkit.solved():
            return DecisionContext(
                summary,
                outcome={
                    "status": "success",
                    "summary": "LIBERO reported native task completion",
                },
            )
        if record.truncated:
            return DecisionContext(
                summary,
                outcome={
                    "status": "failure",
                    "summary": "LIBERO episode step budget exhausted",
                },
            )
        if self._progress.no_progress >= self._config.max_no_progress:
            return DecisionContext(
                summary,
                outcome={
                    "status": "stuck",
                    "summary": "Repeated primitive failure or no observed progress",
                },
            )
        placement_counts = self._progress.placement_attempts_by_object
        active_object = self._progress.active_object
        if isinstance(self._task, PickPlaceTask) and (
            (
                self._progress.held
                and active_object is not None
                and placement_counts[active_object]
                >= self._config.max_placement_attempts
            )
            or all(
                count >= self._config.max_placement_attempts
                for count in placement_counts.values()
            )
        ):
            return DecisionContext(
                summary,
                outcome={
                    "status": "failure",
                    "summary": "Per-object placement attempts exhausted without native success",
                },
            )
        robot = {
            name: record.state[name]
            for name in (
                "robot0_eef_pos",
                "robot0_eef_quat",
                "robot0_gripper_qpos",
            )
        }
        eef = np.asarray(robot["robot0_eef_pos"], dtype=float)
        gripper = np.asarray(robot["robot0_gripper_qpos"], dtype=float)
        if (
            eef.shape != (3,)
            or gripper.shape != (2,)
            or not np.isfinite(eef).all()
            or not np.isfinite(gripper).all()
        ):
            raise ValueError("Invalid public EEF or gripper observation")
        opening = float(np.abs(gripper).sum())
        if isinstance(self._task, DrawerTask):
            actions = generate_contact_candidates(
                self._task, self._progress, self._config
            )
            if not actions:
                return DecisionContext(
                    summary,
                    outcome={
                        "status": "failure",
                        "summary": "Drawer-contact attempts exhausted without native success",
                    },
                )
            media = []
            for name in ("agentview.png", "wrist.png"):
                data = base64.b64encode(
                    store.load_bytes(name, step=record.step_idx)
                ).decode("ascii")
                media.append({"type": "image", "data": "data:image/png;base64," + data})
            summary.update(
                {
                    "agentview": "<image:1>",
                    "wrist": "<image:2>",
                    "robot": robot,
                    "gripper_opening_m": opening,
                    "progress_hypotheses": asdict(self._progress),
                    "recent_actions": self._recent[-4:],
                }
            )
            self._decision_step = record.step_idx
            return DecisionContext(
                summary,
                candidates=actions,
                media=media,
                instructions=CONTACT_INSTRUCTIONS,
            )
        rgb = store.load("agentview.png", step=record.step_idx)
        world = store.load("agentview_world.npz", step=record.step_idx)
        regions, plane_z = propose_regions(
            rgb,
            world,
            receptacle=self._task.receptacle,
            eef=eef,
            config=self._config,
        )
        geometry = {
            "step_idx": record.step_idx,
            "camera": "agentview",
            "resolution": list(np.asarray(rgb).shape[:2]),
            "source": "public_rgb+depth+camera_calibration",
            "support_plane_z_hypothesis": plane_z,
            "regions": [region.as_dict() for region in regions],
            "control_priors": asdict(self._config),
        }
        if store.save("onejev_geometry.json", geometry, step=record.step_idx) is None:
            raise RuntimeError("Could not record OneJev geometry provenance")
        if (
            store.save(
                "onejev_regions.png", mark_regions(rgb, regions), step=record.step_idx
            )
            is None
        ):
            raise RuntimeError("Could not record OneJev RGB region overlay")
        if not regions:
            return DecisionContext(
                summary,
                outcome={
                    "status": "stuck",
                    "summary": "No reliable destination region from public RGB-D; inspect onejev_regions.png or configure target_roi",
                },
            )
        # Match a fresh measured region to the previously selected destination.
        # Old coordinates never supply a new movement target by themselves.
        if self._progress.destination_xyz is not None:
            previous = np.asarray(self._progress.destination_xyz)
            regions = sorted(
                regions,
                key=lambda item: np.linalg.norm(
                    np.asarray(item.xyz[:2]) - previous[:2]
                ),
            )
        candidates = generate_candidates(
            self._task,
            eef=eef,
            gripper_opening=opening,
            regions=regions,
            progress=self._progress,
            config=self._config,
        )
        actions = candidates.actions
        self._destinations = candidates.destinations
        self._pick_objects = candidates.pick_objects
        if not actions:
            reason = "No reliable RGB-D destination proposal or admissible action"
            if not self._progress.held and all(
                self._progress.pick_attempts_by_object[phrase]
                >= self._config.max_pick_attempts
                or self._progress.placement_attempts_by_object[phrase]
                >= self._config.max_placement_attempts
                for phrase in self._task.object_phrases
            ):
                reason = "Per-object grasp or placement attempts exhausted without native success"
            return DecisionContext(
                summary, outcome={"status": "stuck", "summary": reason}
            )
        media = []
        for name in ("onejev_regions.png", "wrist.png"):
            data = base64.b64encode(
                store.load_bytes(name, step=record.step_idx)
            ).decode("ascii")
            media.append({"type": "image", "data": "data:image/png;base64," + data})
        summary.update(
            {
                "agentview_marked": "<image:1>",
                "wrist": "<image:2>",
                "robot": robot,
                "gripper_opening_m": opening,
                "geometry": geometry,
                "progress_hypotheses": asdict(self._progress),
                "recent_actions": self._recent[-4:],
            }
        )
        self._decision_step = record.step_idx
        return DecisionContext(
            summary, candidates=actions, media=media, instructions=INSTRUCTIONS
        )

    def observe(self, action: ActionCandidate, result: ToolResult) -> None:
        """Update hypotheses from fresh robot observations and primitive results."""
        record = self._toolkit.state.latest_record()
        if record is None:
            raise RuntimeError("LIBERO did not record action execution")
        if result.is_finish:
            return
        if record.step_idx <= self._decision_step:
            raise RuntimeError("No fresh public observation after the primitive")
        execution = record.result or {}
        if result.result.get("state_capture_error"):
            raise RuntimeError(result.result["state_capture_error"])
        gripper = np.asarray(record.state["robot0_gripper_qpos"], dtype=float)
        opening = float(np.abs(gripper).sum())
        before = self._toolkit.state.get(self._decision_step)
        eef_before = np.asarray(before.state["robot0_eef_pos"], dtype=float)
        eef_after = np.asarray(record.state["robot0_eef_pos"], dtype=float)
        self._recent.append(
            {
                "step_idx": record.step_idx,
                "tool": action.tool_name,
                "arguments": action.arguments,
                "result": execution,
                "eef_displacement_m": float(np.linalg.norm(eef_after - eef_before)),
            }
        )
        if action.tool_name == "pi0_pick":
            pick_object = self._pick_objects[action.id]
            self._progress.pick_attempts_by_object[pick_object] += 1
        elif action.tool_name == "release":
            release_object = self._progress.active_object
            if release_object is None:
                raise RuntimeError("Release has no active object hypothesis")
            self._progress.placement_attempts_by_object[release_object] += 1
        elif action.tool_name == "pi0_doubled":
            self._progress.contact_attempts += 1
        if execution.get("error") or execution.get("interrupted"):
            self._progress.no_progress += 1
            return
        if action.tool_name == "pi0_doubled":
            self._progress.no_progress = (
                0
                if execution.get("contact_skill_executed")
                else self._progress.no_progress + 1
            )
        elif action.tool_name == "pi0_pick":
            self._progress.held = bool(
                execution.get("success")
                and execution.get("peak_lift_m", 0) >= 0.05
                and self._config.min_gripper_opening
                <= opening
                <= self._config.max_gripper_opening
            )
            self._progress.active_object = pick_object if self._progress.held else None
            self._progress.destination_xyz = None
            self._progress.no_progress = (
                0 if self._progress.held else self._progress.no_progress + 1
            )
        elif action.tool_name == "release":
            if opening > self._config.max_gripper_opening:
                # An open gripper ends this holding hypothesis, not the task.
                self._progress.held = False
                self._progress.active_object = None
                self._progress.destination_xyz = None
                self._progress.pick_attempts_by_object[release_object] = 0
                self._progress.no_progress = 0
            else:
                self._progress.no_progress += 1
        elif action.tool_name == "move_to":
            reached = (
                execution.get("final_dist_m", float("inf"))
                <= self._config.move_tolerance * 2
            )
            moved = np.linalg.norm(eef_after - eef_before) > 0.005
            self._progress.no_progress = (
                0 if reached or moved else self._progress.no_progress + 1
            )
            destination = self._destinations.get(action.id)
            if destination is not None:
                self._progress.destination_xyz = destination.xyz
            if (
                not self._config.min_gripper_opening
                <= opening
                <= self._config.max_gripper_opening
            ):
                self._progress.held = False
                self._progress.active_object = None
                self._progress.destination_xyz = None
        elif action.tool_name == "set_gripper":
            changed = (
                abs(opening - float(np.abs(before.state["robot0_gripper_qpos"]).sum()))
                > 0.005
            )
            self._progress.no_progress = (
                0 if changed else self._progress.no_progress + 1
            )
