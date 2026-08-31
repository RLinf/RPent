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

"""Immutable task-scoped facts for the supported BEHAVIOR tasks."""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Final, Literal, Mapping

BehaviorPhase = Literal["explore", "eval"]
InstanceKind = Literal["explore", "eval", "candidate"]


@dataclass(frozen=True)
class TerminalFailurePolicy:
    """One task-specific visual terminal-failure contract."""

    condition: str
    runner_reason: str
    causes: tuple[str, ...]
    cameras: tuple[str, ...]


@dataclass(frozen=True)
class SurfaceReviewPolicy:
    """One task-specific target/opposite-surface review contract."""

    target_assessment: str
    opposite_assessment: str
    indeterminate_assessment: str
    opposite_cycles_before_pi0_disable: int


@dataclass(frozen=True)
class ReleaseVisualPolicy:
    """One task-specific visual authorization contract for object release."""

    camera: str
    assessment: str


@dataclass(frozen=True)
class BehaviorInstanceClassification:
    """Task-scoped classification of one native activity instance."""

    task_name: str
    instance_id: int
    kind: InstanceKind
    public_seed: int | None


@dataclass(frozen=True)
class BehaviorTaskSpec:
    """Immutable identity, mapping, and task policy for one BEHAVIOR task."""

    task_index: int
    task_name: str
    task_language: str
    prompt_profile_id: str
    activity_definition_id: int
    scene_model: str
    public_seed_to_instance: Mapping[int, int]
    mapping_version: str
    candidate_mapping_version: str
    explore_public_seeds: tuple[int, ...]
    eval_public_seeds: tuple[int, ...]
    terminal_failure_policy: TerminalFailurePolicy | None = None
    surface_review_policy: SurfaceReviewPolicy | None = None
    release_visual_policy: ReleaseVisualPolicy | None = None

    def __post_init__(self) -> None:
        if (
            isinstance(self.task_index, bool)
            or not isinstance(self.task_index, int)
            or self.task_index < 0
        ):
            raise ValueError("task_index must be a non-negative integer")
        if not self.task_name or not self.task_language:
            raise ValueError("task_name and task_language must be non-empty")
        if self.prompt_profile_id != self.task_name:
            raise ValueError("prompt_profile_id must exactly match task_name")
        if (
            isinstance(self.activity_definition_id, bool)
            or not isinstance(self.activity_definition_id, int)
            or self.activity_definition_id < 0
        ):
            raise ValueError("activity_definition_id must be non-negative")
        if not self.scene_model:
            raise ValueError("scene_model must be non-empty")

        mapping = dict(self.public_seed_to_instance)
        if not mapping or any(
            isinstance(seed, bool)
            or not isinstance(seed, int)
            or seed < 0
            or isinstance(instance_id, bool)
            or not isinstance(instance_id, int)
            or instance_id <= 0
            for seed, instance_id in mapping.items()
        ):
            raise ValueError("public seed mapping must contain positive instance IDs")
        if tuple(sorted(mapping)) != tuple(range(len(mapping))):
            raise ValueError("public seed mapping must be contiguous from seed 0")
        if len(set(mapping.values())) != len(mapping):
            raise ValueError("public seed mapping must not reuse an instance")
        object.__setattr__(self, "public_seed_to_instance", MappingProxyType(mapping))

        explore = tuple(self.explore_public_seeds)
        evaluate = tuple(self.eval_public_seeds)
        if (
            not explore
            or len(set(explore)) != len(explore)
            or len(set(evaluate)) != len(evaluate)
            or set(explore).intersection(evaluate)
            or set(explore).union(evaluate) != set(mapping)
        ):
            raise ValueError(
                "Explore and Eval public seeds must partition the public mapping"
            )
        object.__setattr__(self, "explore_public_seeds", explore)
        object.__setattr__(self, "eval_public_seeds", evaluate)

        if not self.mapping_version or not self.candidate_mapping_version:
            raise ValueError("mapping versions must be non-empty")
        if self.release_visual_policy is not None:
            if self.release_visual_policy.camera != "head":
                raise ValueError("release visual policy camera must be head")
            if not self.release_visual_policy.assessment:
                raise ValueError("release visual assessment must be non-empty")

    @property
    def state_dir_name(self) -> str:
        return f"{self.scene_model}_task_{self.task_name}_instances"

    def tag(self, public_seed: int) -> str:
        self.instance_for_public_seed(public_seed)
        return f"{self.task_name}_s{public_seed}"

    def instance_for_public_seed(
        self,
        public_seed: int,
        *,
        phase: BehaviorPhase | None = None,
    ) -> int:
        if isinstance(public_seed, bool) or not isinstance(public_seed, int):
            raise ValueError("public_seed must be an integer")
        try:
            instance_id = self.public_seed_to_instance[public_seed]
        except KeyError as error:
            raise ValueError(
                f"{self.task_name} has no public seed s{public_seed}"
            ) from error
        if phase is not None:
            if phase == "explore":
                allowed = self.explore_public_seeds
            elif phase == "eval":
                allowed = self.eval_public_seeds
            else:
                raise ValueError(f"unsupported BEHAVIOR phase: {phase!r}")
            if public_seed not in allowed:
                raise ValueError(
                    f"{self.task_name} does not allow s{public_seed} in {phase}"
                )
        return instance_id

    def public_seed_for_instance(self, instance_id: int) -> int | None:
        if (
            isinstance(instance_id, bool)
            or not isinstance(instance_id, int)
            or instance_id <= 0
        ):
            raise ValueError("instance_id must be a positive integer")
        return next(
            (
                seed
                for seed, mapped_instance in self.public_seed_to_instance.items()
                if mapped_instance == instance_id
            ),
            None,
        )

    def classify_instance(self, instance_id: int) -> BehaviorInstanceClassification:
        public_seed = self.public_seed_for_instance(instance_id)
        if public_seed in self.explore_public_seeds:
            kind: InstanceKind = "explore"
        elif public_seed in self.eval_public_seeds:
            kind = "eval"
        else:
            kind = "candidate"
        return BehaviorInstanceClassification(
            task_name=self.task_name,
            instance_id=instance_id,
            kind=kind,
            public_seed=public_seed,
        )


_RADIO_TERMINAL_FAILURE_POLICY: Final = TerminalFailurePolicy(
    condition="radio_tipped_flat",
    runner_reason="visual_radio_tipped_flat",
    causes=("knocked_over_by_robot_hand", "dropped_out_of_gripper"),
    cameras=("head", "left_wrist", "right_wrist"),
)

_RADIO_SURFACE_REVIEW_POLICY: Final = SurfaceReviewPolicy(
    target_assessment="target_bearing_surface_confirmed",
    opposite_assessment="opposite_surface_confirmed",
    indeterminate_assessment="side_or_indeterminate",
    opposite_cycles_before_pi0_disable=2,
)

_TRASH_RELEASE_VISUAL_POLICY: Final = ReleaseVisualPolicy(
    camera="head",
    assessment="attached_object_fully_inside_receptacle_opening",
)

TURNING_ON_RADIO_TASK_SPEC: Final = BehaviorTaskSpec(
    task_index=0,
    task_name="turning_on_radio",
    task_language="Turn on the radio receiver that's on the table in the living room.",
    prompt_profile_id="turning_on_radio",
    activity_definition_id=0,
    scene_model="house_double_floor_lower",
    public_seed_to_instance={
        0: 242,
        1: 109,
        2: 181,
        3: 187,
        4: 197,
        5: 203,
        6: 211,
        7: 212,
        8: 295,
        9: 298,
    },
    mapping_version="turning_on_radio_public_seed_v1",
    candidate_mapping_version="turning_on_radio_candidate_instance_v1",
    explore_public_seeds=(0,),
    eval_public_seeds=tuple(range(1, 10)),
    terminal_failure_policy=_RADIO_TERMINAL_FAILURE_POLICY,
    surface_review_policy=_RADIO_SURFACE_REVIEW_POLICY,
)

PICKING_UP_TRASH_TASK_SPEC: Final = BehaviorTaskSpec(
    task_index=1,
    task_name="picking_up_trash",
    task_language=(
        "Put the three soda cans from the living room inside the trash can "
        "in the kitchen."
    ),
    prompt_profile_id="picking_up_trash",
    activity_definition_id=0,
    scene_model="house_double_floor_lower",
    public_seed_to_instance={
        0: 196,
        1: 67,
        2: 155,
        3: 106,
        4: 161,
        5: 245,
        6: 171,
        7: 156,
        8: 162,
        9: 246,
        10: 108,
        11: 152,
        12: 84,
        13: 198,
        14: 199,
        15: 100,
        16: 111,
        17: 151,
        18: 130,
        19: 168,
    },
    mapping_version="picking_up_trash_public_seed_v1",
    candidate_mapping_version="picking_up_trash_candidate_instance_v1",
    explore_public_seeds=tuple(range(10)),
    eval_public_seeds=tuple(range(10, 20)),
    release_visual_policy=_TRASH_RELEASE_VISUAL_POLICY,
)

_TASK_SPECS_BY_NAME: Final[Mapping[str, BehaviorTaskSpec]] = MappingProxyType(
    {
        spec.task_name: spec
        for spec in (TURNING_ON_RADIO_TASK_SPEC, PICKING_UP_TRASH_TASK_SPEC)
    }
)
_TASK_SPECS_BY_INDEX: Final[Mapping[int, BehaviorTaskSpec]] = MappingProxyType(
    {spec.task_index: spec for spec in _TASK_SPECS_BY_NAME.values()}
)


def get_task_spec(task_name: str) -> BehaviorTaskSpec:
    try:
        return _TASK_SPECS_BY_NAME[task_name]
    except (KeyError, TypeError) as error:
        raise ValueError(f"unsupported BEHAVIOR task name: {task_name!r}") from error


def get_task_spec_by_index(task_index: int) -> BehaviorTaskSpec:
    if isinstance(task_index, bool) or not isinstance(task_index, int):
        raise ValueError("task_index must be an integer")
    try:
        return _TASK_SPECS_BY_INDEX[task_index]
    except KeyError as error:
        raise ValueError(f"unsupported BEHAVIOR task index: {task_index!r}") from error


def resolve_task_spec(*, task_name: str, task_index: int) -> BehaviorTaskSpec:
    by_name = get_task_spec(task_name)
    by_index = get_task_spec_by_index(task_index)
    if by_name is not by_index:
        raise ValueError(
            f"BEHAVIOR task identity mismatch: {task_name!r} != index {task_index}"
        )
    return by_name


def instance_for_public_seed(
    task_name: str,
    public_seed: int,
    *,
    phase: BehaviorPhase | None = None,
) -> int:
    return get_task_spec(task_name).instance_for_public_seed(public_seed, phase=phase)


def classify_instance(
    task_name: str,
    instance_id: int,
) -> BehaviorInstanceClassification:
    return get_task_spec(task_name).classify_instance(instance_id)


__all__ = [
    "BehaviorInstanceClassification",
    "BehaviorPhase",
    "BehaviorTaskSpec",
    "InstanceKind",
    "PICKING_UP_TRASH_TASK_SPEC",
    "ReleaseVisualPolicy",
    "SurfaceReviewPolicy",
    "TURNING_ON_RADIO_TASK_SPEC",
    "TerminalFailurePolicy",
    "classify_instance",
    "get_task_spec",
    "get_task_spec_by_index",
    "instance_for_public_seed",
    "resolve_task_spec",
]
