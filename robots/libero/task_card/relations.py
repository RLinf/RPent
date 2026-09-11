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
"""Extract a small, explicit task graph from LIBERO goal instructions.

Task cards record *how* a successful episode acted.  This module records what
the episode must accomplish, independently of the task/swap scene layout.  The
graph is deliberately narrow: unsupported language is rejected instead of
quietly producing a plausible but wrong card.
"""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True)
class GoalRelation:
    """One ordered predicate in a LIBERO task."""

    predicate: str
    subject: str
    object: str | None = None


@dataclass(frozen=True)
class TaskGraph:
    """Semantic part of a task card, in required execution order."""

    language: str
    entities: tuple[str, ...]
    goals: tuple[GoalRelation, ...]

    def as_dict(self) -> dict[str, Any]:
        return {
            "language": self.language,
            "entities": list(self.entities),
            "goals": [asdict(goal) for goal in self.goals],
        }


_ARTICLE = re.compile(r"^(?:the|a|an)\s+", re.IGNORECASE)
_SPACE = re.compile(r"\s+")


def _entity(raw: str) -> str:
    value = _SPACE.sub(" ", raw.strip(" .,_-")).strip()
    return _ARTICLE.sub("", value).strip()


def _graph(language: str, goals: list[GoalRelation]) -> TaskGraph:
    entities: list[str] = []
    for goal in goals:
        for entity in (goal.subject, goal.object):
            if entity and entity not in entities:
                entities.append(entity)
    return TaskGraph(
        language=language.strip(), entities=tuple(entities), goals=tuple(goals)
    )


def extract_goal_relations(language: str) -> TaskGraph:
    """Parse one LIBERO-Goal instruction into ordered symbolic relations.

    Supported predicates cover the ten canonical LIBERO-Goal task types:
    ``open``, ``turn_on``, ``on``, ``in`` and ``in_front_of``.  Common
    pick/lift/set/place paraphrases used by task and language perturbations are
    accepted.  Scene layout never appears in the result.
    """

    text = _SPACE.sub(" ", language.strip().rstrip(". "))
    if not text:
        raise ValueError("task language is empty")

    # The only canonical compound Goal task.  Resolve "inside" against the
    # drawer opened by the first clause, rather than treating it as an entity.
    compound = re.fullmatch(
        r"open\s+(?P<fixture>.+?)\s+and\s+(?:"
        r"(?:pick(?:\s+up)?|lift|grab)\s+(?P<picked>.+?)\s+(?:and\s+)?"
        r"(?:put|place|set)\s+(?:it\s+)?inside(?:\s+it)?|"
        r"(?:put|place|set)\s+(?P<placed>.+?)\s+inside(?:\s+it)?"
        r")",
        text,
        re.IGNORECASE,
    )
    if compound:
        fixture = _entity(compound.group("fixture"))
        item = _entity(compound.group("picked") or compound.group("placed"))
        return _graph(
            language,
            [
                GoalRelation("open", fixture),
                GoalRelation("in", item, fixture),
            ],
        )

    unary_patterns = (
        ("open", r"open\s+(?P<subject>.+)"),
        ("turn_on", r"turn\s+on\s+(?P<subject>.+)"),
        ("turn_on", r"switch\s+on\s+(?P<subject>.+)"),
        ("turn_off", r"turn\s+off\s+(?P<subject>.+)"),
        ("turn_off", r"switch\s+off\s+(?P<subject>.+)"),
    )
    for predicate, pattern in unary_patterns:
        match = re.fullmatch(pattern, text, re.IGNORECASE)
        if match:
            return _graph(
                language,
                [GoalRelation(predicate, _entity(match.group("subject")))],
            )

    relation_patterns = (
        ("in_front_of", r"(?:to\s+)?(?:the\s+)?front\s+of"),
        ("in", r"(?:in|inside|into)"),
        # Longer alternative first: otherwise "on top of" leaves "top of"
        # attached to the destination entity.
        ("on", r"(?:on\s+(?:the\s+)?top\s+of|onto|on)"),
    )
    sentence_patterns = (
        # Canonical Goal instructions: "put the bowl on the stove".
        r"(?:put|place|set)\s+(?P<subject>.+?)\s+{relation}\s+(?P<object>.+)",
        # Language perturbation: "lift the bowl and set it onto the plate".
        r"(?:pick(?:\s+up)?|lift|grab)\s+(?P<subject>.+?)\s+and\s+"
        r"(?:put|place|set)\s+it\s+{relation}\s+(?P<object>.+)",
        # Spatial Goal uses pushing rather than pick-and-place.
        r"push\s+(?P<subject>.+?)\s+{relation}\s+(?P<object>.+)",
    )
    for predicate, relation in relation_patterns:
        for sentence in sentence_patterns:
            match = re.fullmatch(
                sentence.format(relation=relation), text, re.IGNORECASE
            )
            if match:
                return _graph(
                    language,
                    [
                        GoalRelation(
                            predicate,
                            _entity(match.group("subject")),
                            _entity(match.group("object")),
                        )
                    ],
                )

    raise ValueError(f"unsupported LIBERO-Goal instruction: {language!r}")


def extract_long_relations(language: str) -> TaskGraph:
    """Parse one canonical LIBERO-10 instruction into ordered transactions."""
    text = _SPACE.sub(" ", language.strip().rstrip(". "))

    match = re.fullmatch(r"put both (.+?)s on the stove", text, re.I)
    if match:
        # Canonical PRO swap t8 addresses a same-category pair collectively.
        # Left/right are public spatial descriptions, not simulator identities.
        item = _entity(match.group(1))
        return _graph(
            language,
            [
                GoalRelation("on", f"left {item}", "stove"),
                GoalRelation("on", f"right {item}", "stove"),
            ],
        )

    match = re.fullmatch(r"put both (.+?) and (.+?) in the basket", text, re.I)
    if match:
        return _graph(
            language,
            [
                GoalRelation("in", _entity(match.group(1)), "basket"),
                GoalRelation("in", _entity(match.group(2)), "basket"),
            ],
        )

    match = re.fullmatch(r"turn on the stove and put (.+?) on it", text, re.I)
    if match:
        return _graph(
            language,
            [
                GoalRelation("turn_on", "stove"),
                GoalRelation("on", _entity(match.group(1)), "stove"),
            ],
        )

    match = re.fullmatch(
        r"put (.+?) in the bottom drawer of the cabinet and close it", text, re.I
    )
    if match:
        drawer = "bottom drawer of the cabinet"
        return _graph(
            language,
            [
                GoalRelation("in", _entity(match.group(1)), drawer),
                GoalRelation("close", drawer),
            ],
        )

    match = re.fullmatch(r"put (.+?) in the microwave and close it", text, re.I)
    if match:
        return _graph(
            language,
            [
                GoalRelation("in", _entity(match.group(1)), "microwave"),
                GoalRelation("close", "microwave"),
            ],
        )

    match = re.fullmatch(
        r"put (.+?) on the left plate and put (.+?) on the right plate", text, re.I
    )
    if match:
        return _graph(
            language,
            [
                GoalRelation("on", _entity(match.group(1)), "left plate"),
                GoalRelation("on", _entity(match.group(2)), "right plate"),
            ],
        )

    match = re.fullmatch(
        r"pick up (.+?) and place it in the back compartment of the caddy",
        text,
        re.I,
    )
    if match:
        return _graph(
            language,
            [
                GoalRelation(
                    "in", _entity(match.group(1)), "back compartment of the caddy"
                )
            ],
        )

    match = re.fullmatch(
        r"put (.+?) on the plate and put (.+?) to the right of the plate", text, re.I
    )
    if match:
        return _graph(
            language,
            [
                GoalRelation("on", _entity(match.group(1)), "plate"),
                GoalRelation("right_of", _entity(match.group(2)), "plate"),
            ],
        )

    match = re.fullmatch(r"put (.+?) on the stove", text, re.I)
    if match:
        return _graph(
            language,
            [GoalRelation("on", _entity(match.group(1)), "stove")],
        )

    raise ValueError(f"unsupported LIBERO-10 instruction: {language!r}")
