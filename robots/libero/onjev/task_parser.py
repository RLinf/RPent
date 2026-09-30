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
"""Parse supported manipulation tasks from the public instruction."""

from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass(frozen=True)
class PickPlaceTask:
    """Object phrases sharing the destination relation in the public task."""

    instruction: str
    object_phrase: str
    destination_phrase: str
    relation: str
    receptacle: str
    additional_objects: tuple[str, ...] = ()

    @property
    def object_phrases(self) -> tuple[str, ...]:
        """Return the individual objects to grasp and place separately."""
        return (self.object_phrase, *self.additional_objects)

    @property
    def pick_prompt(self) -> str:
        """Return a grasp-only instruction for the existing Pi0.5 primitive."""
        return f"pick up {self.object_phrase}"


@dataclass(frozen=True)
class DrawerTask:
    """An open or close request for a drawer named in the instruction."""

    instruction: str
    verb: str
    target_phrase: str


def parse_task(instruction: str) -> PickPlaceTask | DrawerTask:
    """Parse drawer contact or placement into a supported shared destination.

    Spatial modifiers in the object phrase are preserved. Unsupported verbs,
    separate task stages and ambiguous destination types fail before motion.
    """
    text = " ".join(instruction.strip().split())
    drawer = re.fullmatch(
        r"(open|close)\s+(.+?)[.!]?", text, flags=re.IGNORECASE
    )
    if drawer is not None:
        verb, target = drawer.groups()
        if re.search(r"\bdrawer\b", target, flags=re.IGNORECASE) and not re.search(
            r"\b(?:and|then)\b", target, flags=re.IGNORECASE
        ):
            return DrawerTask(text, verb.lower(), target)
    match = re.fullmatch(
        r"(?:pick(?: up)?|grab|lift)\s+(.+?)\s+and\s+(?:place|put|set)\s+(?:it|them)\s+(in|into|on|onto)\s+(.+?)[.!]?",
        text,
        flags=re.IGNORECASE,
    )
    if match is None:
        match = re.fullmatch(
            r"(?:put|place|set)\s+(.+?)\s+(in|into|on|onto)\s+(.+?)[.!]?",
            text,
            flags=re.IGNORECASE,
        )
        if match is None:
            raise ValueError(
                "OneJev supports pick-and-place instructions with a shared basket, bowl, plate or tray destination"
            )
        object_phrase, relation, destination = match.groups()
    else:
        object_phrase, relation, destination = match.groups()
    if re.search(r"\bthen\b", object_phrase, flags=re.IGNORECASE) or re.search(
        r"\b(?:and|then)\b", destination, flags=re.IGNORECASE
    ):
        raise ValueError("OneJev requires object placements sharing one destination")
    both = re.match(r"both\s+(.+)", object_phrase, flags=re.IGNORECASE)
    if both is not None:
        objects = re.split(r"\s+and\s+", both.group(1), flags=re.IGNORECASE)
        if len(objects) != 2:
            raise ValueError("OneJev requires two explicit objects after 'both'")
    elif re.search(r"\bbetween\b", object_phrase, flags=re.IGNORECASE):
        # The conjunction joins two landmarks that identify one object.
        objects = [object_phrase]
    else:
        objects = re.split(r"\s+and\s+", object_phrase, flags=re.IGNORECASE)
    objects = [phrase.strip() for phrase in objects]
    if any(
        not phrase
        or (
            re.search(r"\band\b", phrase, flags=re.IGNORECASE)
            and not re.search(r"\bbetween\b", phrase, flags=re.IGNORECASE)
        )
        or re.match(
            r"(?:put|place|pick|grab|lift|set|open|close|turn|push|pull)\b",
            phrase,
            flags=re.IGNORECASE,
        )
        for phrase in objects
    ) or len({phrase.casefold() for phrase in objects}) != len(objects):
        raise ValueError("OneJev requires distinct explicit object phrases")
    found = re.findall(
        r"\b(basket|bowl|plate|tray)\b", destination, flags=re.IGNORECASE
    )
    if len(found) != 1:
        raise ValueError("OneJev requires one explicit supported destination type")
    return PickPlaceTask(
        instruction=text,
        object_phrase=objects[0],
        destination_phrase=destination,
        relation=relation.lower(),
        receptacle=found[0].lower(),
        additional_objects=tuple(objects[1:]),
    )
