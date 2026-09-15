# Copyright 2026 The RPent Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
"""Generate one replayable LIBERO task card from one successful trace.

The generator deliberately has no candidate search or cross-seed evaluation.  Its
two required inputs are the final episode audit (JSON) and the recorded primitive
recipe (JSONL).  Segment result JSON files from the same episode are optional: when
present they provide measured anchors; otherwise semantic anchors are reconstructed
from the recipe waypoints and the task relation graph.
"""

from __future__ import annotations

import argparse
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from robots.libero.task_card.relations import (
    TaskGraph,
    extract_goal_relations,
    extract_long_relations,
)

CELL_TAG = re.compile(r"^(10|goal|object|spatial)_(task|swap)_t([0-9])_s(\d+)$")
MOVE_ACTIONS = {"move_to", "move_pose"}
ARTICULATION_VERBS = re.compile(r"\s*(?:turn|switch|open|close)\b", re.I)


@dataclass(frozen=True)
class TraceIdentity:
    family: str
    suite: str
    task: int
    seed: int
    tag: str

    @property
    def key(self) -> str:
        return f"{self.suite}_t{self.task}"


def _identity(audit_path: Path) -> TraceIdentity:
    match = CELL_TAG.fullmatch(audit_path.stem)
    if not match:
        raise ValueError(
            f"audit filename {audit_path.name!r} must be "
            "<10|goal|object|spatial>_<task|swap>_t<0-9>_s<seed>.json"
        )
    family, suite, task, seed = match.groups()
    return TraceIdentity(family, suite, int(task), int(seed), audit_path.stem)


def _read_audit(path: Path) -> dict[str, Any]:
    try:
        audit = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot read audit JSON {path}: {exc}") from exc
    if not isinstance(audit, dict):
        raise ValueError("audit JSON must contain one object")
    if audit.get("libero_terminated") is not True:
        raise ValueError(
            "task cards can only be generated from libero_terminated=true traces"
        )
    return audit


def _task_language(audit: dict[str, Any]) -> str:
    # Perturbation audits use perturbed_task_language; ordinary audits use
    # task_language.  Both are public instruction text, never simulator state.
    language = str(
        audit.get("perturbed_task_language") or audit.get("task_language") or ""
    ).strip()
    if not language:
        raise ValueError("audit JSON has no task language")
    return language


def _read_recipe(path: Path) -> list[dict[str, Any]]:
    plan: list[dict[str, Any]] = []
    try:
        lines = path.read_text().splitlines()
    except OSError as exc:
        raise ValueError(f"cannot read recipe JSONL {path}: {exc}") from exc
    for line_number, line in enumerate(lines, 1):
        if not line.strip():
            continue
        try:
            entry = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"invalid recipe JSONL line {line_number}: {exc}") from exc
        if not isinstance(entry, dict) or not isinstance(entry.get("action"), str):
            raise ValueError(
                f"recipe line {line_number} must be an object with an action"
            )
        plan.append(entry)
    if not plan:
        raise ValueError("recipe JSONL contains no actions")
    return plan


def extract_relations(family: str, language: str) -> TaskGraph:
    """Parse semantics for any of the four LIBERO-PRO families."""
    if family == "10":
        return extract_long_relations(language)
    if family in {"goal", "object", "spatial"}:
        return extract_goal_relations(language)
    raise ValueError(f"unsupported LIBERO family: {family!r}")


def _segment_anchors(segments: Path | None) -> list[dict[str, Any]]:
    if segments is None or not segments.is_dir():
        return []
    anchors: list[dict[str, Any]] = []
    for path in sorted(segments.glob("segment_*.json")):
        try:
            data = json.loads(path.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        xyz = data.get("world_xyz")
        phrase = str(data.get("prompt") or "").strip()
        if (
            data.get("found") is not True
            or not phrase
            or not isinstance(xyz, list)
            or len(xyz) < 3
            or not all(isinstance(value, (int, float)) for value in xyz[:3])
        ):
            continue
        point = [float(value) for value in xyz[:3]]
        if any(anchor["phrase"] == phrase for anchor in anchors):
            continue
        if any(math.dist(point[:2], anchor["median_xy"]) < 0.03 for anchor in anchors):
            continue
        anchors.append(
            {
                "phrase": phrase,
                "locator": "segment" if data.get("mode") == "text" else "molmo",
                "camera": data.get("camera", "agentview"),
                "score": data.get("score"),
                "readings": [point],
                "median_xy": point[:2],
                "z_top": point[2],
                "z_span": 0.0,
            }
        )
    return anchors


def _move_xy(entry: dict[str, Any]) -> list[float] | None:
    xyz = entry.get("xyz")
    if (
        isinstance(xyz, list)
        and len(xyz) == 3
        and all(isinstance(value, (int, float)) for value in xyz)
    ):
        return [float(xyz[0]), float(xyz[1])]
    return None


def _is_articulation(entry: dict[str, Any]) -> bool:
    if entry.get("action") == "pi0_doubled":
        return True
    return entry.get("action") == "pi0_pick" and bool(
        ARTICULATION_VERBS.match(str(entry.get("prompt") or ""))
    )


def _semantic_phrase(entity: str, predicate: str, *, subject: bool) -> str:
    if subject:
        return f"the {entity}"
    if predicate == "in":
        return f"the inside of the {entity}"
    if predicate == "on" and "drawer" in entity:
        return f"the top surface of the {entity}"
    if predicate == "on" and "stove" in entity:
        return "the stove burner"
    if predicate in {"in_front_of", "right_of"}:
        direction = "front" if predicate == "in_front_of" else "right side"
        return f"the {direction} of the {entity}"
    return f"the {entity}"


def _fallback_anchors(
    graph: TaskGraph, plan: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    """Recover semantic reference points when no segment reading was saved.

    Pick/release pairs are consumed in order, matching Long's ordered goal
    transactions.  A ``pi0_pick`` whose prompt starts with an articulation verb is
    deliberately not consumed as an object grasp.
    """
    moves = [(index, _move_xy(entry)) for index, entry in enumerate(plan)]
    moves = [(index, xy) for index, xy in moves if xy is not None]
    if not moves:
        return []
    policy_picks = [
        index
        for index, entry in enumerate(plan)
        if entry.get("action") == "pi0_pick" and not _is_articulation(entry)
    ]
    releases = [
        index for index, entry in enumerate(plan) if entry.get("action") == "release"
    ]
    # Some successful low-level recipes grasp with move + set_gripper rather than
    # pi0_pick.  Treat the first close after each release boundary as that
    # transaction's acquisition point; later close commands merely firm the hold.
    manual_picks: list[int] = []
    previous_release = -1
    for release in releases:
        closing = next(
            (
                index
                for index, entry in enumerate(plan)
                if previous_release < index < release
                and entry.get("action") == "set_gripper"
                and entry.get("gripper") == 1
            ),
            None,
        )
        if closing is not None:
            manual_picks.append(closing)
        previous_release = release
    picks = sorted(set(policy_picks + manual_picks))
    interactions = [
        index for index, entry in enumerate(plan) if _is_articulation(entry)
    ]

    def last_before(boundary: int, *, after: int = -1) -> list[float] | None:
        eligible = [xy for index, xy in moves if after < index < boundary]
        return eligible[-1] if eligible else None

    grasp_cursor = 0
    interaction_cursor = 0
    anchors: list[dict[str, Any]] = []
    for goal in graph.goals:
        pairs: list[tuple[str, list[float] | None]]
        if goal.predicate in {"open", "close", "turn_on", "turn_off"}:
            boundary = (
                interactions[interaction_cursor]
                if interaction_cursor < len(interactions)
                else len(plan)
            )
            interaction_cursor += 1
            suffix = (
                " handle"
                if any(x in goal.subject for x in ("drawer", "door"))
                else " knob"
            )
            pairs = [
                (f"the {goal.subject}{suffix}", last_before(boundary) or moves[0][1])
            ]
        elif goal.predicate == "in_front_of":
            pairs = [
                (
                    _semantic_phrase(goal.subject, goal.predicate, subject=True),
                    moves[0][1],
                )
            ]
        else:
            pick = picks[grasp_cursor] if grasp_cursor < len(picks) else len(plan)
            release = next((item for item in releases if item > pick), len(plan))
            subject_xy = last_before(pick) or moves[0][1]
            destination_xy = last_before(release, after=pick) or last_before(release)
            pairs = [
                (
                    _semantic_phrase(goal.subject, goal.predicate, subject=True),
                    subject_xy,
                )
            ]
            if goal.object:
                pairs.append(
                    (
                        _semantic_phrase(goal.object, goal.predicate, subject=False),
                        destination_xy,
                    )
                )
            grasp_cursor += 1
        for phrase, xy in pairs:
            if xy is None or any(anchor["phrase"] == phrase for anchor in anchors):
                continue
            anchors.append(
                {
                    "phrase": phrase,
                    "locator": "molmo",
                    "camera": "agentview",
                    "score": None,
                    "readings": [[xy[0], xy[1], 0.0]],
                    "median_xy": xy,
                    "z_top": 0.0,
                    "z_span": 0.0,
                }
            )
    return anchors


def _merge_anchors(
    measured: list[dict[str, Any]], inferred: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    anchors = list(measured)
    for candidate in inferred:
        if any(anchor["phrase"] == candidate["phrase"] for anchor in anchors):
            continue
        if any(
            math.dist(candidate["median_xy"], anchor["median_xy"]) < 0.03
            for anchor in anchors
        ):
            continue
        anchors.append(candidate)
    return anchors


def _attach_moves(
    plan: list[dict[str, Any]], anchors: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for raw in plan:
        entry: dict[str, Any] = {
            "action": raw["action"],
            "arguments": {key: value for key, value in raw.items() if key != "action"},
        }
        xy = _move_xy(raw)
        if xy is not None and anchors:
            anchor = min(anchors, key=lambda item: math.dist(xy, item["median_xy"]))
            distance = math.dist(xy, anchor["median_xy"])
            if distance <= 0.20:
                entry.update(
                    {
                        "anchor": anchor["phrase"],
                        "offset": [
                            round(xy[0] - anchor["median_xy"][0], 4),
                            round(xy[1] - anchor["median_xy"][1], 4),
                        ],
                        "anchor_distance": round(distance, 4),
                    }
                )
        output.append(entry)
    return output


def generate_task_card(
    audit_path: str | Path,
    recipe_path: str | Path,
    destination: str | Path,
    *,
    segments: str | Path | None = None,
) -> dict[str, Any]:
    """Generate and index exactly one card; never inspect another seed."""
    audit_path = Path(audit_path)
    recipe_path = Path(recipe_path)
    destination = Path(destination)
    identity = _identity(audit_path)
    expected_recipe = f"recipe_{identity.tag}.jsonl"
    if recipe_path.name != expected_recipe:
        raise ValueError(
            f"recipe filename {recipe_path.name!r} does not match audit; "
            f"expected {expected_recipe!r}"
        )
    audit = _read_audit(audit_path)
    expected_suite = f"libero_{identity.family}_{identity.suite}"
    for field, expected in (
        ("suite", expected_suite),
        ("task_id", identity.task),
        ("seed", identity.seed),
    ):
        if field in audit and audit[field] != expected:
            raise ValueError(
                f"audit {field}={audit[field]!r} does not match filename ({expected!r})"
            )
    language = _task_language(audit)
    graph = extract_relations(identity.family, language)
    raw_plan = _read_recipe(recipe_path)

    segment_dir = (
        Path(segments) if segments is not None else audit_path.parent / "segments"
    )
    anchors = _merge_anchors(
        _segment_anchors(segment_dir), _fallback_anchors(graph, raw_plan)
    )
    plan = _attach_moves(raw_plan, anchors)
    task_name = f"{identity.family}/{identity.key}"
    provenance = {
        "cell": identity.tag,
        "audit": str(audit_path),
        "recipe": str(recipe_path),
    }
    plan_doc = {
        "task": task_name,
        "language": language,
        "source": provenance,
        "goal_graph": graph.as_dict(),
        "plan": plan,
    }
    anchors_doc = {
        "task": task_name,
        "language": language,
        "source": provenance,
        "anchors": anchors,
        "prompts_without_readings": [],
    }
    attached = sum("anchor" in entry for entry in plan)
    moves = sum(entry["action"] in MOVE_ACTIONS for entry in plan)
    trace = (
        f"# {task_name}\n\nTask: {language}\n\nSource: `{identity.tag}`\n\n"
        f"Audit: `{audit_path}`\n\nRecipe: `{recipe_path}`\n\n"
        f"Goals: `{json.dumps(graph.as_dict()['goals'], ensure_ascii=False)}`\n\n"
        f"Anchors: {len(anchors)}\n\nActions: {len(plan)}\n\n"
        f"Anchored waypoints: {attached}/{moves}\n"
    )

    destination.mkdir(parents=True, exist_ok=True)
    card_name = f"{identity.family}_{identity.key}"
    (destination / f"{card_name}_plan.json").write_text(
        json.dumps(plan_doc, ensure_ascii=False, indent=2) + "\n"
    )
    (destination / f"{card_name}_anchors.json").write_text(
        json.dumps(anchors_doc, ensure_ascii=False, indent=2) + "\n"
    )
    (destination / f"{card_name}_trace.md").write_text(trace)

    row = {
        "family": identity.family,
        "suite": identity.suite,
        "task": identity.task,
        "key": identity.key,
        "source": identity.tag,
        "language": language,
        "anchors": len(anchors),
        "actions": len(plan),
    }
    return row


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit", required=True, type=Path)
    parser.add_argument("--recipe", required=True, type=Path)
    parser.add_argument("--destination", required=True, type=Path)
    parser.add_argument(
        "--segments",
        type=Path,
        help="optional directory of segment_*.json readings from this trace",
    )
    args = parser.parse_args()
    row = generate_task_card(
        args.audit, args.recipe, args.destination, segments=args.segments
    )
    print(json.dumps(row, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
