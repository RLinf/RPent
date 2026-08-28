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

import hashlib
import json
from pathlib import Path

import pytest

from rpent.reproduce.robocasa.memory import build_memory_pack, validate_memory_pack
from rpent.reproduce.robocasa.protocol import (
    ALLOWED_ACTIONS,
    EMPTY_MEMORY_TASKS,
    MEMORY_PAIR_TASKS,
)


def _source_tree(root: Path) -> None:
    from rpent.reproduce.robocasa.memory import SOURCE_DIRS, _tasks_for_directory

    for label, dirname in SOURCE_DIRS.items():
        directory = root / dirname
        directory.mkdir()
        for task in _tasks_for_directory(label) - EMPTY_MEMORY_TASKS:
            (directory / f"{task}_s0.json").write_text(
                json.dumps(
                    {
                        "task": task,
                        "seed": 0,
                        "success": True,
                        "task_language": f"do {task}",
                    }
                )
            )
            (directory / f"recipe_{task}_s0.jsonl").write_text(
                json.dumps({"action": "rldx_skill"}) + "\n"
            )


def test_pack_has_only_canonical_pairs_empty_whitelist_and_hashes(tmp_path: Path):
    source, destination = tmp_path / "migration", tmp_path / "pack"
    source.mkdir()
    _source_tree(source)
    manifest = build_memory_pack(source, destination)
    assert len(manifest["entries"]) == 50
    assert manifest["protocol"] == "robocasa-harness-vla-v1"
    assert manifest["source_directories"]["atomic"] == "explore_atomic_recipe"
    assert not validate_memory_pack(destination)
    assert sum(
        entry["memory"] == "seed0_success" for entry in manifest["entries"]
    ) == len(MEMORY_PAIR_TASKS)
    assert all(not list((destination / task).iterdir()) for task in EMPTY_MEMORY_TASKS)
    trace = destination / "OpenDrawer" / "OpenDrawer_s0.jsonl"
    assert json.loads(trace.read_text())["action"] == "vla_act"
    assert not any(path.name.endswith(".md") for path in destination.rglob("*"))


def test_pack_expands_legacy_commands_to_the_frozen_action_surface(tmp_path: Path):
    source, destination = tmp_path / "migration", tmp_path / "pack"
    source.mkdir()
    _source_tree(source)
    trace = (
        source / "exploration_unseen_recipe" / "recipe_RecycleBottlesByType_s0.jsonl"
    )
    trace.write_text(
        "\n".join(
            [
                '{"action":"set_gripper","width":1.0}',
                '{"action":"scripted_grasp","xyz":[1.0,2.0,3.0],"approach_z":0.1}',
            ]
        )
        + "\n"
    )
    build_memory_pack(source, destination)
    packed = destination / "RecycleBottlesByType" / "RecycleBottlesByType_s0.jsonl"
    commands = [json.loads(line) for line in packed.read_text().splitlines()]
    assert len(commands) == 6
    assert {command["action"] for command in commands} <= ALLOWED_ACTIONS
    assert commands[0] == {"action": "set_gripper", "gripper": 1.0}
    assert [command["action"] for command in commands[1:]] == [
        "set_gripper",
        "move_to",
        "move_to",
        "set_gripper",
        "move_to",
    ]
    assert not validate_memory_pack(destination)


def test_pack_removes_embedded_audit_commands_and_normalizes_stats(tmp_path: Path):
    source, destination = tmp_path / "migration", tmp_path / "pack"
    source.mkdir()
    _source_tree(source)
    audit_path = source / "exploration_seen_recipe" / "SearingMeat_s0.json"
    audit = json.loads(audit_path.read_text())
    audit.update(
        {
            "command_sequence": [{"action": "set_gripper", "width": 1.0}],
            "n_commands": 99,
            "manual_cmds": 99,
            "vla_calls": 99,
        }
    )
    audit_path.write_text(json.dumps(audit))

    build_memory_pack(source, destination)

    packed = json.loads(
        (destination / "SearingMeat" / "SearingMeat_s0.json").read_text()
    )
    assert "command_sequence" not in packed
    assert packed["n_commands"] == 1
    assert packed["manual_cmds"] == 0
    assert packed["vla_calls"] == 1
    assert not validate_memory_pack(destination)


def test_pack_rejects_unsuccessful_memory_and_detects_tampering(tmp_path: Path):
    source = tmp_path / "migration"
    source.mkdir()
    _source_tree(source)
    audit = source / "explore_atomic_recipe" / "OpenDrawer_s0.json"
    payload = json.loads(audit.read_text())
    payload["success"] = False
    audit.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="not a successful"):
        build_memory_pack(source, tmp_path / "bad")
    payload["success"] = True
    audit.write_text(json.dumps(payload))
    pack = tmp_path / "good"
    build_memory_pack(source, pack)
    (pack / "OpenDrawer" / "OpenDrawer_s0.jsonl").write_text("tampered\n")
    assert any("hash mismatch" in problem for problem in validate_memory_pack(pack))


def test_memory_validation_reparses_semantics_even_with_updated_hash(tmp_path: Path):
    source, pack = tmp_path / "migration", tmp_path / "pack"
    source.mkdir()
    _source_tree(source)
    build_memory_pack(source, pack)
    trace = pack / "OpenDrawer" / "OpenDrawer_s0.jsonl"
    trace.write_text('{"action":"vla_act","prompt":"wrong"}\n')
    manifest_path = pack / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    entry = next(item for item in manifest["entries"] if item["task"] == "OpenDrawer")
    entry["files"][trace.name] = hashlib.sha256(trace.read_bytes()).hexdigest()
    manifest_path.write_text(json.dumps(manifest))
    assert any(
        "invalid memory semantics" in problem for problem in validate_memory_pack(pack)
    )


def test_memory_validation_rejects_embedded_commands_with_updated_hash(tmp_path: Path):
    source, pack = tmp_path / "migration", tmp_path / "pack"
    source.mkdir()
    _source_tree(source)
    build_memory_pack(source, pack)
    audit_path = pack / "OpenDrawer" / "OpenDrawer_s0.json"
    audit = json.loads(audit_path.read_text())
    audit["command_sequence"] = [{"action": "release"}]
    audit_path.write_text(json.dumps(audit))
    manifest_path = pack / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    entry = next(item for item in manifest["entries"] if item["task"] == "OpenDrawer")
    entry["files"][audit_path.name] = hashlib.sha256(
        audit_path.read_bytes()
    ).hexdigest()
    manifest_path.write_text(json.dumps(manifest))

    assert any("embeds commands" in problem for problem in validate_memory_pack(pack))


def test_memory_pack_rejects_invalid_commands_and_extra_files(tmp_path: Path):
    source = tmp_path / "migration"
    source.mkdir()
    _source_tree(source)
    trace = source / "explore_atomic_recipe" / "recipe_OpenDrawer_s0.jsonl"
    trace.write_text('{"arguments":{}}\n')
    with pytest.raises(ValueError, match="invalid command"):
        build_memory_pack(source, tmp_path / "bad")
    trace.write_text('{"action":"release"}\n')
    pack = tmp_path / "good"
    build_memory_pack(source, pack)
    (pack / "TASK_MEMORY.md").write_text("forbidden")
    assert any(
        "unexpected pack entries" in problem for problem in validate_memory_pack(pack)
    )


@pytest.mark.parametrize(
    "command",
    [
        {"action": "scripted_grasp", "xyz": [0, 0, 0]},
        {"action": "move_to", "target": [0, 0, 0]},
        {"action": "move_to", "xyz": [0, 0, float("nan")]},
        {"action": "release", "steps": 0},
        {"action": "vla_act", "prompt": "wrong"},
    ],
)
def test_memory_validation_rejects_noncanonical_command_schemas(
    tmp_path: Path, command
):
    source, pack = tmp_path / "migration", tmp_path / "pack"
    source.mkdir()
    _source_tree(source)
    build_memory_pack(source, pack)
    trace = pack / "OpenDrawer" / "OpenDrawer_s0.jsonl"
    trace.write_text(json.dumps(command) + "\n")
    manifest_path = pack / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    entry = next(item for item in manifest["entries"] if item["task"] == "OpenDrawer")
    entry["files"][trace.name] = hashlib.sha256(trace.read_bytes()).hexdigest()
    manifest_path.write_text(json.dumps(manifest))
    assert any(
        "invalid memory semantics" in problem for problem in validate_memory_pack(pack)
    )
