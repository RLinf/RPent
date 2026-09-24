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

from __future__ import annotations

from pathlib import Path

import pytest
from pydantic_ai import BinaryContent


def test_plain_context_preserves_rendered_prompts_exactly() -> None:
    from rpent.context import assemble_context

    context = assemble_context(prompt="Rules\n\n", query="  Operator task\n")

    assert context.system_prompt == "Rules\n\n"
    assert context.user_message == "  Operator task\n"
    assert context.query == "  Operator task\n"


def test_memory_is_reference_context_and_skills_are_instructions() -> None:
    from rpent.context import ContextDocument, assemble_context

    memory = ContextDocument("grasp", "Approach from above.", "memory/grasp.md")
    skill = ContextDocument("pick", "Inspect after motion.\n", "skills/pick/SKILL.md")
    context = assemble_context(
        prompt="Use the robot tools.\n",
        query="Place the block.",
        memory=[memory],
        skills=[skill],
    )

    assert context.system_prompt == (
        "Use the robot tools.\n\n\n## Skill: pick\n\nInspect after motion.\n"
    )
    assert context.user_message == (
        "Place the block.\n\n## Memory: grasp\n"
        "Source: memory/grasp.md\n\nApproach from above."
    )
    assert context.memory == (memory,)
    assert context.skills == (skill,)
    assert "Approach from above." not in context.system_prompt


@pytest.mark.parametrize("name", ["SKILL.md", "camera-guide.md"])
def test_skill_loads_fresh_utf8_content_with_provenance(
    tmp_path: Path, name: str
) -> None:
    from rpent.context import assemble_context, load_skill

    skill = tmp_path / "相机操作" / name
    skill.parent.mkdir()
    skill.write_text("观察当前画面。\n", encoding="utf-8")
    first = load_skill(skill)
    skill.write_text("动作后再次观察。\n", encoding="utf-8")
    second = load_skill(str(skill))

    assert first.title == (skill.parent.name if name == "SKILL.md" else skill.stem)
    assert first.text == "观察当前画面。\n"
    assert second.text == "动作后再次观察。\n"
    assert first.source == str(skill.resolve())
    assert assemble_context(
        prompt="rules", query="task", skills=[first]
    ).system_prompt == (f"rules\n\n## Skill: {first.title}\n\n观察当前画面。\n")


def test_missing_skill_preserves_the_file_error(tmp_path: Path) -> None:
    from rpent.context import load_skill

    path = tmp_path / "missing" / "SKILL.md"
    with pytest.raises(FileNotFoundError) as exc:
        load_skill(path)
    assert exc.value.filename == str(path)


def test_assembly_does_not_read_memory_sources(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from rpent.context import ContextDocument, assemble_context

    def reject_read(*args, **kwargs):
        raise AssertionError("assembly must not read source paths")

    monkeypatch.setattr(Path, "read_text", reject_read)
    context = assemble_context(
        prompt="rules",
        query="task",
        memory=[ContextDocument("selected", "Approved excerpt.", "/unread/source.md")],
    )
    assert "Approved excerpt." in context.user_message


def test_multimodal_context_keeps_query_prefix_and_observation_order() -> None:
    from rpent.context import ContextDocument, assemble_context

    image = BinaryContent(data=b"png", media_type="image/png")
    context = assemble_context(
        prompt="rules",
        query="task",
        memory=[ContextDocument("prior", "A previous grasp failed.")],
        initial_context=["Front camera at step 3", image, "Gripper open"],
    )

    assert context.user_message == [
        "task\n\n## Memory: prior\n\nA previous grasp failed.",
        "Front camera at step 3",
        image,
        "Gripper open",
    ]
    assert context.user_message[2] is image


def test_assembly_snapshots_collections_and_returns_fresh_planner_lists() -> None:
    from rpent.context import ContextDocument, assemble_context

    memory = [ContextDocument("prior", "Retain this.")]
    skills = [ContextDocument("inspect", "Use the camera.")]
    observations = ["State at step 0"]
    context = assemble_context(
        prompt="rules",
        query="task",
        memory=memory,
        skills=skills,
        initial_context=observations,
    )
    memory.clear()
    skills.clear()
    observations.append("A different episode")
    first = context.user_message
    first.append("planner-local mutation")

    assert context.user_message == [
        "task\n\n## Memory: prior\n\nRetain this.",
        "State at step 0",
    ]
    assert "Use the camera." in context.system_prompt


@pytest.mark.parametrize("value", [b"raw image bytes", {"text": "unknown block"}, 3])
def test_assembly_rejects_unsupported_observation_parts(value) -> None:
    from rpent.context import assemble_context

    with pytest.raises(
        TypeError, match="initial_context parts must be text or BinaryContent"
    ):
        assemble_context(prompt="rules", query="task", initial_context=[value])


@pytest.mark.parametrize("field", ["memory", "skills"])
def test_assembly_requires_resolved_documents(field: str) -> None:
    from rpent.context import assemble_context

    with pytest.raises(TypeError, match=f"{field} entries must be ContextDocument"):
        assemble_context(prompt="rules", query="task", **{field: ["file.md"]})
