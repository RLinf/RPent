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

from pathlib import Path

import pytest

from robots.franka.prompt_bundle import user_prompt
from rpent.prompt.compose import COMMON, compose


def test_compose_binds_each_template_once_in_selection_order(tmp_path: Path) -> None:
    first = tmp_path / "first.md"
    second = tmp_path / "second.md"
    first.write_text("# Task\n\n{{ instruction }}\n", encoding="utf-8")
    second.write_text("\nSeed: {{seed}}\n", encoding="utf-8")

    result = compose(
        [first, second], {"instruction": "Keep {{seed}} and {braces} 原样", "seed": 7}
    )

    assert result == "# Task\n\nKeep {{seed}} and {braces} 原样\n\nSeed: 7\n"
    assert "Seed: 8" in compose([second], {"seed": 8})


def test_missing_variable_identifies_owning_file_and_name(tmp_path: Path) -> None:
    path = tmp_path / "policy.md"
    path.write_text("Task: {{instruction}}", encoding="utf-8")
    with pytest.raises(KeyError) as exc:
        compose([path], {})
    assert str(path) in str(exc.value)
    assert "instruction" in str(exc.value)


def test_missing_resource_identifies_file(tmp_path: Path) -> None:
    path = tmp_path / "missing.md"
    with pytest.raises(FileNotFoundError) as exc:
        compose([path])
    assert exc.value.filename == str(path)


def test_packaged_resources_and_user_values_ignore_cwd(tmp_path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    assert "tool call" in compose([COMMON / "base.md"])
    message = user_prompt(
        {
            "instruction": "Move {{literal}}",
            "success_criteria": "Keep {x, y} unchanged",
            "constraints": "Observe after each action",
            "task_name": "task-9",
        }
    )
    assert "Move {{literal}}" in message
    assert "Keep {x, y} unchanged" in message
    assert "task-9" in message
