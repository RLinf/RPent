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

"""Every component must be installed by an environment on some full-CI runner."""

import json

from rpent.install.discovery import discover
from tests.tooling import matrix


def _write_project(root):
    """Write the checkout-level pyproject.toml the resolver reads."""
    (root / "pyproject.toml").write_text(
        '[project]\nname = "rpent"\ndependencies = []\n', encoding="utf-8"
    )


def _write_component(root, subject, *, requires=()):
    path = root / "robots" / subject / "component.toml"
    path.parent.mkdir(parents=True, exist_ok=True)
    text = f'schema = 2\nname = "robot-{subject}"\nprobe = "json"\n'
    if requires:
        listed = ", ".join(f'"{name}"' for name in requires)
        text += f"requires = [{listed}]\n"
    path.write_text(text, encoding="utf-8")


def test_component_without_an_environment_is_reported(tmp_path):
    _write_component(tmp_path, "alpha")

    problems = matrix.problems(discover(tmp_path), {"hosted": (), "gpu": ()})

    assert problems == [
        "component 'robot-alpha' is not installed by any environment; add it to "
        "HOSTED or GPU in tests/tooling/matrix.py"
    ]


def test_stale_environment_is_reported(tmp_path):
    _write_component(tmp_path, "alpha")
    _write_project(tmp_path)

    problems = matrix.problems(
        discover(tmp_path),
        {"hosted": (("robot-alpha",), ("robot-removed",)), "gpu": ()},
    )

    assert problems == [
        "hosted lists environment 'robot-removed', but robot-removed does not "
        "exist; remove it from tests/tooling/matrix.py"
    ]


def test_environment_listed_by_two_runners_is_reported(tmp_path):
    _write_component(tmp_path, "alpha")
    _write_project(tmp_path)

    problems = matrix.problems(
        discover(tmp_path),
        {"hosted": (("robot-alpha",),), "gpu": (("robot-alpha",),)},
    )

    assert problems == [
        "gpu lists environment 'robot-alpha' again; hosted already installs it"
    ]


def test_environment_listed_twice_is_reported(tmp_path):
    _write_component(tmp_path, "alpha")
    _write_project(tmp_path)

    problems = matrix.problems(
        discover(tmp_path),
        {"hosted": (("robot-alpha",), ("robot-alpha",)), "gpu": ()},
    )

    assert problems == [
        "hosted lists environment 'robot-alpha' again; hosted already installs it"
    ]


def test_empty_environment_is_reported(tmp_path):
    problems = matrix.problems(discover(tmp_path), {"hosted": ((),), "gpu": ()})

    assert problems == ["hosted lists an empty environment"]


def test_a_required_component_needs_no_environment_of_its_own(tmp_path):
    path = tmp_path / "components/models/sam3/component.toml"
    path.parent.mkdir(parents=True)
    path.write_text(
        'schema = 2\nname = "model-sam3"\nprobe = "json"\n', encoding="utf-8"
    )
    _write_component(tmp_path, "alpha", requires=("model-sam3",))
    _write_project(tmp_path)

    problems = matrix.problems(
        discover(tmp_path), {"hosted": (), "gpu": (("robot-alpha",),)}
    )

    assert problems == []


def test_an_environment_that_cannot_resolve_is_reported(tmp_path):
    _write_project(tmp_path)
    for subject, other in (("alpha", "robot-beta"), ("beta", "robot-alpha")):
        path = tmp_path / f"robots/{subject}/component.toml"
        path.parent.mkdir(parents=True)
        path.write_text(
            f'schema = 2\nname = "robot-{subject}"\nprobe = "json"\n'
            f'env_conflicts = ["{other}"]\n',
            encoding="utf-8",
        )

    problems = matrix.problems(
        discover(tmp_path),
        {"hosted": (), "gpu": (("robot-alpha", "robot-beta"),)},
    )

    assert "cannot be installed" in problems[0]


def test_check_reports_drift_and_fails(tmp_path, capsys):
    _write_component(tmp_path, "alpha")

    assert matrix.main(["--check"], root=tmp_path) == 1
    assert "not installed by any environment" in capsys.readouterr().err


def test_check_passes_when_the_partition_matches(tmp_path, monkeypatch, capsys):
    _write_component(tmp_path, "alpha")
    _write_project(tmp_path)
    monkeypatch.setattr(matrix, "MATRICES", {"hosted": (("robot-alpha",),), "gpu": ()})

    assert matrix.main(["--check"], root=tmp_path) == 0
    assert capsys.readouterr().err == ""


def test_runner_prints_the_selected_partition(monkeypatch, capsys):
    monkeypatch.setattr(
        matrix,
        "MATRICES",
        {"hosted": (("robot-alpha",), ("robot-alpha", "tool-x")), "gpu": ()},
    )

    assert matrix.main(["--runner", "hosted"]) == 0

    assert json.loads(capsys.readouterr().out) == [
        ["robot-alpha"],
        ["robot-alpha", "tool-x"],
    ]
