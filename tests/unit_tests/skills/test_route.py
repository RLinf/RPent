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

import runpy
import subprocess
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
ROUTE_SCRIPT = REPO_ROOT / ".agents/skills/route.py"
route = runpy.run_path(str(ROUTE_SCRIPT))


def test_load_areas_discovers_all_rpent_area_skills():
    areas = {a.name: a for a in route["load_areas"]()}
    assert set(areas) == {"code", "docs", "robots"}
    assert areas["code"].owns("rpent/cli/main.py")
    assert areas["code"].owns("tests/unit_tests/skills/test_route.py")
    assert areas["docs"].owns("docs/source-en/rst_source/simulators/libero.rst")
    assert areas["docs"].owns("README.zh-CN.md")
    assert areas["docs"].owns("CONTRIBUTING.md")
    assert areas["robots"].owns("robots/libero/robot_spec.py")
    assert areas["robots"].owns("pyproject.toml")
    assert areas["robots"].owns(".github/workflows/embodied-gpu-e2e.yml")
    assert not areas["docs"].owns("rpent/cli/main.py")


@pytest.mark.parametrize(
    ("pattern", "path", "expected"),
    [
        ("rpent/**", "rpent/cli/main.py", True),
        ("rpent/**", "robots/libero/robot_spec.py", False),
        ("**/*.py", "top.py", True),
        ("**/*.py", "a/b/c.py", True),
        ("*.py", "a/b.py", False),
        (
            ".github/workflows/embodied-gpu-e2e*.yml",
            ".github/workflows/embodied-gpu-e2e-extra.yml",
            True,
        ),
        ("docs/source-?/index.rst", "docs/source-e/index.rst", True),
    ],
)
def test_glob_regex_matches_single_and_recursive_globs(pattern, path, expected):
    assert bool(route["glob_regex"](pattern).match(path)) is expected


def test_finding_validation_and_report_exit_codes(capsys):
    Finding = route["Finding"]
    with pytest.raises(ValueError, match="severity must be one of"):
        Finding("invalid", "x.py", 1, "code/x", "msg")

    minor = Finding("minor", "x.py", 10, "code/style", "minor note")
    major = Finding("major", "a.py", None, "code/init", "missing init")
    assert route["report"]("code", [minor]) == 0
    assert "code: 1 minor" in capsys.readouterr().out

    assert route["report"]("code", [minor, major, major]) == 1
    out = capsys.readouterr().out
    lines = out.strip().splitlines()
    # Most severe first and deduplicated.
    assert lines[0] == "major a.py code/init: missing init"
    assert lines[1] == "minor x.py:10 code/style: minor note"
    assert lines[2] == "code: 1 major, 1 minor"


def test_selection_filters_modified_lines_but_keeps_added_files(tmp_path, monkeypatch):
    subprocess.run(
        ["git", "init", "-b", "main"], cwd=tmp_path, check=True, capture_output=True
    )
    subprocess.run(
        ["git", "config", "user.email", "test@example.com"], cwd=tmp_path, check=True
    )
    subprocess.run(["git", "config", "user.name", "Test"], cwd=tmp_path, check=True)
    tracked = tmp_path / "mod.py"
    tracked.write_text("line1\nline2\n", encoding="utf-8")
    subprocess.run(["git", "add", "mod.py"], cwd=tmp_path, check=True)
    subprocess.run(
        ["git", "commit", "-m", "init"], cwd=tmp_path, check=True, capture_output=True
    )

    tracked.write_text("line1\nline2\nline3\n", encoding="utf-8")
    (tmp_path / "new.py").write_text("new_line\n", encoding="utf-8")

    monkeypatch.setitem(route["git"].__globals__, "REPO_ROOT", tmp_path)
    route["_added_lines_for"].cache_clear()

    Change = route["Change"]
    Finding = route["Finding"]
    Selection = route["Selection"]
    sel = Selection("main", (Change("M", "mod.py"), Change("A", "new.py")), False)
    findings = [
        Finding("minor", "mod.py", 1, "r/old", "old line"),
        Finding("minor", "mod.py", 3, "r/new", "added line"),
        Finding("minor", "new.py", 1, "r/added", "untracked file"),
        Finding("major", "mod.py", None, "r/file", "file-level"),
        Finding("major", "untouched.py", 5, "r/untouched-line", "pre-existing line"),
        Finding("major", "untouched.py", None, "r/untouched-file", "package-level"),
    ]
    kept = sel.relevant(findings)
    assert [f.rule for f in kept] == [
        "r/new",
        "r/added",
        "r/file",
        "r/untouched-file",
    ]

    all_sel = Selection("main", (Change("M", "mod.py"), Change("A", "new.py")), True)
    assert len(all_sel.relevant(findings)) == len(findings)


def test_selection_tracks_renames_without_flagging_unchanged_lines(
    tmp_path, monkeypatch
):
    subprocess.run(
        ["git", "init", "-b", "main"], cwd=tmp_path, check=True, capture_output=True
    )
    subprocess.run(
        ["git", "config", "user.email", "test@example.com"], cwd=tmp_path, check=True
    )
    subprocess.run(["git", "config", "user.name", "Test"], cwd=tmp_path, check=True)
    old_doc = tmp_path / "docs" / "old.rst"
    old_doc.parent.mkdir(parents=True)
    old_doc.write_text(
        "Title\n=====\n\nLine four.\nLine five.\nLine six.\n", encoding="utf-8"
    )
    subprocess.run(["git", "add", "docs/old.rst"], cwd=tmp_path, check=True)
    subprocess.run(
        ["git", "commit", "-m", "init"], cwd=tmp_path, check=True, capture_output=True
    )

    subprocess.run(
        ["git", "mv", "docs/old.rst", "docs/new.rst"], cwd=tmp_path, check=True
    )
    monkeypatch.setitem(route["git"].__globals__, "REPO_ROOT", tmp_path)
    route["_added_lines_for"].cache_clear()

    changed = route["changes"]("main")
    assert changed == [route["Change"]("R", "docs/new.rst", "docs/old.rst")]
    assert route["added_lines"]("main", "docs/new.rst", "docs/old.rst") == frozenset()

    Finding = route["Finding"]
    sel = route["Selection"]("main", tuple(changed), False)
    preexisting = [Finding("major", "docs/new.rst", 4, "docs/code-name", "old line")]
    assert sel.relevant(preexisting) == []

    # Adding one line to the moved file flags only the added line, including via select().
    new_doc = tmp_path / "docs" / "new.rst"
    new_doc.write_text(
        "Title\n=====\n\nLine four.\nLine five.\nLine six.\nLine seven.\n",
        encoding="utf-8",
    )
    route["_added_lines_for"].cache_clear()
    args = route["check_parser"]("test").parse_args(["--base", "main", str(new_doc)])
    selected = route["select"]("docs", args)
    assert selected.changes == (route["Change"]("R", "docs/new.rst", "docs/old.rst"),)
    kept = selected.relevant(
        [
            Finding("major", "docs/new.rst", 4, "docs/code-name", "old line"),
            Finding("major", "docs/new.rst", 7, "docs/code-name", "added line"),
        ]
    )
    assert [f.line for f in kept] == [7]
