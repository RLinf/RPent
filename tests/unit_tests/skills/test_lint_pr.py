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
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[3] / ".agents/skills/create-pr/lint_pr.py"
lint_pr = runpy.run_path(str(SCRIPT))


@pytest.mark.parametrize(
    ("title", "expected_codes"),
    [
        ("fix(toolkit): retain the original handler error", []),
        ("feat: add support for custom MCP servers", []),
        ("Added support for custom MCP servers", ["title-format"]),
        (
            "feat(toolkit): Added support for custom MCP servers",
            ["title-capital", "title-mood"],
        ),
        ("feat(toolkit): add support for custom MCP servers.", ["title-period"]),
        (
            "feat(toolkit): " + "a" * 51,
            ["title-desc-long"],
        ),
        ("feat(): add support", ["title-scope-empty"]),
    ],
)
def test_check_title_enforces_conventional_commits(title, expected_codes):
    findings = lint_pr["check_title"](title)
    errors = [f.code for f in findings if f.level == "ERROR"]
    assert errors == expected_codes


def test_template_and_check_body(tmp_path, capsys):
    assert lint_pr["main"](["template"]) == 0
    skeleton = capsys.readouterr().out

    # Unticked checklist in template must fail until at least one box is ticked.
    findings = lint_pr["check_body"](skeleton)
    assert any(f.code == "checkbox-unticked" and f.level == "ERROR" for f in findings)

    filled = skeleton.replace("- [ ] My code follows", "- [x] My code follows", 1)
    body_path = tmp_path / "body.md"
    body_path.write_text(filled, encoding="utf-8")
    assert (
        lint_pr["main"](
            [
                "lint",
                "--title",
                "docs(skills): add create-pr and rst markup linters",
                "--body-file",
                str(body_path),
            ]
        )
        == 0
    )


def test_fix_body_deindents_uniform_paste_offset():
    indented = (
        "    ## Description\n"
        "\n"
        "    Add a new helper.\n"
        "\n"
        "    ## Testing\n"
        "\n"
        "    - `pytest tests/unit_tests -v`\n"
        "\n"
        "    ## Manual verification\n"
        "\n"
        "    Not needed\n"
        "\n"
        "    ## Checklist\n"
        "\n"
        "    - [x] My code follows the code style of this project.\n"
    )
    assert any(f.level == "ERROR" for f in lint_pr["check_indentation"](indented))
    fixed = lint_pr["fix_body"](indented)
    assert lint_pr["check_body"](fixed) == []
