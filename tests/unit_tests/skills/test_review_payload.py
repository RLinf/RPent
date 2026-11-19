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

import json
import re
import runpy
from pathlib import Path
from unittest.mock import Mock

import pytest

SCRIPT = Path(__file__).resolve().parents[3] / ".agents/skills/review-pr/lint_review.py"
review = runpy.run_path(str(SCRIPT))
BODY = """\
**Changes requested**

The worker stays open after a failed start.

### Reproduction

The offline startup test leaves one worker running.

### Findings

#### F2 · major · correctness · Failed startup leaks a worker

The error path at `x.py:2` returns before closing the worker.
"""


@pytest.mark.parametrize(
    ("payload", "error"),
    [
        ([], "payload: must be a JSON object"),
        ({"commit_id": None}, "commit_id"),
        ({"comments": None}, "comments: must be a list"),
        ({"comments": ["text"]}, "comment 1: must be an object"),
        ({"body": ["paragraph"]}, "body: must be a string"),
        (
            {"comments": [{"path": "x.py", "line": 1, "body": [1]}]},
            "comment 1 (x.py:1): body must be a non-empty string",
        ),
        (
            {
                "body": BODY.replace(
                    "**Changes requested**", "Blocked workers need a timeout."
                )
            },
            "start with one recommendation",
        ),
        ({"body": BODY.replace("### Reproduction", "### Setup")}, "### Reproduction"),
        (
            {
                "body": BODY
                + "\n#### F2 · minor · docs · Missing example\n\nSee the guide."
            },
            "duplicate finding ID F2",
        ),
    ],
)
def test_lint_rejects_invalid_payload(tmp_path, capsys, payload, error):
    path = tmp_path / "review.json"
    if isinstance(payload, dict):
        payload = {"body": BODY, **payload}
    path.write_text(json.dumps(payload))
    assert review["main"](["lint", str(path)]) == 1
    assert error in capsys.readouterr().out


@pytest.mark.parametrize(
    ("fields", "errors"),
    [
        ({"start_line": 1, "side": "LEFT"}, ["multi-line comment missing start_side"]),
        ({"start_line": 1, "start_side": "RIGHT"}, ["multi-line comment missing side"]),
        ({"start_line": 1, "side": "LEFT", "start_side": "LEFT"}, []),
    ],
)
def test_multiline_comment_requires_both_sides(fields, errors):
    comment = {"path": "x.py", "line": 2, "body": "Suggested fix: Use x.", **fields}
    findings = review["ReviewLinter"]({"body": BODY, "comments": [comment]}, None).run()
    assert [f.message for f in findings if f.level == "ERROR"] == errors


def test_template_can_be_filled_and_linted(tmp_path, capsys):
    assert review["main"](["template"]) == 0
    template = capsys.readouterr().out
    answers = iter(
        [
            "**Ready for senior review**",
            "Fixes a typo.",
            "Not needed for a typo.",
            "No findings.",
        ]
    )
    body = re.sub(r"<!--.*?-->", lambda _: next(answers), template)
    path = tmp_path / "review.json"
    path.write_text(json.dumps({"body": body}))
    assert review["main"](["lint", str(path)]) == 0


@pytest.mark.parametrize(
    ("fields", "status"),
    [
        ({}, 0),
        ({"commit_id": "A" * 40, "comments": []}, 0),
        ({"commit_id": "b" * 40}, 1),
    ],
)
def test_post_uses_reviewed_commit(tmp_path, monkeypatch, fields, status):
    path = tmp_path / "review.json"
    path.write_text(json.dumps({"body": BODY, **fields}))
    head = "a" * 40
    post = Mock()
    monkeypatch.setattr(review["PullRequest"], "head_sha", head)
    monkeypatch.setattr(review["PullRequest"], "post", post)

    assert review["main"](["post", "--pr", "364", str(path)]) == status
    if status:
        post.assert_not_called()
    else:
        post.assert_called_once_with(
            {"body": BODY, "event": "COMMENT", **fields, "commit_id": head}
        )
