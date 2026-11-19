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
import runpy
from pathlib import Path
from unittest.mock import Mock

import pytest

SCRIPT = Path(__file__).resolve().parents[3] / ".agents/skills/review-pr/lint_review.py"
review = runpy.run_path(str(SCRIPT))
BODY = (
    "\n\n".join(
        f"### {i}. {q}\n\nChecked." for i, q in enumerate(review["QUESTIONS"], 1)
    )
    + "\n\nReady for senior review"
)


@pytest.mark.parametrize(
    ("payload", "error"),
    [
        ([], "payload: must be a JSON object"),
        ({"commit_id": "<PR head SHA>"}, "commit_id"),
        ({"comments": None}, "comments: must be a list"),
        ({"comments": 3}, "comments: must be a list"),
        ({"comments": ["text"]}, "comment 1: must be an object"),
        ({"body": ["paragraph"]}, "body: must be a string"),
        (
            {"comments": [{"path": "x.py", "line": 1, "body": [1]}]},
            "comment 1 (x.py:1): body must be a non-empty string",
        ),
    ],
)
def test_lint_reports_invalid_field(tmp_path, capsys, payload, error):
    path = tmp_path / "review.json"
    if isinstance(payload, dict):
        payload = {"body": BODY, **payload}
    path.write_text(json.dumps(payload))
    assert review["main"](["lint", str(path)]) == 1
    assert error in capsys.readouterr().out


@pytest.mark.parametrize(
    ("answer", "levels"),
    [
        ("- **Changes requested**: fix the blocked shutdown path.", []),
        ("Merge is not blocked.", ["ERROR"]),
        ("Blocked workers need a timeout.", ["ERROR"]),
        ("- Changes requested. Fix the tests.\n- Blocked.", ["WARN"]),
    ],
)
def test_recommendation_labels(answer, levels):
    body = BODY.removesuffix("Ready for senior review") + answer
    findings = review["ReviewLinter"]({"body": body}, None).run()
    assert [finding.level for finding in findings] == levels


@pytest.mark.parametrize(
    ("fields", "status"),
    [
        ({}, 0),
        ({"commit_id": "A" * 40, "comments": []}, 0),
        ({"commit_id": "b" * 40}, 1),
        ({"commit_id": None}, 1),
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
